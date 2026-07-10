//! D-Log tonemapping (doc/05 §1.1): Adobe/Resolve `.cube` 3D-LUT parsing
//! and trilinear application, plus a parametric D-Log → Rec.709 fallback
//! for when no official LUT is supplied. Video frames only — photos are
//! already display-referred.

use std::path::Path;

use crate::error::CaptureError;

/// 3D LUT parsed from an Adobe/Resolve `.cube` file.
#[derive(Debug, Clone)]
pub struct Lut3d {
    size: usize,
    /// Lattice values, red-fastest: `data[r + g·size + b·size²]`.
    data: Vec<[f32; 3]>,
    domain_min: [f32; 3],
    domain_max: [f32; 3],
}

impl Lut3d {
    /// Parse `.cube` text. `TITLE`, `#` comments, CRLF endings, and unknown
    /// keywords are tolerated; `LUT_1D_SIZE` is rejected (3D-only pipeline);
    /// `LUT_3D_SIZE` must be in 2..=129 and the data row count must match
    /// exactly.
    pub fn parse_cube(path: &Path, text: &str) -> Result<Lut3d, CaptureError> {
        let err =
            |reason: String| CaptureError::Lut { path: path.to_owned(), reason };
        let mut size: Option<usize> = None;
        let mut domain_min = [0.0f32; 3];
        let mut domain_max = [1.0f32; 3];
        let mut data: Vec<[f32; 3]> = Vec::new();

        for (ln, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let tokens: Vec<&str> = line.split_whitespace().collect();
            if tokens[0].starts_with(|c: char| c.is_ascii_alphabetic()) {
                match tokens[0] {
                    "TITLE" => {}
                    "LUT_1D_SIZE" => {
                        return Err(err("LUT_1D_SIZE: only 3D LUTs are supported".into()));
                    }
                    "LUT_3D_SIZE" => {
                        let n: usize = tokens
                            .get(1)
                            .and_then(|t| t.parse().ok())
                            .ok_or_else(|| err(format!("line {}: bad LUT_3D_SIZE", ln + 1)))?;
                        if !(2..=129).contains(&n) {
                            return Err(err(format!("LUT_3D_SIZE {n} outside 2..=129")));
                        }
                        data.reserve(n * n * n);
                        size = Some(n);
                    }
                    "DOMAIN_MIN" | "DOMAIN_MAX" => {
                        let mut v = [0.0f32; 3];
                        for (c, slot) in v.iter_mut().enumerate() {
                            *slot = tokens
                                .get(c + 1)
                                .and_then(|t| t.parse().ok())
                                .ok_or_else(|| {
                                    err(format!("line {}: bad {}", ln + 1, tokens[0]))
                                })?;
                        }
                        if tokens[0] == "DOMAIN_MIN" {
                            domain_min = v;
                        } else {
                            domain_max = v;
                        }
                    }
                    // Resolve/Adobe emit extra keywords (LUT_IN_VIDEO_RANGE …)
                    _ => {}
                }
                continue;
            }
            if tokens.len() != 3 {
                return Err(err(format!("line {}: expected 3 values, got {}", ln + 1, tokens.len())));
            }
            let mut v = [0.0f32; 3];
            for (c, slot) in v.iter_mut().enumerate() {
                *slot = tokens[c]
                    .parse()
                    .map_err(|_| err(format!("line {}: bad data row {line:?}", ln + 1)))?;
            }
            data.push(v);
        }

        let size = size.ok_or_else(|| err("missing LUT_3D_SIZE".into()))?;
        if data.len() != size * size * size {
            return Err(err(format!("{} data rows, expected {}", data.len(), size * size * size)));
        }
        for c in 0..3 {
            if domain_max[c] <= domain_min[c] {
                return Err(err(format!(
                    "degenerate domain [{}, {}] on channel {c}",
                    domain_min[c], domain_max[c]
                )));
            }
        }
        Ok(Lut3d { size, data, domain_min, domain_max })
    }

    /// Trilinear lookup; input clamped to the domain box.
    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        let n = self.size;
        let mut i0 = [0usize; 3];
        let mut i1 = [0usize; 3];
        let mut f = [0.0f32; 3];
        for c in 0..3 {
            let t = ((rgb[c] - self.domain_min[c])
                / (self.domain_max[c] - self.domain_min[c]))
                .clamp(0.0, 1.0)
                * (n - 1) as f32;
            let lo = (t.floor() as usize).min(n - 1);
            i0[c] = lo;
            i1[c] = (lo + 1).min(n - 1);
            f[c] = t - lo as f32;
        }
        let mut out = [0.0f32; 3];
        for corner in 0..8usize {
            let pick = |bit: usize, c: usize| if corner >> bit & 1 == 1 { i1[c] } else { i0[c] };
            let w = (if corner & 1 == 1 { f[0] } else { 1.0 - f[0] })
                * (if corner >> 1 & 1 == 1 { f[1] } else { 1.0 - f[1] })
                * (if corner >> 2 & 1 == 1 { f[2] } else { 1.0 - f[2] });
            let v = self.data[pick(0, 0) + pick(1, 1) * n + pick(2, 2) * n * n];
            for c in 0..3 {
                out[c] += w * v[c];
            }
        }
        out
    }
}

/// D-Log code value where the linear toe meets the log segment (as
/// published; the exact continuity point is 6.025·0.0078 + 0.0929 =
/// 0.1399005, and the two branches agree to ~6e-6 in between).
const DLOG_BREAK: f32 = 0.14;

/// Scene-linear from a D-Log code value — the reverse conversion published
/// in DJI's "D-Log and D-Gamut" white paper (verified against the PDF
/// 2026-07-10; constants come from the published formula, never from LUT
/// data):
///
/// ```text
/// x ≤ 0.14:  out = (x − 0.0929) / 6.025
/// x > 0.14:  out = (10^(3.89616·x − 2.27752) − 0.0108) / 0.9892
/// ```
///
/// written here as `(x − 0.584555)/0.256663` in the exponent, which is the
/// same expression (1/0.256663 = 3.89616, 0.584555/0.256663 = 2.27752).
fn dlog_decode(y: f32) -> f32 {
    if y <= DLOG_BREAK {
        (y - 0.0929) / 6.025
    } else {
        (10.0f32.powf((y - 0.584555) / 0.256663) - 0.0108) / 0.9892
    }
}

fn srgb_encode(x: f32) -> f32 {
    if x <= 0.003_130_8 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 }
}

/// Parametric D-Log → display-referred sRGB/Rec.709 fallback (doc/05 §1.1):
/// neutral contrast, no creative shoulder; scene-linear values above 1.0
/// clip to white. The Mavic 2 Pro records D-Log *M*, whose exact curve is
/// unpublished — this whitepaper curve is a documented approximation, and
/// the official `.cube` remains the correct path.
pub fn dlog_to_rec709(rgb: [f32; 3]) -> [f32; 3] {
    rgb.map(|y| srgb_encode(dlog_decode(y).clamp(0.0, 1.0)))
}

/// How video frames get display-referred (doc/05 §1.1). Photos never pass
/// through this — they are already sRGB.
#[derive(Debug, Clone)]
pub enum Tonemap {
    /// User-supplied `.cube` (correct for D-Log/D-Log M footage).
    Cube(Lut3d),
    /// Whitepaper approximation, for when no `.cube` is available.
    Parametric,
    /// Footage is already display-referred.
    None,
}

impl Tonemap {
    pub fn apply_px(&self, rgb: [f32; 3]) -> [f32; 3] {
        match self {
            Tonemap::Cube(lut) => lut.apply(rgb),
            Tonemap::Parametric => dlog_to_rec709(rgb),
            Tonemap::None => rgb,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDENTITY2: &str = include_str!("../tests/data/lut/identity2.cube");
    const CONTRAST3: &str = include_str!("../tests/data/lut/contrast3.cube");

    fn cube(text: &str) -> Lut3d {
        Lut3d::parse_cube(Path::new("test.cube"), text).expect("parse")
    }

    fn assert_close(a: [f32; 3], b: [f32; 3], tol: f32, what: &str) {
        for c in 0..3 {
            assert!((a[c] - b[c]).abs() <= tol, "{what}: {a:?} vs {b:?} (channel {c})");
        }
    }

    #[test]
    fn identity_lattice_midpoints_and_clamp() {
        let lut = cube(IDENTITY2);
        for p in [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
            [0.25, 0.5, 0.75],
            [0.1, 0.9, 0.3],
        ] {
            assert_close(lut.apply(p), p, 1e-6, "identity");
        }
        // out-of-domain input clamps to the box
        assert_close(lut.apply([-1.0, 2.0, 0.5]), [0.0, 1.0, 0.5], 1e-6, "clamp");
    }

    #[test]
    fn contrast3_trilinear_known_values() {
        let lut = cube(CONTRAST3);
        // lattice points: per-channel x² at 0 / 0.5 / 1
        assert_close(lut.apply([0.5, 0.5, 0.5]), [0.25, 0.25, 0.25], 1e-6, "lattice mid");
        assert_close(lut.apply([0.0, 0.5, 1.0]), [0.0, 0.25, 1.0], 1e-6, "lattice mixed");
        // cell center: interp between 0 and 0.25 at t = 0.5
        assert_close(lut.apply([0.25, 0.25, 0.25]), [0.125, 0.125, 0.125], 1e-6, "cell center");
        // asymmetric point exercising all three channels differently
        assert_close(lut.apply([0.25, 0.75, 0.5]), [0.125, 0.625, 0.25], 1e-6, "asymmetric");
        // red-fastest ordering: transposed index math would swap these
        assert_close(lut.apply([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0], 1e-6, "red axis");
        assert_close(lut.apply([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0], 1e-6, "blue axis");
    }

    #[test]
    fn crlf_and_unknown_keywords_tolerated() {
        let crlf = CONTRAST3.replace('\n', "\r\n");
        let lut = cube(&crlf);
        assert_close(lut.apply([0.25, 0.75, 0.5]), [0.125, 0.625, 0.25], 1e-6, "crlf");
        let extra = CONTRAST3.replace("TITLE \"contrast 3\"", "LUT_IN_VIDEO_RANGE\nTITLE \"x\"");
        cube(&extra);
    }

    #[test]
    fn domain_scaling() {
        let text = "LUT_3D_SIZE 2\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 2 2 2\n\
                    0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n";
        let lut = cube(text);
        assert_close(lut.apply([2.0, 2.0, 2.0]), [1.0, 1.0, 1.0], 1e-6, "domain top");
        assert_close(lut.apply([1.0, 1.0, 1.0]), [0.5, 0.5, 0.5], 1e-6, "domain mid");
    }

    #[test]
    fn parse_errors() {
        let bad = |text: &str| {
            Lut3d::parse_cube(Path::new("bad.cube"), text).expect_err("should fail")
        };
        bad("LUT_1D_SIZE 2\n0\n1\n");
        bad("LUT_3D_SIZE 1\n");
        bad("LUT_3D_SIZE 130\n");
        bad("0 0 0\n"); // missing LUT_3D_SIZE
        bad("LUT_3D_SIZE 2\n0 0 0\n"); // truncated data
        bad(&format!("{IDENTITY2}0 0 0\n")); // surplus data
        bad("LUT_3D_SIZE 2\n0 0 zebra\n");
        bad("LUT_3D_SIZE 2\n0 0 0 0\n"); // 4-value row
        bad(&IDENTITY2.replace("DOMAIN_MAX 1.0 1.0 1.0", "DOMAIN_MAX 0.0 0.0 0.0"));
    }

    #[test]
    fn parametric_anchors_and_monotonicity() {
        // black anchor: code value 0.0929 is scene-linear 0
        assert_close(dlog_to_rec709([0.0929; 3]), [0.0; 3], 1e-6, "black");
        // below the toe clamps to 0, not negative
        assert_close(dlog_to_rec709([0.0; 3]), [0.0; 3], 1e-6, "sub-black");
        // 18 % grey: encode(0.18) ≈ 0.398766 → srgb(0.18) ≈ 0.46137
        assert_close(dlog_to_rec709([0.398766; 3]), [0.46137; 3], 2e-3, "18% grey");
        // white anchor: encode(1.0) = 0.584555 exactly
        assert_close(dlog_to_rec709([0.584555; 3]), [1.0; 3], 1e-5, "white");
        // strictly display-referred: monotone non-decreasing over the range
        let mut prev = -1.0f32;
        for i in 0..=1000 {
            let y = i as f32 / 1000.0;
            let v = dlog_to_rec709([y; 3])[0];
            assert!(v >= prev, "non-monotone at y={y}: {v} < {prev}");
            prev = v;
        }
        // continuous across the toe/log breakpoint
        let below = dlog_to_rec709([DLOG_BREAK - 1e-4; 3])[0];
        let above = dlog_to_rec709([DLOG_BREAK + 1e-4; 3])[0];
        assert!((above - below).abs() < 1e-2, "kink at breakpoint: {below} vs {above}");
    }

    #[test]
    fn tonemap_none_is_identity() {
        let px = [0.1, 0.5, 0.9];
        assert_close(Tonemap::None.apply_px(px), px, 0.0, "none");
        assert_close(
            Tonemap::Parametric.apply_px([0.584555; 3]),
            [1.0; 3],
            1e-5,
            "parametric dispatch",
        );
    }
}
