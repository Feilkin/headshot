//! Budget-driven keyframe selection (doc/05 §2). Pure data-in/data-out —
//! no I/O, no subprocess — so the M4a gate (deterministic, budget-
//! respecting) is testable in CI on synthetic frames.

use crate::srt::{GpsFix, haversine_m};

/// 8-bit grayscale frame at scoring or thumbnail resolution.
#[derive(Debug, Clone)]
pub struct GrayFrame {
    pub width: u32,
    pub height: u32,
    /// Row-major, `width·height` bytes.
    pub data: Vec<u8>,
}

/// Small RGB8 frame (scoring/preview resolution; the UI's thumbnail type).
#[derive(Debug, Clone)]
pub struct RgbFrame {
    pub width: u32,
    pub height: u32,
    /// Row-major RGB8, `width·height·3` bytes.
    pub data: Vec<u8>,
}

/// Fixed integer rec601 luma — deterministic scoring input regardless of
/// what the decoder would call "gray".
pub fn rgb_to_gray(f: &RgbFrame) -> GrayFrame {
    GrayFrame {
        width: f.width,
        height: f.height,
        data: f
            .data
            .chunks_exact(3)
            .map(|px| {
                ((77 * u32::from(px[0]) + 150 * u32::from(px[1]) + 29 * u32::from(px[2]) + 128)
                    >> 8) as u8
            })
            .collect(),
    }
}

/// Deterministic box-average downscale of an RGB thumb to `out_w` wide.
pub fn rgb_thumb(f: &RgbFrame, out_w: u32) -> RgbFrame {
    let ow = out_w.clamp(1, f.width);
    let oh = ((u64::from(f.height) * u64::from(ow)).div_ceil(u64::from(f.width)).max(1)) as u32;
    let (sw, sh) = (f.width as usize, f.height as usize);
    let mut data = Vec::with_capacity((ow * oh * 3) as usize);
    for oy in 0..oh as usize {
        let y0 = oy * sh / oh as usize;
        let y1 = (((oy + 1) * sh) / oh as usize).max(y0 + 1).min(sh);
        for ox in 0..ow as usize {
            let x0 = ox * sw / ow as usize;
            let x1 = (((ox + 1) * sw) / ow as usize).max(x0 + 1).min(sw);
            let mut acc = [0u64; 3];
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = (y * sw + x) * 3;
                    for (c, a) in acc.iter_mut().enumerate() {
                        *a += u64::from(f.data[i + c]);
                    }
                }
            }
            let cnt = ((y1 - y0) * (x1 - x0)) as u64;
            data.extend(acc.map(|a| (a / cnt) as u8));
        }
    }
    RgbFrame { width: ow, height: oh, data }
}

/// One temporally-sampled video frame entering selection.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub source_frame: u32,
    /// Position within the source video.
    pub time_s: f64,
    /// Variance-of-Laplacian score at scoring resolution.
    pub sharpness: f64,
    /// Small grayscale thumb (~64 px wide) for no-telemetry redundancy
    /// diffs against the last kept frame.
    pub thumb: GrayFrame,
    pub gps: Option<GpsFix>,
    pub gimbal_yaw_deg: Option<f64>,
    /// Not used for selection; rides along into the manifest.
    pub gimbal_pitch_deg: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct SelectParams {
    /// Candidates per sharpness window (≈2 s at the ~2 fps sampling rate).
    pub window: usize,
    /// GPS displacement below which the drone "barely moved" (doc/05 §2).
    pub min_gps_move_m: f64,
    /// Gimbal yaw change that counts as a new viewpoint despite no motion.
    pub min_yaw_deg: f64,
    /// Mean |Δgray| per pixel (of 255) below which two thumbs count as
    /// redundant when telemetry is unavailable.
    pub min_frame_diff: f64,
    pub budget: usize,
}

impl Default for SelectParams {
    fn default() -> Self {
        Self { window: 4, min_gps_move_m: 0.5, min_yaw_deg: 5.0, min_frame_diff: 4.0, budget: 200 }
    }
}

/// Select keyframes: sharpest per window → redundancy pruning against the
/// last kept frame (GPS+yaw when available, thumb difference otherwise) →
/// uniform decimation to the budget preserving first/last (doc/05 §2).
/// Returns sorted indices into `cands`. Pure function of its inputs; all
/// ties break to the lower index.
pub fn select_keyframes(cands: &[Candidate], p: &SelectParams) -> Vec<usize> {
    if p.budget == 0 {
        return Vec::new();
    }
    decimate(&select_survivors(cands, p), p.budget)
}

/// Stages 1+2 only (sharpness windows + redundancy pruning), before any
/// budget decimation — session assembly needs the undecimated survivor
/// counts to split the budget across videos.
pub fn select_survivors(cands: &[Candidate], p: &SelectParams) -> Vec<usize> {
    if cands.is_empty() {
        return Vec::new();
    }

    // stage 1 — sharpest per window (step 2)
    let window = p.window.max(1);
    let mut survivors = Vec::with_capacity(cands.len() / window + 1);
    let mut w0 = 0;
    while w0 < cands.len() {
        let end = (w0 + window).min(cands.len());
        let best = (w0..end)
            .max_by(|&a, &b| cands[a].sharpness.total_cmp(&cands[b].sharpness).then(b.cmp(&a)))
            .expect("non-empty window");
        survivors.push(best);
        w0 = end;
    }

    // stage 2 — redundancy pruning vs the last kept (step 3)
    let mut kept = vec![survivors[0]];
    for &s in &survivors[1..] {
        let a = &cands[*kept.last().expect("non-empty")];
        let b = &cands[s];
        let redundant = match (&a.gps, &b.gps) {
            (Some(ga), Some(gb)) => {
                let moved = haversine_m(ga, gb) >= p.min_gps_move_m;
                let yawed = match (a.gimbal_yaw_deg, b.gimbal_yaw_deg) {
                    (Some(ya), Some(yb)) => wrap180(ya - yb).abs() >= p.min_yaw_deg,
                    _ => false,
                };
                !moved && !yawed
            }
            _ => mean_abs_diff(&a.thumb, &b.thumb) < p.min_frame_diff,
        };
        if !redundant {
            kept.push(s);
        }
    }
    let last = *survivors.last().expect("non-empty");
    if *kept.last().expect("non-empty") != last {
        kept.push(last);
    }
    kept
}

/// Uniform decimation to `budget` via `round(k·(len−1)/(budget−1))`,
/// preserving first and last (stage 3 / doc/05 §2 step 4).
pub(crate) fn decimate(kept: &[usize], budget: usize) -> Vec<usize> {
    if kept.len() <= budget {
        return kept.to_vec();
    }
    if budget == 1 {
        return vec![kept[0]];
    }
    let span = (kept.len() - 1) as f64;
    let mut out = Vec::with_capacity(budget);
    for k in 0..budget {
        let idx = (k as f64 * span / (budget - 1) as f64).round() as usize;
        if out.last() != Some(&kept[idx]) {
            out.push(kept[idx]);
        }
    }
    out
}

fn wrap180(deg: f64) -> f64 {
    let d = deg.rem_euclid(360.0);
    if d > 180.0 { d - 360.0 } else { d }
}

/// Sharpness score: variance of the 4-neighbor Laplacian over the interior
/// (doc/05 §2 step 2). f64 accumulation in scan order — deterministic.
pub fn sharpness_var_laplacian(g: &GrayFrame) -> f64 {
    let (w, h) = (g.width as usize, g.height as usize);
    if w < 3 || h < 3 {
        return 0.0;
    }
    let px = |x: usize, y: usize| g.data[y * w + x] as f64;
    let n = ((w - 2) * (h - 2)) as f64;
    let mut sum = 0.0;
    let mut sum2 = 0.0;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let lap = px(x - 1, y) + px(x + 1, y) + px(x, y - 1) + px(x, y + 1) - 4.0 * px(x, y);
            sum += lap;
            sum2 += lap * lap;
        }
    }
    (sum2 - sum * sum / n) / n
}

/// Mean absolute grayscale difference per pixel; frames must share dims.
pub fn mean_abs_diff(a: &GrayFrame, b: &GrayFrame) -> f64 {
    assert_eq!((a.width, a.height), (b.width, b.height), "thumb dims must match");
    let total: u64 = a
        .data
        .iter()
        .zip(&b.data)
        .map(|(&x, &y)| u64::from(x.abs_diff(y)))
        .sum();
    total as f64 / a.data.len() as f64
}

/// Deterministic box-average downscale to `out_w` wide (aspect kept).
pub fn gray_thumb(g: &GrayFrame, out_w: u32) -> GrayFrame {
    let ow = out_w.clamp(1, g.width);
    let oh = ((u64::from(g.height) * u64::from(ow)).div_ceil(u64::from(g.width)).max(1)) as u32;
    let (sw, sh) = (g.width as usize, g.height as usize);
    let mut data = Vec::with_capacity((ow * oh) as usize);
    for oy in 0..oh as usize {
        let y0 = oy * sh / oh as usize;
        let y1 = (((oy + 1) * sh) / oh as usize).max(y0 + 1).min(sh);
        for ox in 0..ow as usize {
            let x0 = ox * sw / ow as usize;
            let x1 = (((ox + 1) * sw) / ow as usize).max(x0 + 1).min(sw);
            let mut acc = 0u64;
            for y in y0..y1 {
                for x in x0..x1 {
                    acc += u64::from(g.data[y * sw + x]);
                }
            }
            data.push((acc / ((y1 - y0) * (x1 - x0)) as u64) as u8);
        }
    }
    GrayFrame { width: ow, height: oh, data }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_thumb(value: u8) -> GrayFrame {
        GrayFrame { width: 4, height: 4, data: vec![value; 16] }
    }

    /// Candidate with a per-index distinct thumb (mean diff 16 ≥ default
    /// threshold 4), so the no-GPS path never prunes unless a test says so.
    fn cand(i: u32, sharpness: f64) -> Candidate {
        Candidate {
            source_frame: i,
            time_s: f64::from(i) / 2.0,
            sharpness,
            thumb: flat_thumb((i as u8).wrapping_mul(16)),
            gps: None,
            gimbal_yaw_deg: None,
            gimbal_pitch_deg: None,
        }
    }

    fn with_gps(mut c: Candidate, lat: f64, yaw: f64) -> Candidate {
        c.gps = Some(GpsFix { lat, lon: 25.0, rel_alt_m: Some(30.0), abs_alt_m: None });
        c.gimbal_yaw_deg = Some(yaw);
        c
    }

    #[test]
    fn window_argmax_and_tie_to_lower_index() {
        let sharp = [1.0, 5.0, 3.0, 2.0, 7.0, 7.0, 1.0, 0.0];
        let cands: Vec<Candidate> =
            sharp.iter().enumerate().map(|(i, &s)| cand(i as u32, s)).collect();
        let out = select_keyframes(&cands, &SelectParams::default());
        // windows [0..4] → 1; [4..8] → 4 (tie 4 vs 5 breaks low)
        assert_eq!(out, vec![1, 4]);
    }

    #[test]
    fn deterministic_and_budget_respecting() {
        // pseudo-random sharpness from a fixed LCG; mixed GPS presence
        let mut state = 42u64;
        let mut rand = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as f64 / f64::from(u32::MAX)
        };
        let cands: Vec<Candidate> = (0..240)
            .map(|i| {
                let c = cand(i, rand() * 1000.0);
                if i % 3 == 0 {
                    with_gps(c, 60.0 + f64::from(i) * 1e-4, f64::from(i % 360))
                } else {
                    c
                }
            })
            .collect();
        for budget in [1, 2, 7, 20, 1000] {
            let p = SelectParams { budget, ..Default::default() };
            let a = select_keyframes(&cands, &p);
            let b = select_keyframes(&cands, &p);
            assert_eq!(a, b, "budget {budget}: nondeterministic");
            assert!(a.len() <= budget, "budget {budget}: got {}", a.len());
            assert!(a.windows(2).all(|w| w[0] < w[1]), "budget {budget}: not sorted/unique");
        }
        assert!(select_keyframes(&cands, &SelectParams { budget: 0, ..Default::default() }).is_empty());
        assert!(select_keyframes(&[], &SelectParams::default()).is_empty());
    }

    #[test]
    fn hover_collapses_yaw_survives() {
        // 12 candidates hovering at one GPS position, constant yaw:
        // everything between the first and last survivor is redundant
        let hover: Vec<Candidate> = (0..12)
            .map(|i| with_gps(cand(i, f64::from(i)), 60.0, 90.0))
            .collect();
        let out = select_keyframes(&hover, &SelectParams::default());
        // windows: [0..4]→3, [4..8]→7, [8..12]→11 (sharpness increasing);
        // 7 pruned (no move, no yaw), 11 re-appended as the last survivor
        assert_eq!(out, vec![3, 11]);

        // same hover but the gimbal sweeps 40° per window step → all survive
        let sweep: Vec<Candidate> = (0..12)
            .map(|i| with_gps(cand(i, f64::from(i)), 60.0, f64::from(i) * 10.0))
            .collect();
        let out = select_keyframes(&sweep, &SelectParams::default());
        assert_eq!(out, vec![3, 7, 11]);

        // forward flight: ≥0.5 m between survivors → all survive
        let flight: Vec<Candidate> = (0..12)
            .map(|i| with_gps(cand(i, f64::from(i)), 60.0 + f64::from(i) * 3e-5, 90.0))
            .collect();
        let out = select_keyframes(&flight, &SelectParams::default());
        assert_eq!(out, vec![3, 7, 11]);
    }

    #[test]
    fn no_gps_prunes_by_thumb_difference() {
        // static tripod shot, identical thumbs → collapses to ends
        let static_scene: Vec<Candidate> = (0..12)
            .map(|i| {
                let mut c = cand(i, f64::from(i));
                c.thumb = flat_thumb(128);
                c
            })
            .collect();
        let out = select_keyframes(&static_scene, &SelectParams::default());
        assert_eq!(out, vec![3, 11]);

        // distinct thumbs (default `cand`) → nothing pruned
        let moving: Vec<Candidate> = (0..12).map(|i| cand(i, f64::from(i))).collect();
        let out = select_keyframes(&moving, &SelectParams::default());
        assert_eq!(out, vec![3, 7, 11]);
    }

    #[test]
    fn decimation_properties() {
        for len in 1..40usize {
            let kept: Vec<usize> = (0..len).collect();
            for budget in 1..12usize {
                let out = decimate(&kept, budget);
                if len <= budget {
                    assert_eq!(out, kept, "len {len} budget {budget}");
                } else {
                    assert_eq!(out.len(), budget, "len {len} budget {budget}");
                    assert_eq!(*out.first().unwrap(), 0, "len {len} budget {budget}");
                    if budget > 1 {
                        assert_eq!(*out.last().unwrap(), len - 1, "len {len} budget {budget}");
                    }
                    assert!(out.windows(2).all(|w| w[0] < w[1]), "len {len} budget {budget}");
                }
            }
        }
    }

    #[test]
    fn sharpness_orders_sharp_above_blurry() {
        let w = 32u32;
        // flat frame: zero variance
        let flat = GrayFrame { width: w, height: w, data: vec![100; (w * w) as usize] };
        assert_eq!(sharpness_var_laplacian(&flat), 0.0);
        // checkerboard: maximal Laplacian energy
        let checker = GrayFrame {
            width: w,
            height: w,
            data: (0..w * w)
                .map(|i| if (i / w + i % w).is_multiple_of(2) { 0 } else { 255 })
                .collect(),
        };
        // smooth horizontal ramp: tiny second derivative
        let ramp = GrayFrame {
            width: w,
            height: w,
            data: (0..w * w).map(|i| ((i % w) * 8) as u8).collect(),
        };
        let (s_checker, s_ramp) = (sharpness_var_laplacian(&checker), sharpness_var_laplacian(&ramp));
        assert!(s_checker > s_ramp, "{s_checker} vs {s_ramp}");
        assert!(s_ramp < 1.0, "{s_ramp}");
    }

    #[test]
    fn thumb_and_diff_basics() {
        let g = GrayFrame {
            width: 64,
            height: 32,
            data: (0..64 * 32).map(|i| (i % 256) as u8).collect(),
        };
        let t = gray_thumb(&g, 16);
        assert_eq!((t.width, t.height), (16, 8));
        // box average preserves the global mean closely
        let mean =
            |f: &GrayFrame| f.data.iter().map(|&v| f64::from(v)).sum::<f64>() / f.data.len() as f64;
        assert!((mean(&g) - mean(&t)).abs() < 4.0);
        // upscale request clamps to source width
        assert_eq!(gray_thumb(&g, 999).width, 64);

        assert_eq!(mean_abs_diff(&flat_thumb(10), &flat_thumb(14)), 4.0);
        assert_eq!(mean_abs_diff(&flat_thumb(14), &flat_thumb(10)), 4.0);
    }
}
