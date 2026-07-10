//! DJI SRT telemetry (doc/06 §1): tolerant line-oriented parser for the
//! per-frame subtitle telemetry (GPS, altitudes, gimbal, exposure) plus
//! the geodesy helpers keyframe selection needs. Formats vary by firmware;
//! unknown keys and malformed blocks are skipped, and the parser is
//! unit-tested against real samples from our own aircraft.

use std::path::Path;
use std::time::Duration;

use crate::error::CaptureError;

/// One GPS fix from drone telemetry (or photo EXIF).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GpsFix {
    pub lat: f64,
    pub lon: f64,
    /// Barometric altitude relative to takeoff — smoother than GPS altitude
    /// and internally consistent per flight; the preferred vertical axis
    /// (doc/06 §1).
    pub rel_alt_m: Option<f64>,
    pub abs_alt_m: Option<f64>,
}

/// Great-circle distance in meters, horizontal only.
pub fn haversine_m(a: &GpsFix, b: &GpsFix) -> f64 {
    const R_EARTH_M: f64 = 6_371_000.0;
    let (la, lb) = (a.lat.to_radians(), b.lat.to_radians());
    let dlat = (b.lat - a.lat).to_radians();
    let dlon = (b.lon - a.lon).to_radians();
    let s = (dlat / 2.0).sin().powi(2) + la.cos() * lb.cos() * (dlon / 2.0).sin().powi(2);
    2.0 * R_EARTH_M * s.sqrt().asin()
}

/// One subtitle block's telemetry.
#[derive(Debug, Clone, PartialEq)]
pub struct SrtEntry {
    pub start: Duration,
    pub end: Duration,
    pub gps: Option<GpsFix>,
    pub gimbal_yaw_deg: Option<f64>,
    pub gimbal_pitch_deg: Option<f64>,
    pub iso: Option<u32>,
    /// As written in the telemetry (e.g. `1/640.0`).
    pub shutter: Option<String>,
    /// Normalized f-number (some firmware writes ×100: `560` = f/5.6).
    pub fnum: Option<f64>,
    /// Color mode as logged (`d_log` means the footage needs tonemapping).
    pub color_md: Option<String>,
}

pub struct SrtTrack {
    /// Sorted by `start`.
    pub entries: Vec<SrtEntry>,
}

impl SrtTrack {
    /// Parse SRT text: blocks of (numeric index), `HH:MM:SS,mmm -->
    /// HH:MM:SS,mmm`, then payload lines scanned for `key : value` tokens
    /// (the bracketed `[latitude: 47.6]` firmware style — including DJI's
    /// `longtitude` spelling — and the older `GPS(lon,lat,alt)` /
    /// `BAROMETER: x.x` style). HTML font tags and CRLF are tolerated.
    /// Malformed blocks are skipped; zero parsed entries is an error.
    pub fn parse(path: &Path, text: &str) -> Result<SrtTrack, CaptureError> {
        let mut entries: Vec<SrtEntry> = Vec::new();
        let mut block: Vec<&str> = Vec::new();
        for line in text.lines().map(str::trim) {
            if line.is_empty() {
                if !block.is_empty() {
                    entries.extend(parse_block(&block));
                    block.clear();
                }
            } else {
                block.push(line);
            }
        }
        if !block.is_empty() {
            entries.extend(parse_block(&block));
        }
        if entries.is_empty() {
            return Err(CaptureError::Srt {
                path: path.to_owned(),
                reason: "no parsable telemetry entries".into(),
            });
        }
        entries.sort_by_key(|e| e.start);
        Ok(SrtTrack { entries })
    }

    /// Entry whose `[start, end)` span contains `t`, else the nearest entry
    /// within 500 ms (telemetry cadence is per-frame-ish; anything farther
    /// is a real gap).
    pub fn at(&self, t: Duration) -> Option<&SrtEntry> {
        const SLACK: Duration = Duration::from_millis(500);
        let idx = self.entries.partition_point(|e| e.start <= t);
        let mut best: Option<(&SrtEntry, Duration)> = None;
        let neighbors = [idx.checked_sub(1).and_then(|i| self.entries.get(i)), self.entries.get(idx)];
        for e in neighbors.into_iter().flatten() {
            if t >= e.start && t < e.end {
                return Some(e);
            }
            let d = if t < e.start { e.start - t } else { t.saturating_sub(e.end) };
            if d <= SLACK && best.is_none_or(|(_, bd)| d < bd) {
                best = Some((e, d));
            }
        }
        best.map(|(e, _)| e)
    }

    /// Telemetry for source frame `n` of an `fps` video, queried at the
    /// frame midpoint (doc/06 §1).
    pub fn at_frame(&self, n: u32, fps: f64) -> Option<&SrtEntry> {
        self.at(Duration::from_secs_f64((f64::from(n) + 0.5) / fps))
    }
}

fn parse_block(lines: &[&str]) -> Option<SrtEntry> {
    let tc_line = lines.iter().find(|l| l.contains("-->"))?;
    let (start_s, end_s) = tc_line.split_once("-->")?;
    let (start, end) = (parse_timestamp(start_s.trim())?, parse_timestamp(end_s.trim())?);

    let payload: String = lines
        .iter()
        .skip_while(|l| !l.contains("-->"))
        .skip(1)
        .map(|l| strip_tags(l))
        .collect::<Vec<_>>()
        .join(" ");

    let mut e = SrtEntry {
        start,
        end,
        gps: None,
        gimbal_yaw_deg: None,
        gimbal_pitch_deg: None,
        iso: None,
        shutter: None,
        fnum: None,
        color_md: None,
    };
    let (mut lat, mut lon) = (None, None);
    let (mut rel_alt, mut abs_alt) = (None, None);

    // older firmware style: GPS(a,b,alt), conventionally lon-first — swap
    // on a plainly impossible latitude
    if let Some(p) = payload.find("GPS(")
        && let Some(close) = payload[p..].find(')')
    {
        let mut it = payload[p + 4..p + close].split(',').map(str::trim);
        if let (Some(a), Some(b)) = (it.next(), it.next())
            && let (Ok(a), Ok(b)) = (a.parse::<f64>(), b.parse::<f64>())
        {
            let (la, lo) = if a.abs() > 90.0 { (b, a) } else if b.abs() > 90.0 { (a, b) } else { (b, a) };
            (lat, lon) = (Some(la), Some(lo));
        }
    }

    // `key : value` scan — brackets and commas become spaces, colons become
    // their own tokens, keys must start alphabetic (so the embedded
    // date/time lines never match)
    let spaced: String = payload
        .chars()
        .map(|c| if matches!(c, '[' | ']' | ',') { ' ' } else { c })
        .collect::<String>()
        .replace(':', " : ");
    let toks: Vec<&str> = spaced.split_whitespace().collect();
    for i in 0..toks.len() {
        if toks.get(i + 1) != Some(&":") || !toks[i].starts_with(|c: char| c.is_ascii_alphabetic())
        {
            continue;
        }
        let Some(&val) = toks.get(i + 2) else { continue };
        let num = || val.parse::<f64>().ok();
        match toks[i].to_ascii_lowercase().as_str() {
            "latitude" | "lat" => lat = num(),
            // DJI firmware writes "longtitude"
            "longtitude" | "longitude" | "lon" => lon = num(),
            "rel_alt" | "barometer" => rel_alt = num(),
            "abs_alt" => abs_alt = num(),
            "altitude" => abs_alt = abs_alt.or_else(num),
            "gb_yaw" => e.gimbal_yaw_deg = num(),
            "gb_pitch" => e.gimbal_pitch_deg = num(),
            "iso" => e.iso = num().map(|f| f as u32),
            "shutter" => e.shutter = Some(val.to_string()),
            // ×100 encoding: no real lens is f/91+, everything above is
            // firmware shorthand (560 = f/5.6)
            "fnum" => e.fnum = num().map(|f| if f > 90.0 { f / 100.0 } else { f }),
            "color_md" => e.color_md = Some(val.to_string()),
            _ => {}
        }
    }

    // (0, 0) is DJI's no-fix sentinel
    if let (Some(la), Some(lo)) = (lat, lon)
        && (la.abs() > 1e-6 || lo.abs() > 1e-6)
    {
        e.gps = Some(GpsFix { lat: la, lon: lo, rel_alt_m: rel_alt, abs_alt_m: abs_alt });
    }
    Some(e)
}

/// `HH:MM:SS,mmm` (or `.mmm`).
fn parse_timestamp(s: &str) -> Option<Duration> {
    let (hms, ms) = s.split_once([',', '.'])?;
    let mut it = hms.split(':');
    let h: u64 = it.next()?.trim().parse().ok()?;
    let m: u64 = it.next()?.parse().ok()?;
    let sec: u64 = it.next()?.parse().ok()?;
    if it.next().is_some() {
        return None;
    }
    let ms: u64 = ms.trim().parse().ok()?;
    Some(Duration::from_millis(((h * 60 + m) * 60 + sec) * 1000 + ms))
}

/// Drop `<font …>`-style tags.
fn strip_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for ch in s.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            c if !in_tag => out.push(c),
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fix(lat: f64, lon: f64) -> GpsFix {
        GpsFix { lat, lon, rel_alt_m: None, abs_alt_m: None }
    }

    #[test]
    fn haversine_known_distances() {
        assert_eq!(haversine_m(&fix(60.0, 25.0), &fix(60.0, 25.0)), 0.0);
        // one degree of latitude ≈ 111.2 km
        let d = haversine_m(&fix(60.0, 25.0), &fix(61.0, 25.0));
        assert!((d - 111_200.0).abs() < 200.0, "{d}");
        // one degree of longitude at 60°N ≈ half that
        let d = haversine_m(&fix(60.0, 25.0), &fix(60.0, 26.0));
        assert!((d - 55_600.0).abs() < 200.0, "{d}");
        // sub-meter resolution around a hover
        let d = haversine_m(&fix(60.0, 25.0), &fix(60.000004, 25.0));
        assert!((0.2..0.7).contains(&d), "{d}");
    }

    /// Mavic-2-era bracketed style, font tags, `longtitude` typo and all.
    const BRACKETED: &str = "\
1
00:00:00,000 --> 00:00:00,040
<font size=\"36\">FrameCnt : 1, DiffTime : 40ms
2025-06-23 08:25:01,390
[iso : 100] [shutter : 1/640.0] [fnum : 5.4] [ev : 0] [ct : 5300] \
[color_md : default] [focal_len : 280] [latitude: 61.498611] \
[longtitude: 23.760556] [rel_alt: 30.700 abs_alt: 142.986] \
[gb_yaw : -12.3 gb_pitch : -89.9 gb_roll : 0] </font>

2
00:00:00,040 --> 00:00:00,080
<font size=\"36\">FrameCnt : 2, DiffTime : 40ms
2025-06-23 08:25:01,430
[iso : 100] [shutter : 1/640.0] [fnum : 5.4] [latitude: 61.498622] \
[longtitude: 23.760567] [rel_alt: 30.800 abs_alt: 143.086] </font>
";

    #[test]
    fn bracketed_firmware_style() {
        let t = SrtTrack::parse(Path::new("a.srt"), BRACKETED).unwrap();
        assert_eq!(t.entries.len(), 2);
        let e = &t.entries[0];
        assert_eq!(e.start, Duration::ZERO);
        assert_eq!(e.end, Duration::from_millis(40));
        let g = e.gps.expect("gps");
        assert!((g.lat - 61.498611).abs() < 1e-9);
        assert!((g.lon - 23.760556).abs() < 1e-9, "longtitude typo must parse");
        assert_eq!(g.rel_alt_m, Some(30.7));
        assert_eq!(g.abs_alt_m, Some(142.986));
        assert_eq!(e.gimbal_yaw_deg, Some(-12.3));
        assert_eq!(e.gimbal_pitch_deg, Some(-89.9));
        assert_eq!(e.iso, Some(100));
        assert_eq!(e.shutter.as_deref(), Some("1/640.0"));
        assert_eq!(e.fnum, Some(5.4));
    }

    #[test]
    fn gps_paren_and_barometer_style() {
        let text = "\
1
00:00:00,000 --> 00:00:01,000
GPS(23.760556,61.498611,19) BAROMETER:30.1
HOME(23.7601,61.4980) D=12.3m H=30.1m
";
        let t = SrtTrack::parse(Path::new("b.srt"), text).unwrap();
        let g = t.entries[0].gps.expect("gps");
        // lon-first convention
        assert!((g.lat - 61.498611).abs() < 1e-9);
        assert!((g.lon - 23.760556).abs() < 1e-9);
        assert_eq!(g.rel_alt_m, Some(30.1));

        // impossible latitude in slot two → swapped
        let text = "1\n00:00:00,000 --> 00:00:01,000\nGPS(37.795,-122.397,19)\n";
        let t = SrtTrack::parse(Path::new("c.srt"), text).unwrap();
        let g = t.entries[0].gps.expect("gps");
        assert!((g.lat - 37.795).abs() < 1e-9);
        assert!((g.lon + 122.397).abs() < 1e-9);
    }

    #[test]
    fn tolerance_and_errors() {
        // malformed middle block skipped, CRLF + '.' millis tolerated
        let text = "1\r\n00:00:00.000 --> 00:00:00.500\r\n[latitude: 61.5] [longitude: 23.7]\r\n\r\n\
                    garbage without a timecode\r\n\r\n\
                    3\r\n00:00:00,500 --> 00:00:01,000\r\n[latitude: 61.6] [longitude: 23.8]\r\n";
        let t = SrtTrack::parse(Path::new("d.srt"), text).unwrap();
        assert_eq!(t.entries.len(), 2);

        // (0,0) is the no-fix sentinel
        let text = "1\n00:00:00,000 --> 00:00:01,000\n[latitude: 0.000000] [longtitude: 0.000000]\n";
        let t = SrtTrack::parse(Path::new("e.srt"), text).unwrap();
        assert_eq!(t.entries[0].gps, None);

        // zero entries is an error
        assert!(SrtTrack::parse(Path::new("f.srt"), "no blocks here\n").is_err());
        assert!(SrtTrack::parse(Path::new("g.srt"), "").is_err());
    }

    /// Real capture excerpt (DJI drone over the villa scene, 2026-07-10;
    /// M4a gate: parse on real committed samples). Telemetry is verbatim
    /// except lat/lon, offset by (+2.1°, −1.7°) into open water for
    /// privacy — relative movement between entries is authentic. This
    /// firmware logs no altitude fields and writes fnum ×100.
    #[test]
    fn real_villa_excerpt() {
        let text = include_str!("../tests/data/srt/dji-villa-excerpt.srt");
        let t = SrtTrack::parse(Path::new("dji-villa-excerpt.srt"), text).unwrap();
        assert_eq!(t.entries.len(), 8);

        let first = &t.entries[0];
        assert_eq!(first.start, Duration::ZERO);
        assert_eq!(first.end, Duration::from_millis(33));
        let g = first.gps.expect("gps");
        assert!((g.lat - 63.406410).abs() < 1e-9);
        assert!((g.lon - 21.211324).abs() < 1e-9);
        assert_eq!(g.rel_alt_m, None, "this firmware logs no altitude");
        assert_eq!(g.abs_alt_m, None);
        assert_eq!(first.iso, Some(100));
        assert_eq!(first.shutter.as_deref(), Some("1/640.0"));
        assert_eq!(first.fnum, Some(5.6), "fnum 560 must normalize to f/5.6");
        assert_eq!(first.color_md.as_deref(), Some("d_log"));

        let last = &t.entries[7];
        assert_eq!(last.start, Duration::from_millis(166_856));
        // slow drift: entry 0 → 7 moved single-digit meters
        let d = haversine_m(&first.gps.unwrap(), &last.gps.unwrap());
        assert!((3.0..15.0).contains(&d), "{d} m");

        // frame association at the real 29.97 fps
        assert_eq!(t.at_frame(0, 29.97).unwrap().start, Duration::ZERO);
        assert_eq!(
            t.at_frame(667, 29.97).unwrap().start,
            Duration::from_millis(22_254),
            "frame 667 midpoint lands in block 668's span"
        );
        // the excerpt has gaps; a query inside one finds nothing
        assert!(t.at(Duration::from_secs(60)).is_none());
    }

    #[test]
    fn at_association_properties() {
        let t = SrtTrack::parse(Path::new("a.srt"), BRACKETED).unwrap();
        // frame midpoints land in the right span at 25 fps
        let e = t.at_frame(0, 25.0).expect("frame 0"); // t = 20 ms
        assert_eq!(e.start, Duration::ZERO);
        let e = t.at_frame(1, 25.0).expect("frame 1"); // t = 60 ms
        assert_eq!(e.start, Duration::from_millis(40));
        // beyond the last span but within 500 ms → nearest
        assert!(t.at(Duration::from_millis(400)).is_some());
        // a real gap → None
        assert!(t.at(Duration::from_millis(700)).is_none());
        // exact span starts resolve to their own entry
        let e = t.at(Duration::from_millis(40)).expect("boundary");
        assert_eq!(e.start, Duration::from_millis(40));
    }
}
