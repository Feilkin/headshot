//! CI end-to-end test of `prepare_session` (doc/05 §§2–3) on a synthetic
//! session: a mock 40-candidate "video" with scripted sharpness/GPS/SRT
//! plus three PNG photos — no ffmpeg, no real media.

use std::path::{Path, PathBuf};

use headshot_capture::error::CaptureError;
use headshot_capture::keyframe::{RgbFrame, SelectParams};
use headshot_capture::lut::dlog_to_rec709;
use headshot_capture::manifest::{SourceKind, TonemapKind};
use headshot_capture::video::{Rgb48Frame, VideoBackend, VideoMeta};
use headshot_capture::{CaptureConfig, prepare_session};

const N_CANDS: u32 = 40;
const VID_W: u32 = 192;
const VID_H: u32 = 108;

/// rgb48 code values for mock source frame `i` (varying red encodes the
/// frame identity; green/blue fixed).
fn mock_rgb48_px(i: u32) -> [u16; 3] {
    [(500 + i * 700) as u16, 20_000, 40_000]
}

struct MockBackend;

impl VideoBackend for MockBackend {
    fn probe(&self, _path: &Path) -> Result<VideoMeta, CaptureError> {
        Ok(VideoMeta {
            width: VID_W,
            height: VID_H,
            fps: 2.0, // → candidate_step = 1, candidate i ↔ source frame i
            n_frames: Some(u64::from(N_CANDS)),
            duration_s: Some(f64::from(N_CANDS) / 2.0),
            color_range: Some("tv".into()),
            color_transfer: Some("bt709".into()),
            subtitle_stream: None,
        })
    }

    fn sample_thumbs(
        &self,
        _path: &Path,
        _meta: &VideoMeta,
        step: u32,
        out_w: u32,
    ) -> Result<Vec<RgbFrame>, CaptureError> {
        assert_eq!(step, 1, "fps 2.0 must sample every frame");
        let h = out_w * VID_H / VID_W;
        Ok((0..N_CANDS)
            .map(|i| {
                // every 4th frame is a checkerboard with amplitude 50+i
                // (strictly increasing sharpness); the rest are flat.
                // r=g=b so the in-crate rec601 gray is an exact identity.
                let data = if i.is_multiple_of(4) {
                    let amp = (50 + i) as u8;
                    (0..out_w * h)
                        .flat_map(|p| {
                            let (x, y) = (p % out_w, p / out_w);
                            let v = if (x + y).is_multiple_of(2) {
                                128 - amp / 2
                            } else {
                                128 + amp / 2
                            };
                            [v; 3]
                        })
                        .collect()
                } else {
                    vec![128u8; (out_w * h * 3) as usize]
                };
                RgbFrame { width: out_w, height: h, data }
            })
            .collect())
    }

    fn extract_rgb48(
        &self,
        _path: &Path,
        meta: &VideoMeta,
        frames: &[u32],
        sink: &mut dyn FnMut(Rgb48Frame) -> Result<(), CaptureError>,
    ) -> Result<(), CaptureError> {
        for &i in frames {
            let px = mock_rgb48_px(i);
            let data: Vec<u16> =
                std::iter::repeat_n(px, (meta.width * meta.height) as usize).flatten().collect();
            sink(Rgb48Frame { width: meta.width, height: meta.height, data })?;
        }
        Ok(())
    }

    fn extract_embedded_srt(&self, _path: &Path) -> Result<Option<String>, CaptureError> {
        Ok(None)
    }

    fn decode_image_rgb8(&self, path: &Path) -> Result<image::RgbImage, CaptureError> {
        panic!("mock session has no HEIC: {path:?}");
    }
}

/// Scripted sidecar SRT: one entry per candidate at 2 fps, GPS advancing
/// ~2.2 m per candidate (nothing prunes), rel_alt peaking at frame 20.
fn scripted_srt() -> String {
    let mut out = String::new();
    for i in 0..N_CANDS {
        let (s, e) = (i * 500, (i + 1) * 500);
        let ts = |ms: u32| {
            format!("00:{:02}:{:02},{:03}", ms / 60_000, (ms / 1000) % 60, ms % 1000)
        };
        let rel_alt = if i == 20 { 100.0 } else { 10.0 };
        out.push_str(&format!(
            "{}\n{} --> {}\n[latitude: {:.6}] [longtitude: 23.760000] \
             [rel_alt: {rel_alt:.1} abs_alt: {:.1}] [gb_yaw : 90.0 gb_pitch : -45.0]\n\n",
            i + 1,
            ts(s),
            ts(e),
            61.5 + f64::from(i) * 2e-5,
            110.0 + rel_alt,
        ));
    }
    out
}

fn setup_media(dir: &Path) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    // dummy container: enumeration wants the file, the mock never reads it
    std::fs::write(dir.join("flight.mp4"), b"mock").unwrap();
    std::fs::write(dir.join("flight.srt"), scripted_srt()).unwrap();
    // constant-color photos: two landscape, one portrait (crop warning)
    let photo = |name: &str, w: u32, h: u32, rgb: [u8; 3]| {
        image::RgbImage::from_pixel(w, h, image::Rgb(rgb)).save(dir.join(name)).unwrap();
    };
    photo("p1_land.png", 300, 200, [10, 200, 60]);
    photo("p2_land.png", 300, 200, [200, 10, 60]);
    photo("p3_port.png", 600, 800, [60, 10, 200]);
    dir.to_owned()
}

fn run(media: PathBuf) -> (headshot_capture::PreparedSession, Vec<String>) {
    let cfg = CaptureConfig {
        media: vec![media],
        budget: 30,
        dlog_lut: None,
        dlog_parametric: true,
        params: SelectParams { budget: 30, ..Default::default() },
    };
    let mut log = Vec::new();
    let session = prepare_session(&cfg, &MockBackend, &mut |m| log.push(m)).expect("prepare");
    (session, log)
}

#[test]
fn mock_session_end_to_end() {
    let dir = std::env::temp_dir().join(format!("headshot-mock-{}", std::process::id()));
    let media = setup_media(&dir);
    let (session, log) = run(media.clone());
    let (a, b) = (&session, run(media).0);
    std::fs::remove_dir_all(&dir).ok();

    // ---- determinism ----
    assert_eq!(a.manifest, b.manifest);
    assert_eq!(a.frames, b.frames);

    // ---- sizing gate: dims ÷16, ~1024 tokens, bucket = the video's 16:9 ----
    let (tw, th) = (session.width, session.height);
    assert_eq!((tw % 16, th % 16), (0, 0));
    let tokens = (tw / 16) * (th / 16);
    assert!((990..=1058).contains(&tokens), "{tokens} tokens ({tw}x{th})");
    let aspect = f64::from(th) / f64::from(tw);
    assert!((aspect - 9.0 / 16.0).abs() < 0.03, "bucket not 16:9: {tw}x{th}");

    // ---- structure: 10 video keyframes (windows of 4 → every 4th frame,
    // GPS motion keeps all) + 3 photos, under budget 30 ----
    let m = &session.manifest;
    assert_eq!(m.frames.len(), 13);
    assert_eq!(session.frames.len(), 13);
    assert!(session.frames.iter().all(|f| f.len() == (tw * th * 3) as usize));
    let kinds: Vec<_> = m.frames.iter().map(|f| f.kind).collect();
    assert_eq!(kinds.iter().filter(|&&k| k == SourceKind::Video).count(), 10);
    assert_eq!(kinds.iter().filter(|&&k| k == SourceKind::Photo).count(), 3);

    // ---- reference rule: frame 0 is the mid-sequence max-rel_alt frame ----
    let reference = &m.frames[0];
    assert_eq!(reference.kind, SourceKind::Video);
    assert_eq!(reference.source_frame, Some(20));
    assert_eq!(reference.gps.unwrap().rel_alt_m, Some(100.0));
    assert_eq!(m.reference_original_pos, 5);
    // remaining video frames keep chronological order
    let video_sf: Vec<u32> =
        m.frames[1..].iter().filter_map(|f| f.source_frame).collect();
    assert_eq!(video_sf, vec![0, 4, 8, 12, 16, 24, 28, 32, 36]);

    // ---- telemetry inheritance ----
    for f in m.frames.iter().filter(|f| f.kind == SourceKind::Video) {
        let g = f.gps.expect("every mock video frame has GPS");
        let sf = f.source_frame.unwrap();
        assert!((g.lat - (61.5 + f64::from(sf) * 2e-5)).abs() < 1e-9);
        assert_eq!(f.gimbal_yaw_deg, Some(90.0));
        assert_eq!(f.gimbal_pitch_deg, Some(-45.0));
        assert_eq!(f.tonemap, TonemapKind::Parametric);
        assert!(f.sharpness.unwrap() > 0.0);
    }

    // ---- tonemap applied to video frames only ----
    for (slot, f) in m.frames.iter().enumerate() {
        let frame = &session.frames[slot];
        match f.kind {
            SourceKind::Video => {
                // constant-color mock frame → exact parametric output
                let px = mock_rgb48_px(f.source_frame.unwrap());
                let expected = dlog_to_rec709(px.map(|v| f32::from(v) / 65535.0))
                    .map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8);
                for c in 0..3 {
                    assert!(
                        frame[c].abs_diff(expected[c]) <= 1,
                        "slot {slot} ch {c}: {} vs {}",
                        frame[c],
                        expected[c]
                    );
                }
            }
            _ => {
                // photos pass through untouched (constant color survives
                // crop + Lanczos exactly, up to rounding)
                assert_eq!(f.tonemap, TonemapKind::None);
                let name = PathBuf::from(&f.source);
                let expected: [u8; 3] = match name.file_name().unwrap().to_str().unwrap() {
                    "p1_land.png" => [10, 200, 60],
                    "p2_land.png" => [200, 10, 60],
                    "p3_port.png" => [60, 10, 200],
                    other => panic!("unexpected photo {other}"),
                };
                for c in 0..3 {
                    assert!(frame[c].abs_diff(expected[c]) <= 1, "photo {name:?} ch {c}");
                }
            }
        }
    }

    // ---- portrait photo fired the crop warning ----
    assert!(
        log.iter().any(|l| l.contains("p3_port.png") && l.contains("cropped")),
        "no crop warning in {log:?}"
    );
}
