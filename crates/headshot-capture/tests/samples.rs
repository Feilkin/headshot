//! End-to-end tests against real local captures (never committed; doc/05).
//! Point `HEADSHOT_SAMPLES_DIR` at a directory of DJI clips (+ sidecar
//! `.srt`) and run `just samples`. Determinism assertions pin behavior for
//! one machine's ffmpeg build only — cross-version byte-exactness is not
//! guaranteed by ffmpeg.

use std::path::{Path, PathBuf};

use headshot_capture::keyframe::{
    Candidate, SelectParams, gray_thumb, rgb_to_gray, select_keyframes, sharpness_var_laplacian,
};
use headshot_capture::srt::SrtTrack;
use headshot_capture::video::{FfmpegCli, VideoBackend, candidate_step};

/// All files under `HEADSHOT_SAMPLES_DIR`, recursively, sorted. A
/// relative dir resolves against the workspace root (nextest runs tests
/// with the crate dir as cwd).
fn sample_files() -> Vec<PathBuf> {
    let dir = PathBuf::from(
        std::env::var_os("HEADSHOT_SAMPLES_DIR")
            .expect("set HEADSHOT_SAMPLES_DIR to the local captures directory"),
    );
    let dir = if dir.is_relative() {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(dir)
    } else {
        dir
    };
    let mut files = Vec::new();
    let mut stack = vec![dir];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).expect("samples dir readable").flatten() {
            let p = entry.path();
            if p.is_dir() { stack.push(p) } else { files.push(p) }
        }
    }
    files.sort();
    files
}

fn with_ext(files: &[PathBuf], exts: &[&str]) -> Vec<PathBuf> {
    files
        .iter()
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(str::to_ascii_lowercase)
                .is_some_and(|e| exts.contains(&e.as_str()))
        })
        .cloned()
        .collect()
}

fn sample_videos() -> Vec<PathBuf> {
    let vids = with_ext(&sample_files(), &["mp4", "mov"]);
    assert!(!vids.is_empty(), "no .mp4/.mov files under HEADSHOT_SAMPLES_DIR");
    vids
}

fn candidates_for(backend: &FfmpegCli, video: &Path) -> (Vec<Candidate>, f64) {
    let meta = backend.probe(video).expect("probe");
    let step = candidate_step(meta.fps);
    let thumbs = backend.sample_thumbs(video, &meta, step, 480).expect("pass-1 decode");
    assert!(!thumbs.is_empty());
    let srt_path = video.with_extension("srt");
    let srt = std::fs::read_to_string(&srt_path)
        .ok()
        .and_then(|text| SrtTrack::parse(&srt_path, &text).ok());
    let cands = thumbs
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let source_frame = i as u32 * step;
            let entry = srt.as_ref().and_then(|s| s.at_frame(source_frame, meta.fps));
            let gray = rgb_to_gray(t);
            Candidate {
                source_frame,
                time_s: f64::from(source_frame) / meta.fps,
                sharpness: sharpness_var_laplacian(&gray),
                thumb: gray_thumb(&gray, 64),
                gps: entry.and_then(|e| e.gps),
                gimbal_yaw_deg: entry.and_then(|e| e.gimbal_yaw_deg),
                gimbal_pitch_deg: entry.and_then(|e| e.gimbal_pitch_deg),
            }
        })
        .collect();
    (cands, meta.fps)
}

#[test]
#[ignore = "needs local capture samples (HEADSHOT_SAMPLES_DIR) + ffmpeg"]
fn samples_video_keyframes_deterministic_and_budgeted() {
    let backend = FfmpegCli::default();
    for video in sample_videos() {
        let (cands, _) = candidates_for(&backend, &video);
        for budget in [10, 50, 200] {
            let p = SelectParams { budget, ..Default::default() };
            let a = select_keyframes(&cands, &p);
            let b = select_keyframes(&cands, &p);
            assert_eq!(a, b, "{video:?}: nondeterministic at budget {budget}");
            assert!(a.len() <= budget, "{video:?}: budget {budget} exceeded: {}", a.len());
            assert!(!a.is_empty(), "{video:?}: nothing selected");
            assert!(a.windows(2).all(|w| w[0] < w[1]), "{video:?}: unsorted");
        }
    }
}

#[test]
#[ignore = "needs local capture samples (HEADSHOT_SAMPLES_DIR) + ffmpeg"]
fn samples_embedded_srt_agrees_with_sidecar() {
    let backend = FfmpegCli::default();
    let mut checked = 0;
    for video in sample_videos() {
        let srt_path = video.with_extension("srt");
        let Ok(sidecar_text) = std::fs::read_to_string(&srt_path) else { continue };
        let sidecar = SrtTrack::parse(&srt_path, &sidecar_text).expect("sidecar parses");
        let Some(embedded_text) = backend.extract_embedded_srt(&video).expect("extract") else {
            continue;
        };
        let embedded = SrtTrack::parse(&video, &embedded_text).expect("embedded parses");
        let count_ratio = embedded.entries.len() as f64 / sidecar.entries.len() as f64;
        assert!((0.9..=1.1).contains(&count_ratio), "{video:?}: entry counts diverge");
        if let (Some(a), Some(b)) =
            (embedded.entries[0].gps.as_ref(), sidecar.entries[0].gps.as_ref())
        {
            assert!(headshot_capture::srt::haversine_m(a, b) < 5.0, "{video:?}: first fix differs");
        }
        checked += 1;
    }
    if checked == 0 {
        // a data property of the local samples, not a code failure
        eprintln!("note: no video had both a sidecar .srt and an embedded track — nothing compared");
    }
}

#[test]
#[ignore = "needs local capture samples (HEADSHOT_SAMPLES_DIR) + ffmpeg"]
fn samples_extract_rgb48_full_res() {
    let backend = FfmpegCli::default();
    let video = &sample_videos()[0];
    let meta = backend.probe(video).expect("probe");
    let (cands, _) = candidates_for(&backend, video);
    let sel = select_keyframes(&cands, &SelectParams { budget: 3, ..Default::default() });
    let frames: Vec<u32> = sel.iter().map(|&i| cands[i].source_frame).collect();
    let mut seen = 0usize;
    backend
        .extract_rgb48(video, &meta, &frames, &mut |f| {
            seen += 1;
            assert_eq!((f.width, f.height), (meta.width, meta.height));
            // real footage has nonzero midtones: mean strictly inside the
            // 16-bit range
            let mean = f.data.iter().map(|&v| u64::from(v)).sum::<u64>() / f.data.len() as u64;
            assert!((256..65280).contains(&mean), "{video:?}: implausible mean {mean}");
            Ok(())
        })
        .expect("pass-2 decode");
    assert_eq!(seen, frames.len());
}

#[test]
#[ignore = "needs local capture samples (HEADSHOT_SAMPLES_DIR) + ffmpeg"]
fn samples_raw_develop() {
    let raws = with_ext(&sample_files(), headshot_capture::raw::RAW_EXTENSIONS);
    assert!(!raws.is_empty(), "no RAW files under HEADSHOT_SAMPLES_DIR");
    for raw in raws.iter().take(3) {
        let img = headshot_capture::raw::develop_raw(raw).expect("develop");
        let (w, h) = img.dimensions();
        assert!(w > 1000 && h > 1000, "{raw:?}: implausible dims {w}x{h}");
        let mean = img.pixels().map(|p| u64::from(p[0]) + u64::from(p[1]) + u64::from(p[2]))
            .sum::<u64>()
            / (3 * u64::from(w) * u64::from(h));
        assert!((5..250).contains(&mean), "{raw:?}: implausible mean {mean}");
    }
}
