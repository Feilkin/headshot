//! Session assembly (doc/05 §§2–3), in two phases: `plan_session` scores
//! media and auto-selects keyframes into an editable [`SessionPlan`] (no
//! full-resolution work); `realize_session` executes a plan — full-res
//! streaming extraction, preprocessing to uniform RGB8, reference-frame
//! promotion, keyframe manifest. `prepare_session` = plan + realize (the
//! CLI's one-shot path).

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use sha2::Digest;

use crate::error::CaptureError;
use crate::keyframe::{
    Candidate, GrayFrame, RgbFrame, decimate, gray_thumb, rgb_thumb, rgb_to_gray,
    select_survivors, sharpness_var_laplacian,
};
use crate::lut::{Lut3d, Tonemap};
use crate::manifest::{
    KeyframeManifest, KeyframeRecord, MANIFEST_VERSION, SourceKind, TonemapKind,
};
use crate::photo;
use crate::plan::{
    AspectChoice, PlanUnit, PlannedPhoto, PlannedVideo, Selected, SessionPlan, pick_reference,
};
use crate::preprocess;
use crate::srt::SrtTrack;
use crate::video::{VideoBackend, candidate_step};

pub struct CaptureConfig {
    /// Media roots: directories (searched recursively) and/or single
    /// files, e.g. everything dropped onto the app window.
    pub media: Vec<PathBuf>,
    /// Total keyframe budget across all sources (doc/05 §2).
    pub budget: usize,
    /// Official D-Log `.cube`, applied to video frames only.
    pub dlog_lut: Option<PathBuf>,
    /// Parametric whitepaper fallback when no `.cube` is available.
    pub dlog_parametric: bool,
    pub params: crate::keyframe::SelectParams,
}

pub struct PreparedSession {
    pub width: u32,
    pub height: u32,
    /// RGB8 frames, `width·height·3` bytes each; `frames[0]` is the
    /// reference frame.
    pub frames: Vec<Vec<u8>>,
    pub manifest: KeyframeManifest,
}

impl PreparedSession {
    /// Debug dump: `manifest.json` + `kf_0000.png` … (doc/05 §5).
    pub fn dump(&self, dir: &Path) -> Result<(), CaptureError> {
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&self.manifest)?)?;
        for (i, frame) in self.frames.iter().enumerate() {
            image::save_buffer(
                dir.join(format!("kf_{i:04}.png")),
                frame,
                self.width,
                self.height,
                image::ColorType::Rgb8,
            )?;
        }
        Ok(())
    }
}

/// Score all media and auto-select keyframes — everything a UI needs to
/// review; no full-resolution work happens here.
pub fn plan_session(
    cfg: &CaptureConfig,
    backend: &dyn VideoBackend,
    progress: &mut dyn FnMut(String),
) -> Result<SessionPlan, CaptureError> {
    assert!(cfg.budget > 0, "budget must be ≥ 1");
    let (tonemap, tonemap_kind) = build_tonemap(cfg)?;

    let (video_paths, photo_paths) = media_paths(&cfg.media, progress)?;
    let no_media = || {
        CaptureError::NoMedia(cfg.media.first().cloned().unwrap_or_else(|| PathBuf::from(".")))
    };
    if video_paths.is_empty() && photo_paths.is_empty() {
        return Err(no_media());
    }
    progress(format!("{} videos, {} photos", video_paths.len(), photo_paths.len()));

    // ---- score photos (decode once for sharpness/dims/thumb) ----
    // RAW develops take seconds each; fan out across cores, keep order
    let n_photos = photo_paths.len();
    let mut slots: Vec<Option<PlannedPhoto>> = Vec::new();
    slots.resize_with(n_photos, || None);
    let mut first_err: Option<CaptureError> = None;
    if n_photos > 0 {
        let workers = std::thread::available_parallelism().map_or(4, |p| p.get()).min(n_photos);
        let next = std::sync::atomic::AtomicUsize::new(0);
        let (ptx, prx) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            for _ in 0..workers {
                let ptx = ptx.clone();
                let next = &next;
                let paths = &photo_paths;
                scope.spawn(move || {
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= paths.len() {
                            break;
                        }
                        let r = score_photo(&paths[i], backend);
                        if ptx.send((i, r)).is_err() {
                            break;
                        }
                    }
                });
            }
            drop(ptx);
            let mut done = 0usize;
            while let Ok((i, r)) = prx.recv() {
                done += 1;
                match r {
                    Ok(p) => {
                        progress(format!("scored photo {done}/{n_photos}: {}", p.path.display()));
                        slots[i] = Some(p);
                    }
                    Err(e) => first_err = first_err.take().or(Some(e)),
                }
            }
        });
    }
    if let Some(e) = first_err {
        return Err(e);
    }
    let mut photos: Vec<PlannedPhoto> =
        slots.into_iter().map(|s| s.expect("all photos scored")).collect();
    photos.sort_by(|a, b| {
        let key =
            |p: &PlannedPhoto| (p.meta.capture_time.clone().unwrap_or_default(), p.path.clone());
        key(a).cmp(&key(b))
    });
    let times: Vec<Option<i64>> =
        photos.iter().map(|p| p.meta.capture_time.as_deref().and_then(timestamp_secs)).collect();
    let sharp: Vec<f64> = photos.iter().map(|p| p.sharpness).collect();
    let kept = burst_dedup(&times, &sharp);
    if kept.len() < photos.len() {
        progress(format!("burst dedup: {} → {} photos", photos.len(), kept.len()));
    }
    let kept: HashSet<usize> = kept.into_iter().collect();
    for (pi, p) in photos.iter_mut().enumerate() {
        p.kept = kept.contains(&pi);
    }

    // ---- score videos ----
    let mut videos: Vec<PlannedVideo> = Vec::with_capacity(video_paths.len());
    for path in video_paths {
        let meta = backend.probe(&path)?;
        if let Some(t) = &meta.color_transfer
            && matches!(t.as_str(), "arib-std-b67" | "smpte2084")
        {
            progress(format!(
                "{}: HLG/PQ transfer ({t}) — supply a matching .cube or expect wrong contrast",
                path.display()
            ));
        }
        let step = candidate_step(meta.fps);
        progress(format!("decoding candidates from {} …", path.display()));
        let rgb = backend.sample_thumbs(&path, &meta, step, 480)?;
        progress(format!("{}: {} candidates at step {step}", path.display(), rgb.len()));
        let srt = load_telemetry(&path, backend)?;
        if matches!(tonemap, Tonemap::None)
            && srt
                .as_ref()
                .is_some_and(|t| t.entries.iter().any(|e| e.color_md.as_deref() == Some("d_log")))
        {
            progress(format!(
                "{}: telemetry says color_md d_log — pass --dlog-lut <cube> (or --dlog) \
                 or the reconstruction sees flat log footage",
                path.display()
            ));
        }
        let mut cands = Vec::with_capacity(rgb.len());
        let mut thumbs = Vec::with_capacity(rgb.len());
        for (i, t) in rgb.iter().enumerate() {
            let source_frame = i as u32 * step;
            let entry = srt.as_ref().and_then(|s| s.at_frame(source_frame, meta.fps));
            let gray: GrayFrame = rgb_to_gray(t);
            cands.push(Candidate {
                source_frame,
                time_s: f64::from(source_frame) / meta.fps,
                sharpness: sharpness_var_laplacian(&gray),
                thumb: gray_thumb(&gray, 64),
                gps: entry.and_then(|e| e.gps),
                gimbal_yaw_deg: entry.and_then(|e| e.gimbal_yaw_deg),
                gimbal_pitch_deg: entry.and_then(|e| e.gimbal_pitch_deg),
            });
            thumbs.push(rgb_thumb(t, 320));
        }
        let survivors = select_survivors(&cands, &cfg.params);
        videos.push(PlannedVideo { path, meta, cands, thumbs, survivors });
    }

    // ---- auto-selection ----
    let kept_count = photos.iter().filter(|p| p.kept).count();
    if kept_count > cfg.budget {
        progress(format!(
            "budget {} exceeded by photos alone: decimating {kept_count} photos",
            cfg.budget
        ));
    }
    if !videos.is_empty() && cfg.budget.saturating_sub(kept_count.min(cfg.budget)) == 0 {
        progress("budget exhausted by photos; dropping all video frames".into());
    }
    let (selected, reference) = auto_select(&videos, &photos, cfg.budget);
    if selected.is_empty() {
        return Err(no_media());
    }

    Ok(SessionPlan {
        videos,
        photos,
        selected,
        reference,
        budget: cfg.budget,
        aspect: AspectChoice::Auto,
        params: cfg.params.clone(),
        tonemap,
        tonemap_kind,
    })
}

/// Decode + score one photo (parallel-safe; doc/05 §2).
fn score_photo(path: &Path, backend: &dyn VideoBackend) -> Result<PlannedPhoto, CaptureError> {
    let meta = photo::read_photo_meta(path);
    let img = photo::decode_photo(path, backend)?;
    let dims = (img.width(), img.height());
    let full = RgbFrame { width: dims.0, height: dims.1, data: img.into_raw() };
    let scoring = rgb_thumb(&full, 480);
    let is_raw = path.extension().and_then(|e| e.to_str()).is_some_and(crate::raw::is_raw_ext);
    Ok(PlannedPhoto {
        sharpness: sharpness_var_laplacian(&rgb_to_gray(&scoring)),
        thumb: rgb_thumb(&scoring, 320),
        dims,
        is_raw,
        kept: true,
        meta,
        path: path.to_owned(),
    })
}

/// Automatic selection (doc/05 §2): kept photos first, the remaining
/// budget split across videos by largest remainder over survivor counts,
/// each decimated to its share. Pure; returns the chronological selection
/// and the reference frame.
pub(crate) fn auto_select(
    videos: &[PlannedVideo],
    photos: &[PlannedPhoto],
    budget: usize,
) -> (Vec<Selected>, PlanUnit) {
    let mut photo_kept: Vec<usize> =
        (0..photos.len()).filter(|&pi| photos[pi].kept).collect();
    if photo_kept.len() > budget {
        photo_kept = decimate(&photo_kept, budget);
    }
    let video_budget = budget.saturating_sub(photo_kept.len());
    let surv_counts: Vec<usize> = videos.iter().map(|v| v.survivors.len()).collect();
    let shares = split_budget(&surv_counts, video_budget);

    let mut selected: Vec<Selected> = Vec::new();
    for (vi, (v, &share)) in videos.iter().zip(&shares).enumerate() {
        for &ci in &decimate(&v.survivors, share) {
            selected.push(Selected { unit: PlanUnit::Video { vi, ci }, crop_scale: 1.0 });
        }
    }
    selected.extend(
        photo_kept.iter().map(|&pi| Selected { unit: PlanUnit::Photo { pi }, crop_scale: 1.0 }),
    );
    if selected.is_empty() {
        // degenerate (budget 0 handled upstream); pick something stable
        return (Vec::new(), PlanUnit::Photo { pi: 0 });
    }
    let reference = pick_reference(&selected, videos);
    (selected, reference)
}

/// Execute a (possibly user-edited) plan: full-res extraction, preprocess,
/// reference promotion to batch 0, manifest.
pub fn realize_session(
    plan: &SessionPlan,
    backend: &dyn VideoBackend,
    progress: &mut dyn FnMut(String),
) -> Result<PreparedSession, CaptureError> {
    let warnings = plan.validate().map_err(CaptureError::Plan)?;
    for w in warnings {
        progress(w);
    }
    let reference_original_pos =
        plan.selection_index(plan.reference).expect("validated: reference is selected");

    // batch order: reference first, everything else chronological
    let mut batch: Vec<Selected> = Vec::with_capacity(plan.selected.len());
    batch.push(plan.selected[reference_original_pos]);
    batch.extend(
        plan.selected.iter().enumerate().filter(|&(i, _)| i != reference_original_pos).map(|(_, s)| *s),
    );

    let (tw, th) = plan.target_size();
    progress(format!("session target {tw}x{th}, {} frames", batch.len()));
    let px = (tw * th * 3) as usize;
    let mut frames: Vec<Vec<u8>> = vec![Vec::new(); batch.len()];
    let mut crops: Vec<[u32; 4]> = vec![[0; 4]; batch.len()];

    for (vi, v) in plan.videos.iter().enumerate() {
        let mut slots: Vec<(u32, usize, f32)> = batch
            .iter()
            .enumerate()
            .filter_map(|(slot, s)| match s.unit {
                PlanUnit::Video { vi: uvi, ci } if uvi == vi => {
                    Some((v.cands[ci].source_frame, slot, s.crop_scale))
                }
                _ => None,
            })
            .collect();
        if slots.is_empty() {
            continue;
        }
        slots.sort_by_key(|s| s.0);
        let frame_ids: Vec<u32> = slots.iter().map(|&(f, _, _)| f).collect();
        progress(format!("{}: extracting {} keyframes", v.path.display(), frame_ids.len()));
        let mut k = 0usize;
        backend.extract_rgb48(&v.path, &v.meta, &frame_ids, &mut |rgb48| {
            let (_, slot, crop_scale) = slots[k];
            let (rgb8, crop) =
                preprocess::preprocess_rgb48(&rgb48, tw, th, &plan.tonemap, crop_scale);
            frames[slot] = rgb8;
            crops[slot] = crop;
            k += 1;
            Ok(())
        })?;
    }
    for (slot, s) in batch.iter().enumerate() {
        let PlanUnit::Photo { pi } = s.unit else { continue };
        let p = &plan.photos[pi];
        let img = photo::decode_photo(&p.path, backend)?;
        let (rgb8, crop) = preprocess::preprocess_rgb8(&img, tw, th, s.crop_scale);
        // warn only about aspect-forced loss, not a deliberate zoom
        let loss = preprocess::crop_loss(img.width(), img.height(), crop);
        if loss > 0.01 && s.crop_scale >= 0.999 {
            progress(format!(
                "{}: cropped {:.0}% to fit the session aspect",
                p.path.display(),
                loss * 100.0
            ));
        }
        frames[slot] = rgb8;
        crops[slot] = crop;
    }
    debug_assert!(frames.iter().all(|f| f.len() == px));

    let records: Vec<KeyframeRecord> = batch
        .iter()
        .enumerate()
        .map(|(slot, s)| match s.unit {
            PlanUnit::Video { vi, ci } => {
                let (v, c) = (&plan.videos[vi], &plan.videos[vi].cands[ci]);
                KeyframeRecord {
                    batch_index: slot as u32,
                    source: v.path.display().to_string(),
                    kind: SourceKind::Video,
                    source_frame: Some(c.source_frame),
                    time_s: Some(c.time_s),
                    capture_time: None,
                    gps: c.gps,
                    gimbal_yaw_deg: c.gimbal_yaw_deg,
                    gimbal_pitch_deg: c.gimbal_pitch_deg,
                    sharpness: Some(c.sharpness),
                    tonemap: plan.tonemap_kind.clone(),
                    crop: crops[slot],
                }
            }
            PlanUnit::Photo { pi } => {
                let p = &plan.photos[pi];
                KeyframeRecord {
                    batch_index: slot as u32,
                    source: p.path.display().to_string(),
                    kind: if p.is_raw { SourceKind::RawPhoto } else { SourceKind::Photo },
                    source_frame: None,
                    time_s: None,
                    capture_time: p.meta.capture_time.clone(),
                    gps: p.meta.gps,
                    gimbal_yaw_deg: None,
                    gimbal_pitch_deg: None,
                    sharpness: Some(p.sharpness),
                    tonemap: TonemapKind::None,
                    crop: crops[slot],
                }
            }
        })
        .collect();

    Ok(PreparedSession {
        width: tw,
        height: th,
        frames,
        manifest: KeyframeManifest {
            version: MANIFEST_VERSION,
            target_width: tw,
            target_height: th,
            budget: plan.budget,
            reference_original_pos,
            frames: records,
        },
    })
}

/// One-shot media set → frame batch + manifest (the CLI path).
pub fn prepare_session(
    cfg: &CaptureConfig,
    backend: &dyn VideoBackend,
    progress: &mut dyn FnMut(String),
) -> Result<PreparedSession, CaptureError> {
    let plan = plan_session(cfg, backend, progress)?;
    realize_session(&plan, backend, progress)
}

fn build_tonemap(cfg: &CaptureConfig) -> Result<(Tonemap, TonemapKind), CaptureError> {
    if let Some(path) = &cfg.dlog_lut {
        let text = std::fs::read_to_string(path)?;
        let lut = Lut3d::parse_cube(path, &text)?;
        let sha256 = format!("{:x}", sha2::Sha256::digest(text.as_bytes()));
        return Ok((
            Tonemap::Cube(lut),
            TonemapKind::Cube { file: path.display().to_string(), sha256 },
        ));
    }
    if cfg.dlog_parametric {
        return Ok((Tonemap::Parametric, TonemapKind::Parametric));
    }
    Ok((Tonemap::None, TonemapKind::None))
}

/// Cheap discovery pass (no decoding): what a scan of `roots` would
/// ingest, as `(videos, photos)`. The UI shows this as an excludable tree
/// before committing to scoring.
pub fn discover_media(
    roots: &[PathBuf],
    progress: &mut dyn FnMut(String),
) -> Result<(Vec<PathBuf>, Vec<PathBuf>), CaptureError> {
    media_paths(roots, progress)
}

/// Recursively enumerate media files under all roots; RAW+JPEG pairs
/// (same directory and stem) keep only the RAW.
fn media_paths(
    roots: &[PathBuf],
    progress: &mut dyn FnMut(String),
) -> Result<(Vec<PathBuf>, Vec<PathBuf>), CaptureError> {
    let mut videos = Vec::new();
    let mut photos = Vec::new();
    let mut classify = |p: PathBuf| {
        let ext =
            p.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).unwrap_or_default();
        match ext.as_str() {
            "mp4" | "mov" => videos.push(p),
            "jpg" | "jpeg" | "png" | "tif" | "tiff" | "heic" | "heif" => photos.push(p),
            e if crate::raw::is_raw_ext(e) => photos.push(p),
            _ => {}
        }
    };
    let mut stack: Vec<PathBuf> = Vec::new();
    for root in roots {
        if root.is_file() {
            classify(root.to_owned());
        } else {
            stack.push(root.to_owned());
        }
    }
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)?.flatten() {
            let p = entry.path();
            let hidden =
                p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with('.'));
            if hidden {
                continue;
            }
            if p.is_dir() { stack.push(p) } else { classify(p) }
        }
    }
    videos.sort();
    videos.dedup();
    photos.sort();
    photos.dedup();

    let pair_key = |p: &Path| {
        (
            p.parent().map(Path::to_owned).unwrap_or_default(),
            p.file_stem().and_then(|s| s.to_str()).map(str::to_ascii_lowercase).unwrap_or_default(),
        )
    };
    let is_raw =
        |p: &Path| p.extension().and_then(|e| e.to_str()).is_some_and(crate::raw::is_raw_ext);
    let raw_stems: HashSet<_> = photos.iter().filter(|p| is_raw(p)).map(|p| pair_key(p)).collect();
    let before = photos.len();
    photos.retain(|p| is_raw(p) || !raw_stems.contains(&pair_key(p)));
    if photos.len() < before {
        progress(format!("{} RAW+JPEG pairs: keeping the RAW", before - photos.len()));
    }
    Ok((videos, photos))
}

/// Sidecar `.srt`/`.SRT` wins; otherwise the embedded subtitle track.
fn load_telemetry(
    path: &Path,
    backend: &dyn VideoBackend,
) -> Result<Option<SrtTrack>, CaptureError> {
    for ext in ["srt", "SRT"] {
        let sidecar = path.with_extension(ext);
        if let Ok(text) = std::fs::read_to_string(&sidecar) {
            return Ok(Some(SrtTrack::parse(&sidecar, &text)?));
        }
    }
    match backend.extract_embedded_srt(path)? {
        // a video without telemetry is fine; an embedded track that parses
        // to nothing is treated the same
        Some(text) => Ok(SrtTrack::parse(path, &text).ok()),
        None => Ok(None),
    }
}

/// Largest-remainder proportional split of `total` across `surv` counts,
/// min 2 per non-empty video when the total allows; deterministic.
fn split_budget(surv: &[usize], total: usize) -> Vec<usize> {
    let n = surv.len();
    let sum: usize = surv.iter().sum();
    if n == 0 || sum == 0 || total == 0 {
        return vec![0; n];
    }
    let mut share = vec![0usize; n];
    let mut rem: Vec<(f64, usize)> = Vec::with_capacity(n);
    let mut used = 0;
    for i in 0..n {
        let quota = total as f64 * surv[i] as f64 / sum as f64;
        share[i] = (quota.floor() as usize).min(surv[i]);
        used += share[i];
        rem.push((quota - quota.floor(), i));
    }
    rem.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut left = total.saturating_sub(used);
    while left > 0 {
        let mut progressed = false;
        for &(_, i) in &rem {
            if left == 0 {
                break;
            }
            if share[i] < surv[i] {
                share[i] += 1;
                left -= 1;
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }
    // min-2 floor (doc plan) — best effort, then trim any overshoot from
    // the largest shares, later video first among equals so earlier
    // sources keep their frames
    for i in 0..n {
        if surv[i] > 0 {
            share[i] = share[i].max(2.min(surv[i]));
        }
    }
    while share.iter().sum::<usize>() > total {
        let i = (0..n).max_by(|&a, &b| share[a].cmp(&share[b]).then(a.cmp(&b))).expect("n > 0");
        if share[i] == 0 {
            break;
        }
        share[i] -= 1;
    }
    share
}

/// `YYYY-MM-DDTHH:MM:SS` → seconds since a fixed epoch (differences only;
/// bursts come from one camera clock, so no timezone handling).
pub(crate) fn timestamp_secs(t: &str) -> Option<i64> {
    if t.len() < 19 {
        return None;
    }
    let num = |r: std::ops::Range<usize>| t.get(r)?.parse::<i64>().ok();
    let (y, mo, d) = (num(0..4)?, num(5..7)?, num(8..10)?);
    let (h, mi, s) = (num(11..13)?, num(14..16)?, num(17..19)?);
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) {
        return None;
    }
    // days-from-civil (Hinnant)
    let y2 = if mo <= 2 { y - 1 } else { y };
    let era = if y2 >= 0 { y2 } else { y2 - 399 } / 400;
    let yoe = y2 - era * 400;
    let mp = (mo + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(days * 86_400 + h * 3600 + mi * 60 + s)
}

/// Burst dedup (doc/05 §2): consecutive photos < 2 s apart chain into a
/// group; keep each group's sharpest (tie → earlier). Inputs are in
/// (time, filename) order; photos without timestamps never group.
pub(crate) fn burst_dedup(times: &[Option<i64>], sharpness: &[f64]) -> Vec<usize> {
    assert_eq!(times.len(), sharpness.len());
    let mut kept = Vec::new();
    let mut group_start = 0usize;
    for i in 1..=times.len() {
        let boundary = i == times.len()
            || match (times[i - 1], times[i]) {
                (Some(a), Some(b)) => b - a >= 2,
                _ => true,
            };
        if boundary {
            let best = (group_start..i)
                .max_by(|&a, &b| sharpness[a].total_cmp(&sharpness[b]).then(b.cmp(&a)))
                .expect("non-empty group");
            kept.push(best);
            group_start = i;
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_secs_diffs() {
        let t = |s: &str| timestamp_secs(s).unwrap();
        assert_eq!(t("2026-07-10T09:15:01") - t("2026-07-10T09:15:00"), 1);
        assert_eq!(t("2026-07-10T09:16:00") - t("2026-07-10T09:15:00"), 60);
        assert_eq!(t("2026-07-11T00:00:00") - t("2026-07-10T00:00:00"), 86_400);
        // leap year 2028
        assert_eq!(t("2028-03-01T00:00:00") - t("2028-02-28T00:00:00"), 2 * 86_400);
        assert_eq!(timestamp_secs("garbage"), None);
    }

    #[test]
    fn burst_dedup_groups_and_ties() {
        // three-shot burst (0,1s,2s-chained), a loner, and a no-time photo
        let times = [Some(100), Some(101), Some(102), Some(200), None];
        let sharp = [5.0, 9.0, 1.0, 3.0, 2.0];
        assert_eq!(burst_dedup(&times, &sharp), vec![1, 3, 4]);
        // tie inside a burst → earlier photo
        let times = [Some(100), Some(101)];
        assert_eq!(burst_dedup(&times, &[7.0, 7.0]), vec![0]);
        // missing times never group
        let times = [None, None, None];
        assert_eq!(burst_dedup(&times, &[1.0, 2.0, 3.0]), vec![0, 1, 2]);
        assert_eq!(burst_dedup(&[], &[]), Vec::<usize>::new());
    }

    #[test]
    fn split_budget_properties() {
        // proportionality + min-2 floor
        assert_eq!(split_budget(&[30, 10], 20), vec![15, 5]);
        assert_eq!(split_budget(&[30, 1], 20), vec![19, 1]);
        assert_eq!(split_budget(&[5, 5, 5], 4), vec![2, 1, 1]); // forced under floor
        assert_eq!(split_budget(&[], 10), Vec::<usize>::new());
        assert_eq!(split_budget(&[0, 8], 10), vec![0, 8]);
        for (surv, total) in
            [(vec![13, 7, 29], 25usize), (vec![1, 1, 1], 2), (vec![100, 3], 200), (vec![4], 0)]
        {
            let share = split_budget(&surv, total);
            let sum: usize = share.iter().sum();
            assert!(sum <= total, "{surv:?}/{total}: sum {sum}");
            assert!(
                share.iter().zip(&surv).all(|(&s, &c)| s <= c),
                "{surv:?}/{total}: {share:?} exceeds survivors"
            );
            // deterministic
            assert_eq!(share, split_budget(&surv, total));
        }
        // full coverage when the budget allows
        assert_eq!(split_budget(&[3, 4], 100), vec![3, 4]);
    }
}
