//! Editable session plan (doc/05 §§2–3): everything decided before any
//! full-resolution work — scored sources, the keyframe selection, session
//! aspect, reference frame — so a UI can review and edit it before
//! `realize_session` executes it. All editing methods are pure state
//! changes on the plan.

use std::path::PathBuf;

use headshot_shared::sizing;

use crate::keyframe::{Candidate, RgbFrame, SelectParams};
use crate::lut::Tonemap;
use crate::manifest::TonemapKind;
use crate::photo::PhotoMeta;
use crate::video::VideoMeta;

/// One scored video source: pass-1 candidates + UI thumbnails.
pub struct PlannedVideo {
    pub path: PathBuf,
    pub meta: VideoMeta,
    pub cands: Vec<Candidate>,
    /// ~320 px RGB thumbnails, index-aligned with `cands`.
    pub thumbs: Vec<RgbFrame>,
    /// Stage-1+2 survivor candidate indices (the auto-selection pool).
    pub survivors: Vec<usize>,
}

/// One scored photo. Burst-rejected photos stay visible (`kept: false`)
/// so the UI can re-include them.
pub struct PlannedPhoto {
    pub path: PathBuf,
    pub meta: PhotoMeta,
    pub is_raw: bool,
    pub sharpness: f64,
    /// Upright (orientation-applied) dimensions.
    pub dims: (u32, u32),
    pub thumb: RgbFrame,
    /// Survived burst dedup (doc/05 §2).
    pub kept: bool,
}

/// A selectable frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanUnit {
    Video { vi: usize, ci: usize },
    Photo { pi: usize },
}

/// One selected frame with its centered zoom (doc/05 §3: the crop must
/// stay centered; shrinking it only narrows FoV, which the camera head
/// estimates per frame).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Selected {
    pub unit: PlanUnit,
    /// 1.0 = the full centered crop at the session aspect.
    pub crop_scale: f32,
}

pub const CROP_SCALE_RANGE: std::ops::RangeInclusive<f32> = 0.3..=1.0;

/// Session aspect choice — ONE aspect for all frames (the server rejects
/// mixed frame sizes; doc/05 §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AspectChoice {
    /// The source file contributing the most selected frames (doc/05 §3).
    Auto,
    Video(usize),
    Photo(usize),
}

pub struct SessionPlan {
    pub videos: Vec<PlannedVideo>,
    /// Sorted by (EXIF time, filename).
    pub photos: Vec<PlannedPhoto>,
    /// Chronological selection (videos in path order, then photos). The
    /// reference is a member here; realize promotes it to batch index 0.
    pub selected: Vec<Selected>,
    pub reference: PlanUnit,
    pub budget: usize,
    pub aspect: AspectChoice,
    pub params: SelectParams,
    pub(crate) tonemap: Tonemap,
    pub(crate) tonemap_kind: TonemapKind,
}

impl SessionPlan {
    /// Uniform frame size the session will upload, from the aspect choice.
    pub fn target_size(&self) -> (u32, u32) {
        let (w, h) = match self.aspect {
            AspectChoice::Video(vi) => (self.videos[vi].meta.width, self.videos[vi].meta.height),
            AspectChoice::Photo(pi) => self.photos[pi].dims,
            AspectChoice::Auto => self.dominant_dims(),
        };
        let (_, _, tw, th) = sizing::target_size(w, h);
        (tw, th)
    }

    /// Native dims of the source file contributing the most selected
    /// frames (tie: video beats photo, then smaller path).
    fn dominant_dims(&self) -> (u32, u32) {
        let mut video_counts = vec![0usize; self.videos.len()];
        let mut best_photo: Option<usize> = None;
        for s in &self.selected {
            match s.unit {
                PlanUnit::Video { vi, .. } => video_counts[vi] += 1,
                PlanUnit::Photo { pi } => {
                    best_photo = Some(best_photo.map_or(pi, |b| {
                        if self.photos[pi].path < self.photos[b].path { pi } else { b }
                    }));
                }
            }
        }
        let best_video = (0..self.videos.len()).filter(|&i| video_counts[i] > 0).max_by(|&a, &b| {
            video_counts[a]
                .cmp(&video_counts[b])
                .then(self.videos[b].path.cmp(&self.videos[a].path))
        });
        match (best_video, best_photo) {
            (Some(vi), _) => (self.videos[vi].meta.width, self.videos[vi].meta.height),
            (None, Some(pi)) => self.photos[pi].dims,
            (None, None) => (self.videos.first().map_or(640, |v| v.meta.width),
                self.videos.first().map_or(480, |v| v.meta.height)),
        }
    }

    /// Position of `unit` in the selection, if selected.
    pub fn selection_index(&self, unit: PlanUnit) -> Option<usize> {
        self.selected.iter().position(|s| s.unit == unit)
    }

    /// Include or exclude a frame. Inclusion inserts at the chronological
    /// position with crop scale 1.0; excluding the reference re-picks it
    /// from what remains.
    pub fn set_included(&mut self, unit: PlanUnit, on: bool) {
        match (self.selection_index(unit), on) {
            (Some(_), true) | (None, false) => {}
            (None, true) => {
                let key = self.order_key(unit);
                let at = self
                    .selected
                    .iter()
                    .position(|s| self.order_key(s.unit) > key)
                    .unwrap_or(self.selected.len());
                self.selected.insert(at, Selected { unit, crop_scale: 1.0 });
            }
            (Some(at), false) => {
                self.selected.remove(at);
                if self.reference == unit && !self.selected.is_empty() {
                    self.reference = pick_reference(&self.selected, &self.videos);
                }
            }
        }
    }

    /// Set the centered zoom for a selected frame; returns false when the
    /// frame isn't selected. The scale is clamped to `CROP_SCALE_RANGE`.
    pub fn set_crop_scale(&mut self, unit: PlanUnit, scale: f32) -> bool {
        let Some(at) = self.selection_index(unit) else { return false };
        self.selected[at].crop_scale =
            scale.clamp(*CROP_SCALE_RANGE.start(), *CROP_SCALE_RANGE.end());
        true
    }

    /// Re-run the automatic selection at the current budget. Discards
    /// manual includes/excludes and crop scales.
    pub fn reselect(&mut self) {
        let (selected, reference) =
            crate::assemble::auto_select(&self.videos, &self.photos, self.budget);
        self.selected = selected;
        self.reference = reference;
    }

    /// Sanity checks before realize; `Ok` carries non-fatal warnings.
    pub fn validate(&self) -> Result<Vec<String>, String> {
        if self.selected.is_empty() {
            return Err("no frames selected".into());
        }
        if self.selection_index(self.reference).is_none() {
            return Err("reference frame is not selected".into());
        }
        let mut warnings = Vec::new();
        if self.selected.len() < 2 {
            warnings.push("fewer than 2 frames: no multi-view geometry".into());
        }
        if self.selected.len() > self.budget {
            warnings.push(format!(
                "{} frames exceed the budget of {} (server cost is quadratic)",
                self.selected.len(),
                self.budget
            ));
        }
        Ok(warnings)
    }

    /// Chronological ordering: videos in path order (already sorted at
    /// scan) by source frame, then photos in kept order.
    pub(crate) fn order_key(&self, unit: PlanUnit) -> (u8, usize, u32) {
        match unit {
            PlanUnit::Video { vi, ci } => (0, vi, self.videos[vi].cands[ci].source_frame),
            PlanUnit::Photo { pi } => (1, pi, 0),
        }
    }
}

/// Reference frame (doc/05 §2): among the selected video frames' middle
/// half, the highest relative altitude (a scene-overview drone frame);
/// else that slice's center; photos-only sessions keep their first frame
/// (preserves the M2/M3 golden ordering).
pub(crate) fn pick_reference(selected: &[Selected], videos: &[PlannedVideo]) -> PlanUnit {
    let video_units: Vec<PlanUnit> = selected
        .iter()
        .map(|s| s.unit)
        .filter(|u| matches!(u, PlanUnit::Video { .. }))
        .collect();
    let n = video_units.len();
    if n == 0 {
        return selected[0].unit;
    }
    let mid = &video_units[n / 4..(3 * n / 4).max(n / 4 + 1)];
    let rel_alt = |u: &PlanUnit| match u {
        PlanUnit::Video { vi, ci } => videos[*vi].cands[*ci].gps.and_then(|g| g.rel_alt_m),
        PlanUnit::Photo { .. } => None,
    };
    mid.iter()
        .enumerate()
        .filter(|(_, u)| rel_alt(u).is_some())
        .max_by(|(ai, a), (bi, b)| {
            rel_alt(a)
                .expect("filtered")
                .total_cmp(&rel_alt(b).expect("filtered"))
                .then(bi.cmp(ai))
        })
        .map(|(_, u)| *u)
        .unwrap_or(mid[mid.len() / 2])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keyframe::GrayFrame;
    use crate::srt::GpsFix;

    fn rgb(w: u32, h: u32) -> RgbFrame {
        RgbFrame { width: w, height: h, data: vec![128; (w * h * 3) as usize] }
    }

    fn video(path: &str, n: usize, rel_alt_peak: Option<usize>) -> PlannedVideo {
        let cands = (0..n)
            .map(|i| Candidate {
                source_frame: i as u32,
                time_s: i as f64 / 2.0,
                sharpness: i as f64,
                thumb: GrayFrame { width: 4, height: 4, data: vec![0; 16] },
                gps: Some(GpsFix {
                    lat: 61.5 + i as f64 * 2e-5,
                    lon: 23.7,
                    rel_alt_m: Some(if rel_alt_peak == Some(i) { 100.0 } else { 10.0 }),
                    abs_alt_m: None,
                }),
                gimbal_yaw_deg: Some(90.0),
                gimbal_pitch_deg: None,
            })
            .collect::<Vec<_>>();
        PlannedVideo {
            path: path.into(),
            meta: crate::video::VideoMeta {
                width: 1920,
                height: 1080,
                fps: 2.0,
                n_frames: None,
                duration_s: None,
                color_range: None,
                color_transfer: None,
                subtitle_stream: None,
            },
            survivors: (0..n).collect(),
            thumbs: (0..n).map(|_| rgb(8, 4)).collect(),
            cands,
        }
    }

    fn photo(path: &str, dims: (u32, u32)) -> PlannedPhoto {
        PlannedPhoto {
            path: path.into(),
            meta: Default::default(),
            is_raw: false,
            sharpness: 1.0,
            dims,
            thumb: rgb(8, 8),
            kept: true,
        }
    }

    fn test_plan() -> SessionPlan {
        let videos = vec![video("a.mp4", 8, Some(4))];
        let photos = vec![photo("p1.png", (3000, 2000)), photo("p2.png", (2000, 3000))];
        let (selected, reference) = crate::assemble::auto_select(&videos, &photos, 30);
        SessionPlan {
            videos,
            photos,
            selected,
            reference,
            budget: 30,
            aspect: AspectChoice::Auto,
            params: crate::keyframe::SelectParams::default(),
            tonemap: Tonemap::None,
            tonemap_kind: TonemapKind::None,
        }
    }

    #[test]
    fn include_exclude_round_trip_and_ordering() {
        let mut p = test_plan();
        let n0 = p.selected.len();
        assert_eq!(n0, 8 + 2, "all survivors + both photos under budget");
        let unit = PlanUnit::Video { vi: 0, ci: 3 };
        assert!(p.selection_index(unit).is_some());

        p.set_included(unit, false);
        assert_eq!(p.selected.len(), n0 - 1);
        p.set_included(unit, false); // idempotent
        assert_eq!(p.selected.len(), n0 - 1);

        p.set_included(unit, true);
        assert_eq!(p.selected.len(), n0);
        // chronological re-insertion: video frames ascend, photos last
        let keys: Vec<_> = p.selected.iter().map(|s| p.order_key(s.unit)).collect();
        assert!(keys.windows(2).all(|w| w[0] < w[1]), "{keys:?}");
        assert_eq!(p.selected[p.selection_index(unit).unwrap()].crop_scale, 1.0);
    }

    #[test]
    fn excluding_reference_repicks_it() {
        let mut p = test_plan();
        // rel_alt peaks at candidate 4 in the middle half → reference
        assert_eq!(p.reference, PlanUnit::Video { vi: 0, ci: 4 });
        p.set_included(p.reference, false);
        assert!(p.selection_index(p.reference).is_some(), "new reference is selected");
        assert_ne!(p.reference, PlanUnit::Video { vi: 0, ci: 4 });
        assert!(p.validate().is_ok());
    }

    #[test]
    fn crop_scale_clamps() {
        let mut p = test_plan();
        let unit = p.selected[0].unit;
        assert!(p.set_crop_scale(unit, 5.0));
        assert_eq!(p.selected[0].crop_scale, 1.0);
        assert!(p.set_crop_scale(unit, 0.0));
        assert_eq!(p.selected[0].crop_scale, 0.3);
        assert!(p.set_crop_scale(unit, 0.5));
        assert_eq!(p.selected[0].crop_scale, 0.5);
        assert!(!p.set_crop_scale(PlanUnit::Photo { pi: 99 }, 0.5));
    }

    #[test]
    fn reselect_is_deterministic_and_resets_edits() {
        let mut p = test_plan();
        let auto = p.selected.clone();
        p.set_included(PlanUnit::Video { vi: 0, ci: 2 }, false);
        p.set_crop_scale(p.selected[0].unit, 0.5);
        p.reselect();
        assert_eq!(p.selected, auto);
        p.budget = 3;
        p.reselect();
        assert!(p.selected.len() <= 3);
    }

    #[test]
    fn validate_and_target_size() {
        let mut p = test_plan();
        assert_eq!(p.validate().unwrap(), Vec::<String>::new());
        // 16:9 video dominates → 688x384 (43×24 = 1032 tokens)
        assert_eq!(p.target_size(), (688, 384));
        // explicit landscape photo aspect → 3:2 bucket
        p.aspect = AspectChoice::Photo(0);
        assert_eq!(p.target_size(), (624, 416));

        let units: Vec<PlanUnit> = p.selected.iter().map(|s| s.unit).collect();
        for u in units {
            p.set_included(u, false);
        }
        assert!(p.validate().is_err());
    }
}
