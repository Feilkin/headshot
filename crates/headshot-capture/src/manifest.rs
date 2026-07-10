//! Keyframe manifest (doc/05 §2, doc/06 §4): the record of where every
//! uploaded frame came from — source file, timestamp, GPS, sharpness,
//! tonemap, crop. This is the input contract for GPS Sim(3) alignment
//! (doc/06 §2) and part of the session bundle.

use serde::{Deserialize, Serialize};

use crate::srt::GpsFix;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceKind {
    Video,
    RawPhoto,
    Photo,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TonemapKind {
    /// User-supplied `.cube`, hashed for reproducibility.
    Cube { file: String, sha256: String },
    /// Whitepaper D-Log approximation.
    Parametric,
    None,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeyframeManifest {
    pub version: u32,
    pub target_width: u32,
    pub target_height: u32,
    pub budget: usize,
    /// Original chronological position of the frame promoted to batch
    /// index 0 (the reference frame; doc/05 §2).
    pub reference_original_pos: usize,
    /// `frames[0]` is the reference frame.
    pub frames: Vec<KeyframeRecord>,
}

pub const MANIFEST_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeyframeRecord {
    pub batch_index: u32,
    /// Source path as given.
    pub source: String,
    pub kind: SourceKind,
    /// Source video frame number (videos only).
    pub source_frame: Option<u32>,
    /// Seconds into the source video.
    pub time_s: Option<f64>,
    /// EXIF `DateTimeOriginal`, `YYYY-MM-DDTHH:MM:SS` (photos only).
    pub capture_time: Option<String>,
    /// SRT telemetry (video) or EXIF GPS (photo). Drone keyframes for
    /// doc/06 §2 are `kind == Video && gps.is_some()`.
    pub gps: Option<GpsFix>,
    pub gimbal_yaw_deg: Option<f64>,
    pub gimbal_pitch_deg: Option<f64>,
    /// Variance-of-Laplacian at scoring resolution.
    pub sharpness: Option<f64>,
    pub tonemap: TonemapKind,
    /// `(x, y, w, h)` center-crop applied on the source before resize.
    pub crop: [u32; 4],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_json_round_trip() {
        let m = KeyframeManifest {
            version: MANIFEST_VERSION,
            target_width: 624,
            target_height: 416,
            budget: 200,
            reference_original_pos: 17,
            frames: vec![
                KeyframeRecord {
                    batch_index: 0,
                    source: "DJI_0001.MP4".into(),
                    kind: SourceKind::Video,
                    source_frame: Some(510),
                    time_s: Some(17.017),
                    capture_time: None,
                    gps: Some(GpsFix {
                        lat: 61.498611,
                        lon: 23.760556,
                        rel_alt_m: Some(30.7),
                        abs_alt_m: Some(142.986),
                    }),
                    gimbal_yaw_deg: Some(-12.3),
                    gimbal_pitch_deg: Some(-89.9),
                    sharpness: Some(812.5),
                    tonemap: TonemapKind::Parametric,
                    crop: [0, 0, 2688, 1512],
                },
                KeyframeRecord {
                    batch_index: 1,
                    source: "villa_nikon/DSC_0042.NEF".into(),
                    kind: SourceKind::RawPhoto,
                    source_frame: None,
                    time_s: None,
                    capture_time: Some("2026-07-10T09:15:00".into()),
                    gps: None,
                    gimbal_yaw_deg: None,
                    gimbal_pitch_deg: None,
                    sharpness: Some(4031.0),
                    tonemap: TonemapKind::None,
                    crop: [0, 504, 6048, 4024],
                },
            ],
        };
        let json = serde_json::to_string_pretty(&m).unwrap();
        let back: KeyframeManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);
        assert!(json.contains("rel_alt_m"));
    }
}
