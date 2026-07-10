//! Error type for the capture pipeline (doc/05). Application code
//! (headshot-client) wraps these in `anyhow` at the session boundary.

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    #[error("ffmpeg/ffprobe not found on PATH (need ffmpeg ≥ 6): {0}")]
    FfmpegMissing(#[source] std::io::Error),

    #[error("{tool} failed ({status}) on {path}: {stderr_tail}")]
    Ffmpeg { tool: &'static str, status: i32, path: PathBuf, stderr_tail: String },

    #[error("bad ffprobe output for {path}: {reason}")]
    Probe { path: PathBuf, reason: String },

    #[error("decode error on {path}: {reason}")]
    Decode { path: PathBuf, reason: String },

    #[error("bad .cube LUT {path}: {reason}")]
    Lut { path: PathBuf, reason: String },

    #[error("bad SRT {path}: {reason}")]
    Srt { path: PathBuf, reason: String },

    #[error("RAW not supported ({0}); convert to DNG (Adobe DNG Converter) or JPEG and retry")]
    RawUnsupported(String),

    #[error("no usable media under {0}")]
    NoMedia(PathBuf),

    #[error("invalid session plan: {0}")]
    Plan(String),

    #[error("manifest serialization: {0}")]
    Manifest(#[from] serde_json::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Image(#[from] image::ImageError),
}
