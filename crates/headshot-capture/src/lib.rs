//! Capture preprocessing (doc/05 §§1–3): everything between raw media and
//! the uniform RGB8 frame batch a session uploads — video decode, D-Log
//! tonemapping, RAW development, keyframe selection, SRT telemetry, and the
//! keyframe manifest that the metric-scale stage (doc/06) consumes.

pub mod assemble;
pub mod error;
pub mod keyframe;
pub mod lut;
pub mod manifest;
pub mod photo;
pub mod plan;
pub mod preprocess;
pub mod raw;
pub mod srt;
pub mod video;

pub use assemble::{
    CaptureConfig, PreparedSession, discover_media, plan_session, prepare_session,
    realize_session,
};
pub use error::CaptureError;
pub use plan::{AspectChoice, PlanUnit, SessionPlan};
