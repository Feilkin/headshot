//! headshot inference engine library (doc/03).
//!
//! - [`engine`] — wgpu device, tensors, WGSL kernels and ops.
//! - [`weights`] — converted-checkpoint loading and validation (doc/02 §6).
//! - [`parity`] — comparison metrics and fixture loading for the parity
//!   harness (doc/02 §5).

pub mod engine;
pub mod model;
pub mod parity;
pub mod server;
pub mod weights;
