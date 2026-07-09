//! Shared between the headshot client and server.
//!
//! - [`model`] — VGGT-Omega 1B architecture constants (doc/01).
//! - [`pose`] — pose_enc → extrinsics/intrinsics, unprojection (doc/01 §5).
//! - [`filter`] — confidence/depth-edge point filters (doc/01 §5.3).
//! - [`sizing`] — resize/crop math to model resolution (doc/05 §3).
//! - [`ply`] — binary point-cloud export (doc/06 §4).
//! - `protocol` — client/server message schema (doc/04 §3), added in M3.

pub mod filter;
pub mod model;
pub mod ply;
pub mod pose;
pub mod sizing;
