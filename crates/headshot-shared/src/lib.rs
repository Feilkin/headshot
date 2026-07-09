//! Shared between the headshot client and server.
//!
//! Planned contents (see `doc/`):
//! - [`model`] — VGGT-Omega 1B architecture constants (doc/01).
//! - `protocol` — client/server message schema (doc/04 §3), added in M3.
//! - `pose` — pose_enc → extrinsics/intrinsics, unprojection (doc/01 §5), added in M2.

pub mod model;
