//! Session server (doc/04): RGB frames in, pose encodings and depth/conf
//! maps out, over the length-prefixed protocol of
//! [`headshot_shared::protocol`]. Stateless between sessions apart from
//! the loaded weights.
//!
//! Blocking I/O, one thread per connection; inference sessions serialize
//! on a compute lock (the GPU is saturated by one session anyway, doc/03
//! §7). A per-connection reader thread keeps parsing while inference runs
//! so `Cancel` can interrupt mid-trunk via the engine's cancel flag.

use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Instant;

use anyhow::Result;
use half::f16;
use tracing::{info, warn};

use crate::engine::tensor::Dtype;
use crate::engine::{Cancelled, GpuContext};
use crate::model::{camera_head::CameraHead, dense_head::DenseHead, dino::Dino, trunk::Trunk};
use crate::weights::Weights;
use headshot_shared::protocol::{ErrorCode, Message, Stage};

/// Everything loaded once and shared across sessions.
pub struct Engine {
    pub ctx: GpuContext,
    dino: Dino,
    trunk: Trunk,
    camera: CameraHead,
    dense: DenseHead,
    model_hash: String,
    pub frame_cap: u32,
    /// Serializes inference across connections.
    compute_lock: Mutex<()>,
}

impl Engine {
    pub fn load(weights_dir: &Path, precision: Dtype, frame_cap: u32) -> Result<Self> {
        let ctx = GpuContext::new()?;
        let weights = Weights::open(weights_dir)?;
        Ok(Self {
            dino: Dino::load(&ctx, &weights, precision)?,
            trunk: Trunk::load(&ctx, &weights, precision)?,
            camera: CameraHead::load(&ctx, &weights)?,
            dense: DenseHead::load(&ctx, &weights)?,
            model_hash: weights.manifest.model_sha256.clone(),
            frame_cap,
            compute_lock: Mutex::new(()),
            ctx,
        })
    }
}

/// Accept loop; blocks forever. Bind to an ephemeral port and use
/// [`TcpListener::local_addr`] for in-process tests.
pub fn serve(listener: TcpListener, engine: Arc<Engine>) -> Result<()> {
    info!("listening on {}", listener.local_addr()?);
    for stream in listener.incoming() {
        let stream = stream?;
        let engine = engine.clone();
        std::thread::spawn(move || {
            let peer = stream.peer_addr().map(|a| a.to_string()).unwrap_or_default();
            if let Err(e) = handle_connection(stream, engine) {
                warn!("connection {peer}: {e:#}");
            }
        });
    }
    Ok(())
}

/// One connection: a sequence of sessions.
pub fn handle_connection(stream: TcpStream, engine: Arc<Engine>) -> Result<()> {
    let mut reader = stream.try_clone()?;
    let writer = stream;
    let (tx, rx) = mpsc::channel::<Message>();

    // The reader thread parses everything; Cancel additionally flips the
    // engine's cancel flag immediately (the session thread may be deep in
    // the trunk). cancel_session tracks which session the flag belongs to.
    let cancel_session = Arc::new(AtomicU64::new(0));
    let cancel_session2 = cancel_session.clone();
    let reader_engine = engine.clone();
    std::thread::spawn(move || {
        // EOF or a protocol error ends the connection
        while let Ok(Some(msg)) = Message::read_from(&mut reader) {
            if let Message::Cancel { session_id } = &msg
                && cancel_session2.load(Ordering::SeqCst) == *session_id
            {
                reader_engine.ctx.request_cancel();
            }
            if tx.send(msg).is_err() {
                break;
            }
        }
    });

    let mut writer = writer;
    loop {
        let Ok(msg) = rx.recv() else { return Ok(()) }; // connection closed
        match msg {
            Message::OpenSession { session_id, n_frames, width, height, .. } => {
                cancel_session.store(session_id, Ordering::SeqCst);
                run_session(
                    &engine,
                    session_id,
                    n_frames,
                    width,
                    height,
                    &rx,
                    &mut writer,
                )?;
                cancel_session.store(0, Ordering::SeqCst);
            }
            Message::Cancel { .. } => {} // stale cancel between sessions
            other => {
                send_error(
                    &mut writer,
                    ErrorCode::Protocol,
                    &format!("expected OpenSession, got {other:?}"),
                )?;
            }
        }
    }
}

fn send_error(w: &mut impl Write, code: ErrorCode, message: &str) -> Result<()> {
    warn!("session error ({code:?}): {message}");
    Message::Error { code, message: message.into() }.write_to(w)?;
    Ok(())
}

/// One session: frame collection → validation → inference → streamed
/// results. Errors that end the *session* send `Error` and return Ok; only
/// I/O failures propagate (ending the connection).
fn run_session(
    engine: &Engine,
    session_id: u64,
    n_frames: u32,
    width: u32,
    height: u32,
    rx: &mpsc::Receiver<Message>,
    w: &mut TcpStream,
) -> Result<()> {
    // ---- validation (doc/04 §6) ----
    if width == 0 || height == 0 || !width.is_multiple_of(16) || !height.is_multiple_of(16) {
        return send_error(w, ErrorCode::Validation, "dims must be nonzero multiples of 16");
    }
    if n_frames == 0 || n_frames > engine.frame_cap {
        return send_error(
            w,
            ErrorCode::Validation,
            &format!("n_frames must be in 1..={}", engine.frame_cap),
        );
    }
    let frame_bytes = width as usize * height as usize * 3;

    // ---- frame collection (idempotent re-upload; doc/04 §6) ----
    let mut frames: Vec<Option<Vec<u8>>> = vec![None; n_frames as usize];
    loop {
        let Ok(msg) = rx.recv() else { return Ok(()) };
        match msg {
            Message::Frame { session_id: sid, frame_idx, rgb8 } if sid == session_id => {
                if frame_idx >= n_frames {
                    return send_error(w, ErrorCode::Validation, "frame_idx out of range");
                }
                if rgb8.len() != frame_bytes {
                    return send_error(w, ErrorCode::Validation, "frame size mismatch");
                }
                frames[frame_idx as usize] = Some(rgb8);
                Message::FrameAck { frame_idx, s1_done: false }.write_to(w)?;
            }
            Message::Reconstruct { session_id: sid } if sid == session_id => break,
            Message::Cancel { session_id: sid } if sid == session_id => {
                return send_error(w, ErrorCode::Cancelled, "cancelled before reconstruct");
            }
            other => {
                return send_error(
                    w,
                    ErrorCode::Protocol,
                    &format!("unexpected message during upload: {other:?}"),
                );
            }
        }
    }
    let missing = frames.iter().filter(|f| f.is_none()).count();
    if missing > 0 {
        return send_error(w, ErrorCode::Validation, &format!("{missing} frames never uploaded"));
    }

    // ---- inference (serialized across connections) ----
    let _compute = engine.compute_lock.lock().unwrap();
    engine.ctx.reset_cancel();
    let ctx = &engine.ctx;
    let (n, wpx, hpx) = (n_frames as usize, width as usize, height as usize);
    let (h_p, w_p) = (hpx / 16, wpx / 16);
    let px = wpx * hpx;

    // RGB8 HWC → f32 CHW planes
    let mut images = Vec::with_capacity(n * 3 * px);
    for frame in &frames {
        let rgb = frame.as_ref().unwrap();
        for c in 0..3 {
            images.extend(rgb.chunks_exact(3).map(|p| p[c] as f32 / 255.0));
        }
    }
    drop(frames);
    let images = ctx.tensor_from_slice(&[n, 3, hpx, wpx], &images);

    let peak_mem = spawn_peak_sampler();
    let t0 = Instant::now();
    let result: Result<()> = (|| {
        let tokens = self_dino(engine, ctx, &images, n, w)?;
        let s1 = t0.elapsed();

        let mut progress_w = w.try_clone()?;
        let mut tap = |name: &str, _t: &crate::engine::tensor::GpuTensor| {
            if let Some(k) = name.strip_prefix("trunk.inter.")
                && let Ok(done) = k.parse::<u32>()
            {
                let _ = Message::Progress { stage: Stage::S2Trunk, done: done + 1, total: 24 }
                    .write_to(&mut progress_w);
            }
        };
        let caches = engine.trunk.forward(ctx, &tokens, n, h_p, w_p, Some(&mut tap))?;
        drop(tokens);
        let s2 = t0.elapsed();

        // Cameras arrive in one message right after the trunk (doc/04 §2)
        let pose_enc = engine.camera.forward(ctx, &caches[3], n, None);
        Message::Cameras { pose_enc }.write_to(w)?;
        let s3 = t0.elapsed();

        let total_chunks =
            n.div_ceil(headshot_shared::model::DEPTH_CHUNK_FRAMES) as u32;
        let mut chunks_done = 0u32;
        let mut chunk_w = w.try_clone()?;
        let mut on_chunk = |first: usize, nc: usize, depth: &[f32], conf: &[f32]| {
            let to_f16 = |v: &[f32]| v.iter().map(|x| f16::from_f32(*x).to_bits()).collect();
            let _ = Message::DepthChunk {
                first_frame_idx: first as u32,
                n_frames: nc as u32,
                px_per_frame: px as u32,
                depth: to_f16(depth),
                conf: to_f16(conf),
            }
            .write_to(&mut chunk_w);
            chunks_done += 1;
            let _ = Message::Progress {
                stage: Stage::S4Depth,
                done: chunks_done,
                total: total_chunks,
            }
            .write_to(&mut chunk_w);
        };
        engine.dense.forward(ctx, &caches, n, h_p, w_p, None, Some(&mut on_chunk))?;
        ctx.sync();
        let s4 = t0.elapsed();

        Message::Done {
            s1_secs: s1.as_secs_f32(),
            s2_secs: (s2 - s1).as_secs_f32(),
            s3_secs: (s3 - s2).as_secs_f32(),
            s4_secs: (s4 - s3).as_secs_f32(),
            peak_mem_bytes: peak_mem.load(Ordering::Relaxed),
            model_hash: engine.model_hash.clone(),
        }
        .write_to(w)?;
        Ok(())
    })();

    match result {
        Ok(()) => {
            info!("session {session_id}: done ({n} frames, {:.1?})", t0.elapsed());
            Ok(())
        }
        Err(e) if e.is::<Cancelled>() => {
            // tear down: buffers drop with the closure scope; flag reset
            // happens at the next session start
            ctx.sync();
            send_error(w, ErrorCode::Cancelled, "session cancelled")
        }
        Err(e) => {
            ctx.sync();
            send_error(w, ErrorCode::Internal, &format!("{e:#}"))
        }
    }
}

/// DINO with an S1 progress message when done (per-frame overlap with the
/// upload is a later optimization; doc/04 §2 notes it is not load-bearing).
fn self_dino(
    engine: &Engine,
    ctx: &GpuContext,
    images: &crate::engine::tensor::GpuTensor,
    n: usize,
    w: &mut TcpStream,
) -> Result<crate::engine::tensor::GpuTensor> {
    let tokens = engine.dino.forward(ctx, images, None)?;
    Message::Progress { stage: Stage::S1Dino, done: n as u32, total: n as u32 }.write_to(w)?;
    Ok(tokens)
}

/// amdgpu GTT+VRAM peak sampler (bytes); returns 0s on non-amdgpu systems.
fn spawn_peak_sampler() -> Arc<AtomicU64> {
    let peak = Arc::new(AtomicU64::new(0));
    let peak2 = peak.clone();
    std::thread::spawn(move || {
        let files: Vec<_> = std::fs::read_dir("/sys/class/drm")
            .into_iter()
            .flatten()
            .flatten()
            .flat_map(|card| {
                ["mem_info_gtt_used", "mem_info_vram_used"]
                    .map(|n| card.path().join("device").join(n))
            })
            .filter(|p| p.exists())
            .collect();
        if files.is_empty() {
            return;
        }
        for _ in 0..36000 {
            let total: u64 = files
                .iter()
                .filter_map(|p| std::fs::read_to_string(p).ok())
                .filter_map(|s| s.trim().parse::<u64>().ok())
                .sum();
            peak2.fetch_max(total, Ordering::Relaxed);
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    });
    peak
}
