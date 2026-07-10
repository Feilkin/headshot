//! Client side of a reconstruction session: preprocess frames (doc/05 §3),
//! drive the protocol, unproject depth chunks into world-space points
//! (doc/04 §4). Runs on a background thread; talks to the bevy app through
//! a crossbeam channel.

use std::net::TcpStream;
use std::path::PathBuf;

use anyhow::{Context, Result};
use crossbeam_channel::Sender;
use half::f16;
use headshot_capture::keyframe::SelectParams;
use headshot_capture::video::FfmpegCli;
use headshot_capture::{CaptureConfig, prepare_session};
use headshot_shared::filter::depth_edge_mask;
use headshot_shared::pose::Camera;
use headshot_shared::protocol::{Message, Stage};

/// One depth chunk, unprojected and edge-filtered. Confidence and frame id
/// ride along so the viewer can re-filter interactively without
/// re-inference (doc/05 §4).
pub struct ChunkPoints {
    pub positions: Vec<[f32; 3]>,
    pub colors: Vec<[f32; 4]>,
    pub conf: Vec<f32>,
    pub frame: Vec<u16>,
}

pub enum ViewerEvent {
    Status(String),
    Cameras(Vec<Camera>),
    Chunk(ChunkPoints),
    Done,
    Failed(String),
}

pub struct SessionConfig {
    pub server: String,
    /// Mixed media directory (videos + photos + sidecar .srt) or a single
    /// file (doc/05 §1).
    pub media: PathBuf,
    /// Total keyframe budget across all sources (doc/05 §2).
    pub budget: usize,
    /// Official D-Log→Rec.709 `.cube`; video frames only.
    pub dlog_lut: Option<PathBuf>,
    /// Parametric D-Log fallback when no `.cube` is available.
    pub dlog: bool,
    /// Debug dump of the preprocessed keyframes + manifest.
    pub dump_keyframes: Option<PathBuf>,
    pub edge_threshold: f32,
}

pub fn run(config: SessionConfig, tx: Sender<ViewerEvent>) {
    if let Err(e) = run_inner(&config, &tx) {
        let _ = tx.send(ViewerEvent::Failed(format!("{e:#}")));
    }
}

fn run_inner(config: &SessionConfig, tx: &Sender<ViewerEvent>) -> Result<()> {
    // ---- capture preprocessing (doc/05 §§1–3) ----
    let prepared = prepare_session(
        &CaptureConfig {
            media: vec![config.media.clone()],
            budget: config.budget,
            dlog_lut: config.dlog_lut.clone(),
            dlog_parametric: config.dlog,
            params: SelectParams { budget: config.budget, ..Default::default() },
        },
        &FfmpegCli::default(),
        &mut |msg| {
            let _ = tx.send(ViewerEvent::Status(msg));
        },
    )
    .with_context(|| format!("preparing session from {}", config.media.display()))?;
    if let Some(dir) = &config.dump_keyframes {
        prepared.dump(dir)?;
        tx.send(ViewerEvent::Status(format!("dumped keyframes to {}", dir.display())))?;
    }
    run_protocol(prepared, &config.server, config.edge_threshold, tx)
}

/// Drive a reconstruction session (doc/04) for an already-prepared frame
/// batch: upload, reconstruct, unproject streamed depth chunks. Shared by
/// the CLI path and the GUI worker.
pub fn run_protocol(
    prepared: headshot_capture::PreparedSession,
    server: &str,
    edge_threshold: f32,
    tx: &Sender<ViewerEvent>,
) -> Result<()> {
    let (tw, th) = (prepared.width, prepared.height);
    let (width, height) = (tw as usize, th as usize);
    let px = width * height;
    let rgb_frames = prepared.frames;
    let n = rgb_frames.len();

    // ---- session (doc/04) ----
    let session_id = std::process::id() as u64;
    let mut s =
        TcpStream::connect(server).with_context(|| format!("connecting to {server}"))?;
    Message::OpenSession {
        session_id,
        n_frames: n as u32,
        width: tw,
        height: th,
        draft: false,
        model: "vggt-omega-1b-512".into(),
    }
    .write_to(&mut s)?;
    for (i, rgb8) in rgb_frames.iter().enumerate() {
        Message::Frame { session_id, frame_idx: i as u32, rgb8: rgb8.clone() }.write_to(&mut s)?;
    }
    Message::Reconstruct { session_id }.write_to(&mut s)?;
    tx.send(ViewerEvent::Status(format!("uploaded {n} frames; reconstructing…")))?;

    let mut cameras: Vec<Camera> = Vec::new();
    loop {
        let msg = Message::read_from(&mut s)?.context("server closed mid-session")?;
        match msg {
            // acks stream back interleaved with our uploads; nothing to do
            Message::FrameAck { .. } => {}
            Message::Progress { stage, done, total } => {
                let label = match stage {
                    Stage::S1Dino => "DINO",
                    Stage::S2Trunk => "trunk",
                    Stage::S4Depth => "depth",
                };
                tx.send(ViewerEvent::Status(format!("{label}: {done}/{total}")))?;
            }
            Message::Cameras { pose_enc } => {
                cameras =
                    pose_enc.chunks(9).map(|e| Camera::from_pose_enc(e, tw, th)).collect();
                tx.send(ViewerEvent::Cameras(cameras.clone()))?;
            }
            Message::DepthChunk { first_frame_idx, n_frames, px_per_frame, depth, conf } => {
                anyhow::ensure!(px_per_frame as usize == px, "chunk pixel-count mismatch");
                let chunk = unproject_chunk(
                    &cameras,
                    &rgb_frames,
                    first_frame_idx as usize,
                    n_frames as usize,
                    width,
                    height,
                    &depth,
                    &conf,
                    edge_threshold,
                );
                tx.send(ViewerEvent::Chunk(chunk))?;
            }
            Message::Done { s1_secs, s2_secs, s3_secs, s4_secs, .. } => {
                tx.send(ViewerEvent::Status(format!(
                    "done (s1 {s1_secs:.1}s, s2 {s2_secs:.1}s, s3 {s3_secs:.2}s, s4 {s4_secs:.1}s)"
                )))?;
                tx.send(ViewerEvent::Done)?;
                return Ok(());
            }
            Message::Error { code, message } => {
                anyhow::bail!("server error ({code:?}): {message}");
            }
            other => anyhow::bail!("unexpected message: {other:?}"),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn unproject_chunk(
    cameras: &[Camera],
    rgb_frames: &[Vec<u8>],
    first: usize,
    n: usize,
    width: usize,
    height: usize,
    depth_bits: &[u16],
    conf_bits: &[u16],
    edge_threshold: f32,
) -> ChunkPoints {
    let px = width * height;
    let mut out = ChunkPoints {
        positions: Vec::new(),
        colors: Vec::new(),
        conf: Vec::new(),
        frame: Vec::new(),
    };
    for fi in 0..n {
        let frame = first + fi;
        let cam = &cameras[frame];
        let rgb = &rgb_frames[frame];
        let depth: Vec<f32> =
            depth_bits[fi * px..][..px].iter().map(|b| f16::from_bits(*b).to_f32()).collect();
        let keep = depth_edge_mask(&depth, height, width, edge_threshold);
        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;
                let d = depth[i];
                if !d.is_finite() || !keep[i] {
                    continue;
                }
                let c = f16::from_bits(conf_bits[fi * px + i]).to_f32();
                out.positions.push(cam.unproject(x as u32, y as u32, d));
                out.colors.push([
                    rgb[i * 3] as f32 / 255.0,
                    rgb[i * 3 + 1] as f32 / 255.0,
                    rgb[i * 3 + 2] as f32 / 255.0,
                    1.0,
                ]);
                out.conf.push(c);
                out.frame.push(frame as u16);
            }
        }
    }
    out
}
