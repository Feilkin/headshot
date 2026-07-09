//! M3 gates (doc/README): loopback session over the real protocol vs the
//! direct engine path; streaming order; idempotent re-upload; cancellation.
//! Parity-gated (GPU + local fixtures): run via `just parity`.

use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

use half::f16;
use headshot_server::engine::tensor::Dtype;
use headshot_server::parity::{Dump, fixtures_dir};
use headshot_server::server::{Engine, handle_connection};
use headshot_shared::protocol::{ErrorCode, Message, Stage};

/// In-process server on an ephemeral port; returns the connect address.
fn start_server(engine: Arc<Engine>) -> std::net::SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let engine = engine.clone();
            std::thread::spawn(move || {
                let _ = handle_connection(stream, engine);
            });
        }
    });
    addr
}

/// Fixture scene as RGB8 frames (the wire format).
fn fixture_frames(scene: &str) -> (usize, u32, u32, Vec<Vec<u8>>) {
    let dump = Dump::open(scene, "f32").expect("dump");
    let (shape, images) = dump.tensor("images").unwrap();
    let [n, _, h, w] = shape[..] else { panic!() };
    let px = h * w;
    let frames = (0..n)
        .map(|i| {
            let planes = &images[i * 3 * px..][..3 * px];
            let mut rgb = Vec::with_capacity(3 * px);
            for p in 0..px {
                for c in 0..3 {
                    rgb.push((planes[c * px + p] * 255.0).round().clamp(0.0, 255.0) as u8);
                }
            }
            rgb
        })
        .collect();
    (n, w as u32, h as u32, frames)
}

/// Drive one full session; returns every server message in arrival order.
fn run_session(
    addr: std::net::SocketAddr,
    session_id: u64,
    width: u32,
    height: u32,
    frames: &[Vec<u8>],
    resend_frame0: bool,
) -> Vec<Message> {
    let mut s = TcpStream::connect(addr).unwrap();
    Message::OpenSession {
        session_id,
        n_frames: frames.len() as u32,
        width,
        height,
        draft: false,
        model: "vggt-omega-1b-512".into(),
    }
    .write_to(&mut s)
    .unwrap();
    if resend_frame0 {
        // garbage first — the later correct upload must win (idempotent
        // re-upload, doc/04 §6)
        Message::Frame {
            session_id,
            frame_idx: 0,
            rgb8: vec![0u8; frames[0].len()],
        }
        .write_to(&mut s)
        .unwrap();
    }
    for (i, rgb8) in frames.iter().enumerate() {
        Message::Frame { session_id, frame_idx: i as u32, rgb8: rgb8.clone() }
            .write_to(&mut s)
            .unwrap();
    }
    Message::Reconstruct { session_id }.write_to(&mut s).unwrap();

    let mut messages = Vec::new();
    loop {
        let msg = Message::read_from(&mut s).unwrap().expect("server closed early");
        let done = matches!(msg, Message::Done { .. } | Message::Error { .. });
        messages.push(msg);
        if done {
            return messages;
        }
    }
}

#[test]
#[ignore = "needs local fixtures + GPU (just parity)"]
fn parity_loopback_session_matches_direct() {
    let engine = Arc::new(Engine::load(&fixtures_dir(), Dtype::F16, 512).unwrap());
    let addr = start_server(engine.clone());
    let (n, width, height, frames) = fixture_frames("small");
    let px = (width * height) as usize;

    let messages = run_session(addr, 1, width, height, &frames, true);

    // ---- streaming order (doc/04 §2) ----
    let acks = messages.iter().filter(|m| matches!(m, Message::FrameAck { .. })).count();
    assert_eq!(acks, n + 1, "one ack per upload incl. the re-upload");
    let idx_of = |pred: &dyn Fn(&Message) -> bool| messages.iter().position(pred);
    let cameras_at = idx_of(&|m| matches!(m, Message::Cameras { .. })).expect("CamerasMsg");
    let first_chunk_at =
        idx_of(&|m| matches!(m, Message::DepthChunk { .. })).expect("DepthChunkMsg");
    assert!(cameras_at < first_chunk_at, "cameras before first depth chunk");
    let s2_at = idx_of(&|m| {
        matches!(m, Message::Progress { stage: Stage::S2Trunk, .. })
    })
    .expect("S2 progress");
    assert!(s2_at < cameras_at, "trunk progress precedes cameras");
    // incremental chunks: assert on the message timeline, not final state
    let chunk_firsts: Vec<u32> = messages
        .iter()
        .filter_map(|m| match m {
            Message::DepthChunk { first_frame_idx, .. } => Some(*first_frame_idx),
            _ => None,
        })
        .collect();
    let expected_chunks = n.div_ceil(headshot_shared::model::DEPTH_CHUNK_FRAMES);
    assert_eq!(chunk_firsts.len(), expected_chunks);
    assert!(chunk_firsts.windows(2).all(|w| w[0] < w[1]), "chunks in frame order");
    assert!(matches!(messages.last(), Some(Message::Done { .. })));

    // ---- results == direct engine path (same rgb8-derived input) ----
    let mut images = Vec::with_capacity(n * 3 * px);
    for rgb in &frames {
        for c in 0..3 {
            images.extend(rgb.chunks_exact(3).map(|p| p[c] as f32 / 255.0));
        }
    }
    let ctx = &engine.ctx;
    let gi = ctx.tensor_from_slice(&[n, 3, height as usize, width as usize], &images);
    let (h_p, w_p) = (height as usize / 16, width as usize / 16);
    let weights = headshot_server::weights::Weights::open(&fixtures_dir()).unwrap();
    let dino = headshot_server::model::dino::Dino::load(ctx, &weights, Dtype::F16).unwrap();
    let trunk = headshot_server::model::trunk::Trunk::load(ctx, &weights, Dtype::F16).unwrap();
    let camera = headshot_server::model::camera_head::CameraHead::load(ctx, &weights).unwrap();
    let dense = headshot_server::model::dense_head::DenseHead::load(ctx, &weights).unwrap();
    let tokens = dino.forward(ctx, &gi, None).unwrap();
    let caches = trunk.forward(ctx, &tokens, n, h_p, w_p, None).unwrap();
    let pose_direct = camera.forward(ctx, &caches[3], n, None);
    let out = dense.forward(ctx, &caches, n, h_p, w_p, None, None).unwrap();

    let pose_session = messages
        .iter()
        .find_map(|m| match m {
            Message::Cameras { pose_enc } => Some(pose_enc.clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(pose_session.len(), pose_direct.len());
    for (a, b) in pose_session.iter().zip(&pose_direct) {
        assert!((a - b).abs() < 1e-4, "pose {a} vs {b}");
    }

    let mut depth_session = vec![0.0f32; n * px];
    for m in &messages {
        if let Message::DepthChunk { first_frame_idx, depth, .. } = m {
            let start = *first_frame_idx as usize * px;
            for (i, bits) in depth.iter().enumerate() {
                depth_session[start + i] = f16::from_bits(*bits).to_f32();
            }
        }
    }
    // session depth is f16-quantized on the wire (doc/04 §3)
    for (i, (a, b)) in depth_session.iter().zip(&out.depth).enumerate() {
        let expected = f16::from_f32(*b).to_f32();
        assert!(
            (a - expected).abs() <= 2.0 * (expected.abs() * 1e-3).max(1e-4),
            "depth[{i}]: session {a} vs direct {expected}"
        );
    }
}

#[test]
#[ignore = "needs local fixtures + GPU (just parity)"]
fn parity_cancel_mid_trunk_then_next_session_succeeds() {
    let engine = Arc::new(Engine::load(&fixtures_dir(), Dtype::F16, 512).unwrap());
    let addr = start_server(engine.clone());
    let (_, width, height, frames) = fixture_frames("realistic");

    // session 1: cancel as soon as the trunk reports progress
    let mut s = TcpStream::connect(addr).unwrap();
    Message::OpenSession {
        session_id: 10,
        n_frames: frames.len() as u32,
        width,
        height,
        draft: false,
        model: "vggt-omega-1b-512".into(),
    }
    .write_to(&mut s)
    .unwrap();
    for (i, rgb8) in frames.iter().enumerate() {
        Message::Frame { session_id: 10, frame_idx: i as u32, rgb8: rgb8.clone() }
            .write_to(&mut s)
            .unwrap();
    }
    Message::Reconstruct { session_id: 10 }.write_to(&mut s).unwrap();
    let outcome = loop {
        match Message::read_from(&mut s).unwrap().expect("server closed") {
            Message::Progress { stage: Stage::S2Trunk, done: 1, .. } => {
                Message::Cancel { session_id: 10 }.write_to(&mut s).unwrap();
            }
            Message::Error { code, .. } => break code,
            Message::Done { .. } => panic!("session completed despite cancel"),
            _ => {}
        }
    };
    assert_eq!(outcome, ErrorCode::Cancelled);

    // session 2 on the same server process: must succeed
    let (_, w2, h2, frames2) = fixture_frames("small");
    let messages = run_session(addr, 11, w2, h2, &frames2, false);
    assert!(
        matches!(messages.last(), Some(Message::Done { .. })),
        "next session failed: {:?}",
        messages.last()
    );
}
