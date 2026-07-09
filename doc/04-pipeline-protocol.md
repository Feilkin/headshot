# 04 — End-to-End Pipeline and Client/Server Protocol

## 1. Division of labor

**Client** (bevy app, runs anywhere): media decode, tonemapping, keyframe
selection, crop/resize to model resolution, SRT/GPS parsing, upload;
receives cameras + depth maps; performs unprojection, filtering, Sim(3)
scale alignment, visualization, export.

**Server** (Strix Halo box): pure model inference — RGB frames in,
`pose_enc` + depth/conf maps out. Stateless between sessions apart from
loaded weights.

Rationale: the server stays a deterministic tensor function (easy to test
against the parity harness); everything subjective/interactive (thresholds,
color, selection) lives client-side where iteration is free. Unprojection is
client-side so confidence/edge filtering is interactive without re-inference.

## 2. Stage graph — what waits, what streams

```
stage                         can start when            emits
──────────────────────────────────────────────────────────────────────────────
U   upload frames             session open              per-frame ack
S1  DINO patch embed          per frame, on arrival     progress (frames done)
S2  trunk (48 blocks)         ALL frames uploaded       progress (block k/24)
S3  camera head               trunk done                CamerasMsg (all N at once)
S4  dense head (chunks of 8)  trunk done (after S3)     DepthChunkMsg per chunk
E   end                       all chunks sent           DoneMsg (stats)
```

Hard rules the UI must reflect:
- **Nothing about the reconstruction exists until S2 completes.** Global
  attention couples every frame; there are no partial poses or partial depth
  during the trunk. Progress there is honest but content-free (block index).
- S1 is the only upload-overlapped compute (~1/3 of GEMM FLOPs, but a small
  share of total time at high frame counts where quadratic attention
  dominates). Worth doing but not architecturally load-bearing.
- **Adding/removing a frame invalidates S2–S4 entirely** (full recompute).
  The client should treat frame selection as committed at "Reconstruct".
- Cameras arrive in one message right after the trunk — the client can draw
  all frusta immediately, seconds-to-minutes before depth finishes.
- Depth streams per 8-frame chunk; the client unprojects and grows the point
  cloud incrementally. Chunk order = frame order; no reordering needed.

## 3. Message schema (transport-agnostic; length-prefixed binary over one
   duplex stream — WebSocket or plain TCP both fine; little-endian)

Client → Server:

```
OpenSession   { session_id, n_frames, width, height,        // uniform, /16
                draft: bool,            // frame-subset preview handled client-side;
                                        // reserved for future server modes
                model: "vggt-omega-1b-512" }
Frame         { session_id, frame_idx: u32, rgb8: bytes }    // W·H·3, sRGB-ish [0,255]
Reconstruct   { session_id }            // = "all frames sent, go"
Cancel        { session_id }
```

Server → Client:

```
FrameAck      { frame_idx, s1_done: bool }
Progress      { stage: enum{S1,S2,S4}, done: u32, total: u32 }
CamerasMsg    { pose_enc: [N][9] f32 }                      // t, quat xyzw, fov_h, fov_w
DepthChunkMsg { first_frame_idx, n,                          // n ≤ 8
                depth:  [n][H][W] f16,                       // z-depth
                conf:   [n][H][W] f16 }                      // ≥ 1
DoneMsg       { timings per stage, peak_mem, model_hash }
Error         { code, message }                              // e.g. size mismatch, OOM, cancelled
```

Sizing: RGB8 frame at 624×416 ≈ 780 KB (LZ4-frame the stream if the link is
slow; content is already noisy, don't bother with image codecs). Depth+conf
chunk ≈ 8 × 2 × 519 KB ≈ 8.3 MB. A 200-frame session moves ~150 MB up,
~210 MB down — trivial on LAN.

The client keeps the original full-res sources and the preprocessed frames;
the server never needs colors back (client colors points from its own copy of
the preprocessed frames — same pixel grid as the depth maps).

## 4. Client-side post-processing order (per DepthChunkMsg)

1. Decode depth/conf to f32.
2. Compute depth-edge mask (01 §5.3) per frame.
3. Unproject valid pixels with that frame's intrinsics/extrinsics (01 §5.2)
   → world-space points in frame-0 camera coordinates.
4. Color from the preprocessed RGB frame (same resolution — 1:1 pixel map).
5. Insert into the render structure tagged with {frame_idx, conf} so the
   confidence-percentile slider and per-source toggles re-filter without
   recompute. Percentile threshold needs the global conf distribution —
   maintain a running histogram (e.g. 4096 log-spaced bins), refine as chunks
   arrive.
6. After DoneMsg: optional Sim(3) GPS alignment (06), then export.

## 5. Draft-preview flow (client-driven)

"Preview" button = run a session with every k-th selected keyframe
(default k=4). Quadratic attention makes this ~16× faster. Preview and final
are separate sessions; the client keeps them in separate scene layers (their
coordinate frames differ — both are frame-0-relative, and frame 0 of the
preview subset should be chosen as the same physical frame as the final run's
frame 0 so the views roughly coincide; scale may still differ slightly).

## 6. Session lifecycle / errors

- Server enforces: uniform frame sizes, dims divisible by 16, frame-count cap
  (config; default 512), single active session (queue otherwise).
- On any GPU error / device loss: Error + session teardown; client offers
  retry with fewer frames.
- Idempotent frame upload (re-send frame_idx overwrites) so flaky links can
  retry without renegotiation.
