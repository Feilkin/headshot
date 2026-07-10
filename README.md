# headshot

Research project: fuse multiple videos and photos of a scene into one
registered 3D point cloud using the VGGT-Omega 1B feed-forward reconstruction
model, with inference hand-rolled in Rust + wgpu/WESL. bevy client, headless
inference server.

**Start with [doc/README.md](doc/README.md)** — implementation plan,
milestones, and the licensing constraints (our code is MIT/Apache-2.0, but
Meta's model weights and their derivatives are FAIR Noncommercial and must
never enter this repo).

## Workspace

| crate | role |
|---|---|
| `headshot-shared` | protocol messages, model constants, pose/camera math |
| `headshot-server` | headless wgpu/WESL inference engine + session server (doc/03, 04) |
| `headshot-client` | bevy app: preprocessing, keyframe selection, viewer, export (doc/05, 06) |

`tools/` holds the standalone Python scripts for weight conversion and
reference activation dumps (doc/02).

## Status

M0–M3 complete: weight conversion + parity harness, the full WGSL engine
(f32 debug + f16/WMMA fast path, both parity-green against the reference),
camera + dense heads, and the client/server split. Remaining milestones:
M4a capture preprocessing (D-Log/RAW/keyframes/SRT), M4b GPS metric scale;
perf follow-ups tracked in-session (wave-level attention, buffer pool).

## Running

One-time: download `vggt_omega_1b_512.pt` from Hugging Face (FAIR license),
then `just tools-venv && just convert <ckpt>` (and `just dump <ckpt>` for
the parity suite).

```sh
# offline: photos in, PLY out
cargo run --release -p headshot-server --bin reconstruct -- <frames-dir> -o scene.ply

# server (Strix Halo box); 0.0.0.0 to accept LAN clients
cargo run --release -p headshot-server -- --listen 0.0.0.0:9276

# capture GUI (any machine; needs a display + ffmpeg on PATH)
cargo run --release -p headshot-client -- --server <box-ip>:9276

# CLI flow: reconstruct <media-dir> immediately, no interaction
cargo run --release -p headshot-client -- <media-dir> --server <box-ip>:9276
```

The GUI starts on a Setup screen: add media (type a path, drag & drop on
X11, or launch with `<media-dir> --review` to pre-fill), prune the
discovered file tree, scan, then edit keyframes in the Review screen —
scrub candidates, toggle inclusion, zoom the centered crop per frame, pick
the session aspect — before reconstructing with a live log pane.

`<media-dir>` may mix DJI video (H.264/H.265 + sidecar or embedded `.srt`
telemetry), RAW/JPEG/HEIC photos — keyframes are selected under `--budget`
(default 200; server cost is quadratic). D-Log footage: pass the official
LUT with `--dlog-lut <file.cube>`, or `--dlog` for the parametric
approximation. `--dump-keyframes <dir>` writes the preprocessed frames +
manifest for inspection.

Viewer controls: drag = orbit, wheel = zoom, `[`/`]` = confidence
percentile, `G` = frame groups, `F` = frusta. `--headless` runs the session
without a window. Manual viewer checklist: doc/m3-viewer-checklist.md.
Capture-sample suite: `HEADSHOT_SAMPLES_DIR=<dir> just samples` (local DJI
clips + RAW; never committed).

## Building & testing

```sh
cargo build --workspace
cargo nextest run --workspace   # CI suite (no checkpoint needed)
just parity                     # parity + integration suite; needs local fixtures (doc/02 §5)
```

Requires Rust 1.95+, [cargo-nextest](https://nexte.st), and optionally
[just](https://github.com/casey/just).

## License

MIT OR Apache-2.0, at your option (source code only — see the licensing note
in doc/README.md regarding model weights and outputs).
