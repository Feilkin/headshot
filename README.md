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

## Building & testing

```sh
cargo build --workspace
cargo nextest run --workspace   # CI suite (no checkpoint needed)
just parity                     # parity suite; needs locally generated fixtures (doc/02 §5)
```

Requires Rust 1.95+, [cargo-nextest](https://nexte.st), and optionally
[just](https://github.com/casey/just).

## License

MIT OR Apache-2.0, at your option (source code only — see the licensing note
in doc/README.md regarding model weights and outputs).
