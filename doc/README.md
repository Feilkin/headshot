# VGGT-Omega Reconstruction App — Implementation Plan

Research project: combine multiple videos and photos of a scene into a single
registered 3D point cloud (mesh reconstruction is a follow-up project), using
the VGGT-Omega 1B feed-forward reconstruction model with inference handrolled
in Rust + wgpu/WGSL. bevy client, headless inference server.

These documents are self-contained: together with the paper and the parity
harness they specify the model, the kernels, the pipeline, and the protocol
precisely enough to implement from scratch. Where behavior must match the
reference PyTorch implementation bit-for-bit-ish, the exact constants and
conventions are written out here.

## Documents

| File | Contents |
|---|---|
| [01-model-spec.md](01-model-spec.md) | Complete inference-time spec of VGGT-Omega 1B: tokenization, aggregator, heads, all hyperparameters, coordinate/output conventions |
| [02-weights-and-parity.md](02-weights-and-parity.md) | Checkpoint conversion (`.pt` → safetensors), the K-bias-mask trap, f16 outlier scan, and the activation-parity test harness |
| [03-server-inference.md](03-server-inference.md) | WGSL kernel inventory, precision plan (f16 storage / f32 accumulate, WMMA), execution schedule, memory & compute budgets for Strix Halo |
| [04-pipeline-protocol.md](04-pipeline-protocol.md) | End-to-end stage graph: what streams, what blocks, client/server message schema |
| [05-client.md](05-client.md) | Client-side preprocessing: decode, tonemapping (D-Log, RAW), keyframe selection, resize/crop math (must match reference exactly), SRT telemetry parsing, bevy visualization |
| [06-scale-and-output.md](06-scale-and-output.md) | Unprojection, confidence/edge filtering, GPS Sim(3) metric-scale alignment, output formats |
| [07-capture-ui.md](07-capture-ui.md) | Capture UI architecture (bevy feathers): screens, worker protocol, theming targets, follow-up list |

## Reference materials

- Paper: *VGGT-Omega* (arXiv 2605.15195). LaTeX source in the arXiv source
  bundle (`arXiv-2605.15195v1/`); project page `vggt-omega.github.io`.
- Reference implementation: `github.com/facebookresearch/vggt-omega`
  (a copy exists in `arXiv-2605.15195v1/code/vggt-omega/`). Inference-only
  PyTorch code; ~2.5k lines. **See licensing note below before reading it
  while writing Rust code.**
- Checkpoint: `huggingface.co/facebook/VGGT-Omega`, file
  `vggt_omega_1b_512.pt` (gated; automated access approval). Only the 1B
  variant exists publicly. There is also a 256-res text-alignment variant we
  do not need.

## Licensing constraints (read first)

Meta's code **and weights** are under the **FAIR Noncommercial Research
License** (noncommercial use only, including outputs; derivatives inherit the
license). Our source code is planned as MIT + Apache-2.0, which is viable
under the llama.cpp precedent (independent MIT code, users fetch weights from
Meta themselves) **only if we keep the implementation clean**:

1. **Do not copy or line-by-line translate the FAIR-licensed Python into this
   repo.** Implement from these spec documents and the paper. Consulting the
   Python to resolve an ambiguity is acceptable; transliterating it is not.
2. **Never commit or distribute weights in any form**, including converted
   safetensors — a format conversion is still the weights. The repo ships a
   standalone conversion script; every user downloads the checkpoint from
   Hugging Face (accepting the FAIR license) and converts locally.
3. Do not bundle Meta's example videos, figures, or README text.
4. Our own *use* of the model and its outputs remains bound by the FAIR terms
   (noncommercial research) regardless of our code license. Published
   reconstructions should be labeled accordingly.

Running the reference implementation locally (for the parity harness in
02) is fine; its code just must not enter this repo.

## Architecture at a glance

```
┌────────────────────── client (bevy) ──────────────────────┐
│ video decode → tonemap → keyframe select → crop/resize    │
│ SRT/GPS parse                    point cloud viz, filters │
└──────────────┬────────────────────────────▲───────────────┘
        frames (RGB8, uniform size)   cameras (one msg), then
               ▼                            │ depth+conf chunks (streamed)
┌────────────────────── server (wgpu) ──────────────────────┐
│ DINOv3 patch embed (per frame, overlapped with upload)    │
│ aggregator trunk (global barrier — needs ALL frames)      │
│ camera head → dense depth head (chunked, streamed)        │
└───────────────────────────────────────────────────────────┘
```

Key structural fact driving the whole design: the aggregator mixes all frames
through 19 global-attention layers, so **no reconstruction output exists until
every frame has been uploaded and the trunk has run**. Only the DINO stage
(per-frame) can overlap the upload, and only the heads stream results.
Adding a frame later means rerunning the trunk from scratch.

## Target hardware

Single Strix Halo board (Radeon 8060S iGPU, RDNA 3.5, 40 CU), up to 116 GB
unified LPDDR5X (~256 GB/s). Memory is effectively unconstrained for this
model (weights ≈ 2.3 GB f16; activations linear in frame count); compute is
the budget and is dominated by global attention, **quadratic in frame count**.
f16/f32 cooperative-matrix (WMMA) via wgpu (available since ~wgpu 0.29) is
the intended GEMM path; it is known to work on this exact hardware from a
prior in-house project.

Rough wall-clock at ~15 TFLOPS sustained f16 (~1040 tokens/frame at 512-area
resolution): 100 frames ≈ 1–2 min, 200 ≈ 4–6 min, 500 ≈ 25–45 min. Keyframe
selection is worth more than kernel tuning.

## Milestones

Dependency graph — note M4a is client-only and can run in parallel with
M1–M3; everything else is sequential:

```
M0 ──► M1 ──► M2 ──► M3 ──► M4b ──► M5+
 └────────────────────────► M4a ─────┘   (M4a independent after M0 repo setup)
```

Every milestone is done only when its listed tests pass in CI (except the
parity tests, which need the gated checkpoint and run as a local/self-hosted
suite — mark them `#[ignore]`-style, wired to a `just parity` target).
Parity fixtures (converted weights, reference activation dumps) are
**generated locally by each developer** and never committed (weights-derived;
see licensing above and 02 §5).

### M0 — Weights + parity harness (02)
Deliverables: `tools/convert_weights.py`, `tools/dump_reference.py`, Rust
safetensors loader, comparison-test scaffolding.
**Done when:**
- Conversion of the real checkpoint succeeds with zero unexpected / missing /
  mis-shaped keys against the tree in 02 §3 (hard assertion, not a warning),
  the K-bias-mask rule (02 §2) validates on every attention block, and the
  f16 outlier report is emitted.
- `dump_reference.py` produces the two fixture dumps (small 4×128×96 and
  realistic 8×624×416), in both bf16-autocast and f32-forced variants,
  including the exact preprocessed input tensors.
- Rust loader test: loads the converted file, verifies every tensor's
  shape/dtype against the manifest, and round-trips a checksum.

### M1 — Trunk (01 §§2–3, 03)
Deliverables: DINO + aggregator in the WGSL engine; non-causal flash
attention, axial RoPE, QK-norm kernels; all-f32 debug mode.
**Done when:** per-layer parity vs. the M0 fixtures passes the gates of
02 §5 on both fixture scenes: all-f32 mode within tight gates through
layer 23 and the four cached tensors; f16 mode within the cosine gates with
no per-layer cliff. Kernel-level unit tests (RoPE table values, QK-norm,
attention vs. a naive f32 reference implementation on random small inputs)
run in CI without the checkpoint.

### M2 — Heads + first point cloud (01 §§4–5, 06 §3)
Deliverables: camera head, dense head, unprojection + filtering, minimal CLI
(`frames dir in → PLY out`) using plain JPEG/PNG input and the resize math of
05 §3 (no tonemapping yet).
**Done when:**
- `pose_enc` and `depth`/`conf` parity gates (02 §5) pass on the fixtures.
- Golden-scene regression: the CLI on a checked-in set of ~20 phone photos
  reproduces stored camera poses and depth stats within tolerance across
  engine changes (tolerances from the f16 parity experience).
- The exported PLY of the golden scene opens in an external viewer
  (CloudCompare/MeshLab) and passes documented visual inspection —
  record screenshots in the repo.
- Unit tests: quat→matrix (incl. non-unit q), pose-encoding round-trip,
  unprojection round-trip (project(unproject(d)) = pixel), depth-edge filter
  on synthetic step edges.

### M3 — Client/server split (04, 05 §4)
Deliverables: protocol, server binary, bevy client with progressive cloud.
**Done when:**
- Loopback integration test: a full session over the real protocol on the
  golden scene produces results equal (within f32 serialization tolerance)
  to the M2 CLI path.
- Streaming-order test: `CamerasMsg` arrives before the first
  `DepthChunkMsg`; chunks arrive incrementally (assert on message timeline,
  not just final state); frame re-upload is idempotent.
- Cancellation test: `Cancel` mid-trunk tears the session down and the next
  session on the same server process succeeds.
- bevy viewer renders the golden scene progressively with working
  confidence-percentile and per-source filters (manual checklist, recorded).

### M4a — Capture preprocessing (client-only; parallel to M1–M3) (05)
Deliverables: D-Log LUT application, RAW development path, keyframe
selection, SRT parser.
**Done when:** unit tests pass for: LUT application against known
input→output triples; resize/crop math (property tests: dims divisible by
16, patch count 1024 ± rounding, aspect clamped to [0.5, 2.0], plus the
worked 3:2 → 624×416 example); SRT parsing on real captured samples checked
into the repo (telemetry only — verify the files carry no imagery/PII);
keyframe selection is deterministic and respects the budget on a fixture
video.

### M4b — Metric scale (needs M2 outputs) (06)
Deliverables: ENU conversion, RANSAC + Umeyama Sim(3), pose/point transform
application, geo-referenced export.
**Done when:** synthetic recovery test passes (random Sim(3) + GPS-like
noise → scale recovered within noise bounds, 06 §5); camera-transform
round-trip test passes (transformed camera reprojects transformed points to
identical pixels); one field validation against a measured ground-truth
distance is documented in the repo.

### M5+ — Later (not specced here)
Mesh extraction (TSDF fusion + marching cubes over the session bundle),
register-only draft mode evaluation (03 §6), COLMAP-refinement export.
