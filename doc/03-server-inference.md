# 03 — Server: WGSL Inference Engine

Target: Strix Halo (Radeon 8060S, 40 CU RDNA 3.5, up to 116 GB unified
LPDDR5X @ ~256 GB/s), wgpu with f16 (`shader-f16`) and f16/f32
cooperative-matrix / WMMA (available since ~wgpu 0.29; already used
successfully on this hardware). Base: the existing in-house WGSL inference
engine (LLM kernels, composable primitives).

## 1. Kernel inventory

### Reusable from the LLM engine (likely as-is or near)
- Tiled WMMA GEMM (f16 in, f32 accumulate) — the workhorse; everything below
  that says "linear" or "1×1 conv" is this kernel.
- LayerNorm (must support eps 1e-5, weight+bias, f32 math).
- GELU (exact tanh-approx choice: reference uses PyTorch default `nn.GELU()`
  = erf-based; if the engine has only tanh-GELU, parity will show ~1e-3
  noise — acceptable, but note it).
- Softmax/online-softmax pieces, residual add, elementwise scale.

### New / modified kernels
1. **Non-causal flash attention** (the big one). Bidirectional SDPA with
   online softmax over sequences up to ~1M tokens (N·T). If the existing
   prefill kernel is causal, the mask removal is trivial but check the tile
   scheduler doesn't assume triangular work. Head_dim 64 (trunk) and 128
   (camera head). Scale 1/√head_dim. No attention bias, no dropout.
2. **2D axial RoPE** (01 §2.1): halves-rotation, 17- or 5-token unrotated
   prefix, f32 tables. Distinct from LLM interleaved RoPE — new kernel, or
   fold into the attention kernel's q/k load path.
3. **Per-head QK-LayerNorm** (LayerNorm over head_dim=64 with weight+bias,
   applied to q and k before RoPE, trunk blocks only). Foldable into the
   attention prologue.
4. **LayerScale** — per-channel multiply; fold into the residual-add.
5. **im2col + GEMM 3×3 conv** (pad 1, stride 1 and one stride-2 case) — DPT
   head only, tiny resolutions (≤ H/4), low effort; a direct conv kernel is
   also fine.
6. **ConvTranspose with kernel==stride** (4×4s4, 2×2s2): non-overlapping ⇒
   implement as 1×1 GEMM producing k²·C_out channels + pixel-shuffle. No
   overlap-add machinery needed.
7. **Pixel shuffle** (×4) — reshape/permute kernel.
8. **Bilinear resize, align_corners=True** — DPT fusion cascade. Get the
   convention exactly right: `src = dst · (src_size−1)/(dst_size−1)`.
9. **Patch embed** — 16×16-stride-16 conv ≡ reshape + GEMM.
10. **UV positional embedding add** (01 §4.3) — precompute per session on
    CPU (f32), upload, elementwise add.
11. Small stuff: ImageNet normalize, `exp` / `1+exp` output activations,
    token gather/scatter for register attention (first 17 of every frame),
    concat for the 2048-ch cached tensors.

## 2. Precision plan

- Weights f16 (except the f32 list in 02 §4). Activations f16 in the trunk,
  **f32 accumulation in every GEMM and attention softmax/PV accumulate**.
- All LayerNorms, QK-norms, RoPE, softmax statistics: f32.
- From the cached 2048-ch tensors onward (camera head, dense head): **all
  f32**, matching the reference's autocast boundary. These stages are a few
  percent of total FLOPs; precision here directly hits output quality.
- Rationale: the reference trunk runs bf16. f16 has the same mantissa
  as bf16(+3 bits) but far smaller exponent range; f32 accumulation plus
  f32 norms removes the overflow-prone spots. Parity harness (02 §5)
  validates the result; an all-f32 debug mode isolates kernel bugs from
  precision drift.
- **Exception (measured 2026-07-09): DINO must not run f16.** Its residual
  stream carries per-frame "massive activations" of ~1.4e5 from block 0
  onward — beyond f16's 65504 max (bf16 survives on exponent range alone).
  The final DINO norm squashes them (output max ~2) and the aggregator
  trunk's activations stay ≤ ~165, so: DINO runs f32 (f32 GEMMs, ~0.8 s for
  8 frames — cheap, linear in N), the trunk runs f16/WMMA, one cast at the
  stage boundary. If a future variant needs f16 DINO, the fix is an f32
  residual stream with f16 GEMM in/outputs, not f16 storage throughout.

## 3. Execution schedule

Per session (N frames, per-frame tokens T = P+17, e.g. P=1014/T=1031 at
624×416):

```
S1  DINO (per frame, batchable, can run as frames arrive):
      24 blocks × (LN → qkv GEMM → RoPE+SDPA → proj → LN → MLP)
S2  assemble trunk tokens: prepend camera+register variants (frame0 vs rest)
S3  for k in 0..24:
      frame block k        # batch N seqs of length T
      inter-frame block k  # global: 1 seq of N·T   (k ∉ {2,6,9,14,20})
                           # register: 1 seq of N·17 (k ∈ {2,6,9,14,20})
      if k in {4,11,17,23}: write concat(frame_out, inter_out) → cache_k (f32)
S4  camera head: 4 blocks over (N·17, 2048) → pose_enc  (emit immediately)
S5  dense head: for each chunk of 8 frames: DPT pyramid → depth+conf (emit per chunk)
```

Barriers: S3 is a hard barrier per inter-frame block (global attention needs
every frame's tokens). S4 is tiny (milliseconds). S5 chunks are independent.

## 4. Memory budget (f16 activations, N = 200, T = 1031)

| item | size |
|---|---|
| weights | ~2.3 GB |
| live trunk tokens (double-buffered) | 2 × N·T·1024 × 2 B ≈ 0.85 GB |
| flash-attn workspace | O(N·T·head_dim) tiles, < 1 GB |
| 4 cached head inputs (f32) | 4 × N·T·2048 × 4 B ≈ 6.8 GB |
| DPT per-chunk activations | < 1 GB |

Comfortably inside 116 GB even at N = 1000 (~35 GB cached tensors). Memory is
not the constraint; do not spend effort on it beyond freeing the DINO/trunk
intermediates.

## 5. Compute budget & scaling

FLOP model (d=1024): GEMMs ≈ `N·T × 72 blocks × 24·d²` (DINO 24 + trunk 48);
global attention ≈ `19 × 4·(N·T)²·d`. Attention dominates from ~50 frames up
and scales **quadratically in N**:

| frames | attention | GEMMs | @ ~15 TFLOPS sustained |
|---|---|---|---|
| 100 | ~0.85 PFLOP | ~0.2 PFLOP | ~1–2 min |
| 200 | ~3.4 PFLOP | ~0.4 PFLOP | ~4–6 min |
| 500 | ~21 PFLOP  | ~1 PFLOP  | ~25–45 min |

(Cross-check: this model reproduces Meta's reported 240 s @ 1000 frames on an
A100.) Consequences:
- **Keyframe count is the quality/latency dial.** The client should default
  to 100–250 frames (05).
- ~256 GB/s bandwidth is adequate: these are large batched GEMMs and tiled
  attention, compute-bound at f16 WMMA rates. Only the DPT head is
  bandwidth-flavored and it is small.
- Submit work in command-buffer-sized chunks (per block or per few blocks) so
  the GPU stays fed but progress events (04) and cancellation stay responsive.

## 6. Preview / draft modes

1. **Frame-subset preview (safe, default).** Run the full model on every
   4th selected keyframe: ~16× faster, fully in-distribution. Client then
   requests the full set.
2. **Register-only trunk (experimental).** Routing *all* 24 inter-frame
   blocks through the register path drops attention FLOPs to ~6%. The paper
   reports this configuration reaches only original-VGGT quality — but it is
   ambiguous whether that variant was *trained* register-only; running the
   released checkpoint this way at inference is **unvalidated**. Implement
   behind a flag, evaluate against the parity scenes, and keep only if
   useful. Do not build product behavior on it before measuring.

## 7. Server process shape

- Headless Rust binary, owns the wgpu device and the loaded weights
  (load once, serve many sessions).
- One inference session at a time (the GPU is saturated anyway); queue
  further requests.
- All preprocessing besides ImageNet normalization happens on the client
  (05); the server validates only: uniform frame size, dims divisible by 16,
  N ≥ 1, and a frame-count cap from a config (compute guard).
- Cancellation: check a flag between command submissions; drop the session's
  buffers on cancel.
