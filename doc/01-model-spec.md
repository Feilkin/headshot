# 01 — VGGT-Omega 1B Inference Spec

Everything needed to implement the forward pass. All constants verified
against the released reference implementation (v0.0.1) and the paper.
Shapes use `B` = batch (always 1 for us), `N` = frames, `H×W` = input pixels,
`H' = H/16`, `W' = W/16`, `P = H'·W'` patch tokens per frame, `C = 1024`.

## 0. Model I/O

Input: `images` — `(N, 3, H, W)` f32, RGB in `[0, 1]`, all frames identical
`H×W`, both divisible by 16. (Client delivers this; see 05.)

Output per forward pass:

| name | shape | meaning |
|---|---|---|
| `pose_enc` | `(B, N, 9)` | `[t(3), quat xyzw(4), fov_h, fov_w]` per frame |
| `depth` | `(B, N, H, W, 1)` | per-pixel depth, `> 0` |
| `depth_conf` | `(B, N, H, W)` | confidence, `≥ 1`, higher = more confident |
| `camera_and_register_tokens` | `(B, N, 17, 2048)` | final-layer camera+register tokens (optional export; useful for future heads) |

Conventions:
- **Extrinsics are camera-from-world (world→camera), OpenCV axes** (x right,
  y down, z forward): `p_cam = R · p_world + t`.
- Frame 0 is the **reference frame**: its predicted pose is ~identity; the
  whole scene lives in frame 0's camera coordinate system.
- **Scale is arbitrary** (training normalized scenes to unit scale). Metric
  scale is recovered downstream (06).
- Quaternion is scalar-last `xyzw` and is **not necessarily unit-norm**; the
  quat→matrix conversion must divide by `|q|²` (see §5.1).
- Principal point is assumed at the image center. Intrinsics from FoV:
  `fy = (H/2)/tan(fov_h/2)`, `fx = (W/2)/tan(fov_w/2)`, `cx = W/2`, `cy = H/2`.
- Depth is z-depth in each frame's own camera (not distance along ray).

## 1. Input normalization

First op in the model (do it server-side so the wire format stays RGB8):

```
x = (rgb01 - mean) / std ;  mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
```

## 2. Stage 1 — DINOv3 ViT-L/16 patch embedding (per frame)

A full 24-block ViT run **independently per frame** (batchable as N separate
sequences). Fine-tuned weights ship inside the checkpoint — no external
DINOv3 download.

Config: patch 16×16, dim 1024, 24 blocks, 16 heads (head_dim 64), MLP ratio 4
(GELU), LayerNorm eps **1e-5** everywhere, LayerScale init 1e-5 (learned
per-channel scale after both attention and MLP), qkv bias with **masked K
bias** (see 02 §2 — after conversion this is just a normal bias whose middle
third is zero), **no** QK-norm inside DINO.

Per frame:
1. Patch embed: 16×16-stride-16 conv (≡ linear over flattened 16×16×3
   patches) → `(P, 1024)`. No norm after the conv.
2. Prepend 1 `cls` token + 4 `storage` tokens (learned constants) →
   `(P+5, 1024)`.
3. 24 × pre-norm transformer block:
   `x += LS1(Attn(LN(x)))` ; `x += LS2(MLP(LN(x)))`.
   Attention is standard SDPA, scale `1/√64`, with **RoPE applied to the P
   patch tokens only** — the 5 prefix tokens get no rotation (RoPE tables
   have P rows; the first `seq_len − P` tokens are passed through).
4. Final LayerNorm over all tokens; **keep only the P patch tokens**
   (drop cls+storage) → `(P, 1024)` per frame.

### 2.1 RoPE (used identically in DINO and aggregator frame-attention)

Axial 2D RoPE, no learned parameters, computed in f32.

- `head_dim = 64`; frequencies: `D_head/4 = 16` periods
  `periods[i] = base^(2i / (D_head/2))`, `base = 100`, `i = 0..15`.
  (Stored in the checkpoint as buffer `rope_embed.periods`; load it rather
  than recomputing, values are identical.)
- Patch-center coordinates, **normalized by `max(H',W')`** then mapped to
  `[-1, 1]`:
  `coords_h[r] = 2·((r+0.5)/max(H',W')) − 1`, same for `coords_w`, meshgrid
  row-major (h outer, w inner), flattened to `(P, 2)`.
- Angles: `angles = 2π · coords[:, axis] / periods` → `(P, 2, 16)` →
  flatten last two dims → `(P, 32)` → **tile ×2** → `(P, 64)`.
  So channels `[0..32)` hold (h-freqs then w-freqs) and `[32..64)` repeat them.
- Rotation is **rotate-half over contiguous halves** (NOT interleaved pairs):
  `rot(x) = concat(−x[32:64], x[0:32])`;
  `x' = x·cos + rot(x)·sin`, applied to q and k per head, in f32, result cast
  back to compute dtype.

Same sin/cos table is shared by every layer and every frame (all frames are
the same size); compute once per session.

## 3. Stage 2 — Aggregator trunk

### 3.1 Token assembly

Per frame `i`, prepend to its P DINO tokens:
- 1 **camera token** and 16 **register tokens** — learned constants of shape
  `(2, 1, 1024)` and `(2, 16, 1024)`; variant index 0 is used for frame 0
  (reference), variant index 1 for all other frames.

Token order within a frame: `[camera(1), registers(16), patches(P)]`, i.e.
`patch_token_start = 17`, per-frame length `T = P + 17`.

### 3.2 Block structure

24 pairs of blocks, executed in order for `k = 0..23`:

1. **Frame block `k`** — self-attention over each frame independently:
   N sequences of length T. RoPE from §2.1 with 17-token unrotated prefix.
2. **Inter-frame block `k`** — one of:
   - **Global attention** (19 layers): one sequence of length `N·T`
     (frames concatenated in order). **No RoPE.**
   - **Register attention** (layers `k ∈ {2, 6, 9, 14, 20}`): only the first
     17 tokens of every frame participate — gather to one sequence of length
     `N·17` (frame-major order), run the block, scatter back; **patch tokens
     pass through this block completely unchanged**. No RoPE.

Both block types share one architecture (same as DINO blocks) **plus QK-norm**:
after splitting heads, apply LayerNorm(64) with learned weight+bias (eps 1e-5)
to q and to k separately (distinct parameters per block), *before* RoPE.
LayerScale init 1e-5, masked-K-bias qkv, MLP ratio 4, GELU, LayerNorm eps 1e-5.
`frame_blocks` and `inter_frame_blocks` are separate weight sets, 24 each.

### 3.3 Cached activations for the heads

After inter-frame blocks `k ∈ {4, 11, 17, 23}`, cache
`concat([frame_block_k_output, inter_frame_block_k_output], dim=channel)` →
`(B, N, T, 2048)`. (Frame-block output = the tokens *entering* the paired
inter-frame block.) Nothing else from the trunk is needed downstream; free
everything else. The reference runs the whole trunk under bf16 autocast; we
use f16-storage/f32-accumulate (03 §2) and cast the cached tensors to f32 at
head input.

## 4. Stage 3 — Heads (all f32)

### 4.1 Camera head

Input: cached layer 23 only, prefix tokens `(B, N, 17, 2048)`.

1. LayerNorm(2048).
2. Flatten to `(B, N·17, 2048)`; run **4 transformer blocks** at dim 2048,
   16 heads (head_dim 128), MLP ratio 4, LayerScale 1e-5, masked-K-bias,
   **no QK-norm, no RoPE** (plain attention over all frames' prefix tokens).
3. Take each frame's camera token (position 0 of each 17-group) →
   LayerNorm(2048) → `Linear(2048→1024)` → GELU → `Linear(1024→9)`.
4. Activation: components `[0:3]` (translation) and `[3:7]` (quaternion)
   unchanged; `[7:9]` (FoV, radians) → `relu(x) + 0.01`.

Single pass, no iterative refinement. Output is `pose_enc` (§0).

### 4.2 Dense (depth) head

Input: all four cached layers; **patch tokens only** (`[:, :, 17:, :]`).
Processed in **frame chunks of 8** to bound activation memory; chunks are
independent → this is the streaming unit (04).

A DPT-style pyramid, per chunk of `n ≤ 8` frames (batch `n·B`):

1. For each cached layer `l ∈ [4, 11, 17, 23]` (index `j = 0..3`):
   a. LayerNorm(2048) (shared single `norm` across the four layers)
   b. reshape to `(n, 2048, H', W')`
   c. 1×1 conv → channels `oc_j ∈ [256, 512, 1024, 1024]`
   d. **add UV positional embedding** (§4.3) scaled by 0.1
   e. resize: `j=0` → ConvTranspose 4×4 stride 4 (→ 4·H'); `j=1` →
      ConvTranspose 2×2 stride 2; `j=2` → identity; `j=3` → 3×3 stride-2
      conv, pad 1 (→ H'/2). NB: kernel==stride transpose convs are
      non-overlapping ≡ linear + pixel-shuffle.
2. Per level, a 3×3 conv (no bias) → 256 channels (`layer{1..4}_rn`).
3. Refinement cascade, deepest first (all at 256 ch):
   `out = RN4(l4)` → `RN3(out, l3)` → `RN2(out, l2)` → `RN1(out, l1)`.
   Fusion block `RNk(x, skip?)`:
   - if skip present: `x = x + RCU_a(skip)`
   - `x = RCU_b(x)`
   - bilinear-resize x to the next level's spatial size, **align_corners=True**
     (RN4→size of l3, RN3→size of l2, RN2→size of l1, RN1→size of l1, i.e.
     RN1's resize is a no-op)
   - 1×1 conv (256→256)
   RCU = `x + conv3×3(relu(conv3×3(relu(x))))` (biased convs, ReLU not in-place
   semantics — plain).
   RN4 has no `RCU_a`; RN1..RN3 do. Weight names: `refinenet{1..4}`,
   `resConfUnit1/2`, `out_conv`.
4. Fused map is `(n, 256, H/4, W/4)`. Add UV positional embedding again (×0.1).
5. Two independent 1×1 convs 256→16 (`proj` for depth, `proj_conf` for
   confidence) → pixel-shuffle ×4 → `(n, H, W, 1)` each.
6. `depth = exp(depth_logits)` ; `conf = 1 + exp(conf_logits)`.

### 4.3 UV positional embedding (MoGe-style)

For a feature map of size `(h, w)` with image aspect `a = W/H`:
- spans: `d = √(a²+1)`, `span_x = a/d`, `span_y = 1/d`
- `x_coords = linspace(−span_x·(w−1)/w, +span_x·(w−1)/w, w)`, same for y with
  `span_y, h`; meshgrid `indexing="xy"` → grid `(h, w, 2)` (u first).
- Sinusoidal embed to the feature channel count `E`: split `E/2` for u, `E/2`
  for v; for each: `omega_k = 1 / 100^(k/(E/4))`, `k = 0..E/4−1`;
  `emb = [sin(pos·omega), cos(pos·omega)]` concatenated → `(h, w, E)`.
- Add `0.1 × emb` to the feature map. (Reference computes the sinusoid in
  f64 then casts; f32 is fine — values are tiny — but note it in parity
  tolerances.)

## 5. Post-processing (reference semantics, implement in 06)

### 5.1 pose_enc → matrices

`quat_to_mat` (scalar-last, robust to non-unit q): with `s = 2/|q|²`, standard
quaternion rotation matrix. Extrinsic `E = [R | t]` (3×4, world→cam).
Intrinsics per §0.

### 5.2 Unprojection (depth → world points)

For integer pixel coords `(x, y)` (reference uses integer coordinates, no
half-pixel offset — match it):

```
p_cam   = [ (x − cx)/fx · d,  (y − cy)/fy · d,  d ]
p_world = Rᵀ · (p_cam − t)          # world = frame-0 camera space
```

### 5.3 Filtering defaults (from the reference demo, good starting values)

- Drop non-finite points/conf.
- Confidence threshold: percentile-based — keep `conf ≥ percentile(conf, q)`
  with `q ≈ 20–50` (%), applied over all valid pixels of the whole scene.
- **Depth-edge filter** (matters a lot for clean clouds): in each 3×3
  neighborhood compute `(max−min)/max(|d|, 1e-6)`; drop pixels where this
  relative jump `> 0.03`.
- Optional: sky segmentation mask (the demo uses an off-the-shelf ONNX sky
  segmenter; we can defer).

## 6. Constraints & known failure modes (paper)

- Trained at ~512²-pixel area, aspect ratio ∈ [0.33, 1.33] (as h/w or w/h,
  augmented); the loader enforces aspect ∈ [0.5, 2.0] via center crop. Do not
  feed higher resolutions expecting better output.
- Frames are **unordered** (permutation-invariant, only frame 0 is special);
  mixing sources (drone video frames + phone photos) in one pass is
  architecturally supported. Per-frame FoV handles mixed cameras.
- No lens-distortion model. Strong distortion (ultra-wide/fisheye) is a listed
  failure mode; so are strong motion blur and abrupt FoV changes.
- Dynamic scenes reconstruct per-frame, but there are no motion masks; moving
  objects will ghost when many frames are fused into one cloud.
- Trained frame count was 1–24 per sample, but the released model is
  demonstrated on 100s–1000+ frames (memory table in Meta's README goes to
  500 frames on one A100).
