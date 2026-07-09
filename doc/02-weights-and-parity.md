# 02 — Weight Conversion and Parity Harness

## 1. Checkpoint

`huggingface.co/facebook/VGGT-Omega` → `vggt_omega_1b_512.pt` (gated access;
FAIR Noncommercial Research License — **never redistribute, in any format,
including converted files**). It is a PyTorch pickle of a flat
`state_dict` loadable directly into the reference `VGGTOmega` module (no
wrapper prefix).

The repo ships `tools/convert_weights.py` — a **standalone** script (imports
only `torch`, `safetensors`, `json`; must NOT import Meta's `vggt_omega`
package) that converts `.pt` → `model.safetensors` + `manifest.json`. Every
user runs it locally on their own downloaded checkpoint.

## 2. The masked-K-bias trap (critical)

Every attention qkv projection in the model (DINO blocks, aggregator frame &
inter-frame blocks, camera-head and text-head blocks) is a fused
`Linear(dim → 3·dim)` with a companion buffer `bias_mask` of the same shape
as the bias. At runtime the reference computes `bias · bias_mask`. In the
released `vggt_omega_1b_512.pt` two patterns occur (verified 2026-07-09
during conversion — an earlier revision of this doc wrongly claimed the
K-only pattern was universal):
- **DINO + trunk blocks (72×, dim 1024): the mask is all-zero** — the qkv
  bias is entirely disabled (the stored bias is also zero);
- **camera-head blocks (4×, dim 2048): 1.0 everywhere except the middle
  third (the K rows), which is 0.0** — only the K bias is zeroed.

In the module code, `bias_mask` is *initialized to NaN* and only becomes
valid when the checkpoint (or DINO init) fills it — so any mistake here
produces NaN everywhere, and any tool that drops "unused" buffers silently
corrupts the model.

Conversion rule: for every `*.attn.qkv.bias` with a sibling
`*.attn.qkv.bias_mask`:
1. assert `bias_mask` values ∈ {0.0, 1.0}, rows `[dim, 2·dim)` (K) all 0,
   and the Q and V thirds each uniformly 0 or uniformly 1 and equal to each
   other — fail loudly otherwise;
2. emit `bias := bias * bias_mask`;
3. drop `bias_mask` from the output.

After conversion the Rust side needs no masking logic at all.

## 3. Expected state_dict tree (prefixes)

```
aggregator.patch_embed.            # DINOv3 ViT-L/16
  cls_token (1,1,1024)  storage_tokens (1,4,1024)  mask_token (1,1024)  [mask_token unused at inference]
  patch_embed.proj.{weight (1024,3,16,16), bias}
  rope_embed.periods (16,)
  blocks.{0..23}.{norm1,norm2}.{weight,bias}
  blocks.{0..23}.attn.{qkv.weight (3072,1024), qkv.bias, qkv.bias_mask, proj.weight, proj.bias}
  blocks.{0..23}.{ls1,ls2}.gamma (1024,)
  blocks.{0..23}.mlp.{fc1 (4096,1024), fc2 (1024,4096)}.{weight,bias}
  norm.{weight,bias}               # final DINO LayerNorm
aggregator.rope_embed.periods (16,)
aggregator.camera_token   (1,2,1,1024)     # [ref-frame variant, other-frame variant]
aggregator.register_token (1,2,16,1024)
aggregator.frame_blocks.{0..23}.        # as DINO blocks PLUS:
  attn.{q_norm,k_norm}.{weight,bias} (64,)   # per-head QK LayerNorm
aggregator.inter_frame_blocks.{0..23}.  # same shape set as frame_blocks
camera_head.token_norm.{weight,bias} (2048,)
camera_head.trunk.{0..3}.               # blocks at dim 2048 (no q_norm/k_norm)
camera_head.trunk_norm.{weight,bias}
camera_head.camera_branch.{0,2}.{weight,bias}   # 2048→1024, 1024→9
dense_head.norm.{weight,bias} (2048,)
dense_head.projects.{0..3}.{weight,bias}        # 1×1 convs 2048→[256,512,1024,1024]
dense_head.resize_layers.{0,1,3}.{weight,bias}  # convT 4×4s4 / convT 2×2s2 / conv 3×3s2 (idx 2 = identity, absent)
dense_head.scratch.layer{1..4}_rn.weight        # 3×3, no bias
dense_head.scratch.refinenet{1..4}.{resConfUnit1?,resConfUnit2}.{conv1,conv2}.{weight,bias}
dense_head.scratch.refinenet{1..4}.out_conv.{weight,bias}
dense_head.proj.{weight,bias}       (16,256,1,1)
dense_head.proj_conf.{weight,bias}  (16,256,1,1)
```

Notes: `refinenet4` has no `resConfUnit1`. The buffers `_resnet_mean/std` are
non-persistent (absent from the checkpoint) — hardcode them. There is no
`text_alignment_head` in the default checkpoint. Treat unexpected/missing
keys as hard errors, print the full diff.

## 4. Precision policy at conversion

Checkpoint tensors are (mostly) f32 masters from bf16 training. Emit:
- **f16** for all big GEMM weights (DINO + aggregator + heads' linear/conv),
  with an **outlier scan**: any tensor containing `|w| > 65504` (f16 max) or
  where f16 rounding changes the value by > 1e-2 relative on > 0.1% of
  elements gets flagged and stored f32 instead (manifest records per-tensor
  dtype). Expected: zero or near-zero flagged tensors, but the guard is cheap
  and an unguarded inf poisons everything.
- **f32** always for: LayerNorm/QK-norm weights+biases, LayerScale gammas,
  `rope_embed.periods`, camera/register/cls/storage tokens, the 9-dim camera
  branch final layer, `proj`/`proj_conf`.

`manifest.json`: model config constants (from 01), tensor name → {shape,
dtype, offset}, checkpoint SHA256, conversion-tool version.

## 5. Parity harness (build this FIRST — it is the only way to debug kernels)

**Licensing note:** converted weights AND activation dumps are derived from
the FAIR-licensed checkpoint — they are local artifacts, generated by each
developer, and must never be committed or distributed. Parity tests
therefore cannot run in public CI; keep them behind an opt-in target
(`just parity`) that assumes the fixtures exist locally, and have CI run
only the checkpoint-free kernel unit tests.

Two halves:

**Python dump script** (`tools/dump_reference.py`, standalone, runs the
*reference* implementation in a venv — reference code stays outside our repo,
installed via `pip install` from Meta's GitHub):
- Fixed input: a small deterministic set (e.g. 4 frames, 128×96 — small P
  keeps dumps tiny; also one realistic 624×416 × 8-frame set).
- Forward hooks capture, in f32: DINO patch-embed output; DINO tokens after
  final norm; aggregator tokens after every frame block and every inter-frame
  block (48 tensors); the 4 cached concat tensors; camera-head trunk output;
  `pose_enc`; fused DPT map; `depth`, `depth_conf`.
- Save as one safetensors file + the exact preprocessed input tensor.

**Rust comparison test**: runs our engine on the dumped input, compares each
checkpoint tensor with: max-abs-error, mean-abs-error, cosine similarity.
Gates (initial, tune with experience):
- f32 stages (heads): max-abs < 1e-4, cosine > 1 − 1e-7.
- f16-compute trunk: drift grows with depth; gate on cosine > 0.999 per layer
  and eyeball the trend — a kernel bug shows as a cliff at one layer, not a
  gradual drift. For exact isolation, support an all-f32 mode in the engine
  (slow, debug only) with tight gates: max-abs < 1e-3 by layer 23.
- `pose_enc`: abs < 1e-3 per component. `depth`: relative < 1e-2 at f16,
  < 1e-3 at f32.

The reference wraps the trunk in bf16 autocast; for apples-to-apples, the
dump script should also produce an f32-forced variant (patch the autocast to
disabled) — that is the target for our all-f32 debug mode; the bf16 dump is
the realism check for our f16 pipeline.

## 6. Rust-side loading

- `safetensors` crate; mmap the file; upload per-tensor to GPU buffers grouped
  by stage (DINO / trunk / heads) so stages can be evicted independently
  (not that memory pressure requires it on 116 GB).
- Fuse at load time where kernels want it: e.g. pre-transpose weights to the
  GEMM layout the WMMA kernel expects; keep the manifest dtype authoritative.
- Validate every expected tensor exists with the expected shape before
  creating any pipeline; print a full diff on mismatch.
