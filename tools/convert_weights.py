#!/usr/bin/env python3
"""Convert the VGGT-Omega 1B checkpoint to safetensors + manifest.

Standalone by design (doc/02 §1): imports only torch, safetensors, and the
standard library — never Meta's ``vggt_omega`` package. Every user downloads
``vggt_omega_1b_512.pt`` from Hugging Face themselves (accepting the FAIR
Noncommercial Research License) and runs this locally. The outputs are
weights-derived and must never be committed or distributed.

What it does (doc/02):
  §3  validates the state_dict against the exact expected tree — any
      unexpected / missing / mis-shaped key is a hard error with a full diff;
  §2  applies the bias-mask rule: asserts every ``attn.qkv.bias_mask`` is
      {0,1} with the K third fully zeroed and Q/V thirds uniform (in the
      released checkpoint DINO/trunk masks are all-zero, camera-head masks
      zero only K), emits ``bias := bias * bias_mask``, drops the mask;
  §4  f16 outlier scan: big GEMM weights go to f16 unless |w| > f16 max or
      f16 rounding shifts > 0.1% of elements by > 1e-2 relative — those stay
      f32 (per-tensor dtype recorded in the manifest);
      norms / gammas / tokens / periods / final camera layer / proj heads
      always stay f32.

Usage:
  convert_weights.py path/to/vggt_omega_1b_512.pt -o fixtures/
Writes ``model.safetensors`` and ``manifest.json`` into the output directory.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

TOOL_VERSION = "0.1.1"

# Architecture constants (doc/01). The expected tree below is derived from
# these; they are also recorded in the manifest for the Rust side.
MODEL_CONFIG = {
    "model": "vggt-omega-1b-512",
    "patch_size": 16,
    "embed_dim": 1024,
    "num_heads": 16,
    "head_dim": 64,
    "mlp_ratio": 4,
    "num_dino_blocks": 24,
    "num_trunk_blocks": 24,
    "num_register_tokens": 16,
    "dino_prefix_tokens": 5,
    "trunk_prefix_tokens": 17,
    "register_attention_layers": [2, 6, 9, 14, 20],
    "cached_layer_indices": [4, 11, 17, 23],
    "camera_head_dim": 2048,
    "camera_head_blocks": 4,
    "camera_head_heads": 16,
    "dense_out_channels": [256, 512, 1024, 1024],
    "dense_features": 256,
    "pose_enc_len": 9,
    "layernorm_eps": 1e-5,
    "rope_base": 100.0,
    "imagenet_mean": [0.485, 0.456, 0.406],
    "imagenet_std": [0.229, 0.224, 0.225],
}

# Keys validated in the input but not emitted (beyond *.bias_mask, which is
# consumed by the masked-K-bias rule).
DROPPED_KEYS = {"aggregator.patch_embed.mask_token"}


def transformer_block(prefix: str, dim: int, qk_norm: bool, head_dim: int = 64) -> dict:
    """Expected tensors of one pre-norm block (doc/01 §2, §3.2)."""
    hidden = 4 * dim
    tree = {
        f"{prefix}.norm1.weight": (dim,),
        f"{prefix}.norm1.bias": (dim,),
        f"{prefix}.attn.qkv.weight": (3 * dim, dim),
        f"{prefix}.attn.qkv.bias": (3 * dim,),
        f"{prefix}.attn.qkv.bias_mask": (3 * dim,),
        f"{prefix}.attn.proj.weight": (dim, dim),
        f"{prefix}.attn.proj.bias": (dim,),
        f"{prefix}.ls1.gamma": (dim,),
        f"{prefix}.norm2.weight": (dim,),
        f"{prefix}.norm2.bias": (dim,),
        f"{prefix}.mlp.fc1.weight": (hidden, dim),
        f"{prefix}.mlp.fc1.bias": (hidden,),
        f"{prefix}.mlp.fc2.weight": (dim, hidden),
        f"{prefix}.mlp.fc2.bias": (dim,),
        f"{prefix}.ls2.gamma": (dim,),
    }
    if qk_norm:
        tree.update(
            {
                f"{prefix}.attn.q_norm.weight": (head_dim,),
                f"{prefix}.attn.q_norm.bias": (head_dim,),
                f"{prefix}.attn.k_norm.weight": (head_dim,),
                f"{prefix}.attn.k_norm.bias": (head_dim,),
            }
        )
    return tree


def expected_tree() -> dict[str, tuple[int, ...]]:
    """The full expected state_dict: name -> shape (doc/02 §3)."""
    c = MODEL_CONFIG
    dim = c["embed_dim"]
    tree: dict[str, tuple[int, ...]] = {}

    # DINOv3 ViT-L/16 (doc/01 §2)
    dino = "aggregator.patch_embed"
    tree[f"{dino}.cls_token"] = (1, 1, dim)
    tree[f"{dino}.storage_tokens"] = (1, 4, dim)
    tree[f"{dino}.mask_token"] = (1, dim)  # unused at inference, dropped
    tree[f"{dino}.patch_embed.proj.weight"] = (dim, 3, 16, 16)
    tree[f"{dino}.patch_embed.proj.bias"] = (dim,)
    tree[f"{dino}.rope_embed.periods"] = (16,)
    for i in range(c["num_dino_blocks"]):
        tree.update(transformer_block(f"{dino}.blocks.{i}", dim, qk_norm=False))
    tree[f"{dino}.norm.weight"] = (dim,)
    tree[f"{dino}.norm.bias"] = (dim,)

    # Aggregator trunk (doc/01 §3)
    tree["aggregator.rope_embed.periods"] = (16,)
    tree["aggregator.camera_token"] = (1, 2, 1, dim)
    tree["aggregator.register_token"] = (1, 2, c["num_register_tokens"], dim)
    for i in range(c["num_trunk_blocks"]):
        tree.update(transformer_block(f"aggregator.frame_blocks.{i}", dim, qk_norm=True))
        tree.update(transformer_block(f"aggregator.inter_frame_blocks.{i}", dim, qk_norm=True))

    # Camera head (doc/01 §4.1)
    cdim = c["camera_head_dim"]
    tree["camera_head.token_norm.weight"] = (cdim,)
    tree["camera_head.token_norm.bias"] = (cdim,)
    for i in range(c["camera_head_blocks"]):
        tree.update(transformer_block(f"camera_head.trunk.{i}", cdim, qk_norm=False))
    tree["camera_head.trunk_norm.weight"] = (cdim,)
    tree["camera_head.trunk_norm.bias"] = (cdim,)
    tree["camera_head.camera_branch.0.weight"] = (dim, cdim)
    tree["camera_head.camera_branch.0.bias"] = (dim,)
    tree["camera_head.camera_branch.2.weight"] = (c["pose_enc_len"], dim)
    tree["camera_head.camera_branch.2.bias"] = (c["pose_enc_len"],)

    # Dense (depth) head (doc/01 §4.2)
    ocs = c["dense_out_channels"]
    feat = c["dense_features"]
    tree["dense_head.norm.weight"] = (cdim,)
    tree["dense_head.norm.bias"] = (cdim,)
    for j, oc in enumerate(ocs):
        tree[f"dense_head.projects.{j}.weight"] = (oc, cdim, 1, 1)
        tree[f"dense_head.projects.{j}.bias"] = (oc,)
    # resize layers keep per-level channels; idx 2 is identity (absent)
    tree["dense_head.resize_layers.0.weight"] = (ocs[0], ocs[0], 4, 4)  # ConvT 4x4 s4
    tree["dense_head.resize_layers.0.bias"] = (ocs[0],)
    tree["dense_head.resize_layers.1.weight"] = (ocs[1], ocs[1], 2, 2)  # ConvT 2x2 s2
    tree["dense_head.resize_layers.1.bias"] = (ocs[1],)
    tree["dense_head.resize_layers.3.weight"] = (ocs[3], ocs[3], 3, 3)  # conv 3x3 s2
    tree["dense_head.resize_layers.3.bias"] = (ocs[3],)
    for k in range(1, 5):
        tree[f"dense_head.scratch.layer{k}_rn.weight"] = (feat, ocs[k - 1], 3, 3)
    for k in range(1, 5):
        rn = f"dense_head.scratch.refinenet{k}"
        units = ["resConfUnit2"] if k == 4 else ["resConfUnit1", "resConfUnit2"]
        for unit in units:
            for conv in ("conv1", "conv2"):
                tree[f"{rn}.{unit}.{conv}.weight"] = (feat, feat, 3, 3)
                tree[f"{rn}.{unit}.{conv}.bias"] = (feat,)
        tree[f"{rn}.out_conv.weight"] = (feat, feat, 1, 1)
        tree[f"{rn}.out_conv.bias"] = (feat,)
    shuffle_ch = (c["patch_size"] // 4) ** 2
    for proj in ("proj", "proj_conf"):
        tree[f"dense_head.{proj}.weight"] = (shuffle_ch, feat, 1, 1)
        tree[f"dense_head.{proj}.bias"] = (shuffle_ch,)

    return tree


def is_always_f32(name: str) -> bool:
    """Tensors that must stay f32 regardless of the outlier scan (doc/02 §4)."""
    return (
        # LayerNorm / QK-norm / final norms — all end in norm.weight/bias
        ".norm.weight" in name
        or ".norm.bias" in name
        or any(f".{n}." in name for n in ("norm1", "norm2", "q_norm", "k_norm", "token_norm", "trunk_norm"))
        or name.endswith((".gamma", ".periods"))
        or name
        in (
            "aggregator.camera_token",
            "aggregator.register_token",
            "aggregator.patch_embed.cls_token",
            "aggregator.patch_embed.storage_tokens",
        )
        # 9-dim final camera layer, depth/conf projection heads
        or name.startswith(("camera_head.camera_branch.2.", "dense_head.proj."))
        or name.startswith("dense_head.proj_conf.")
        # biases are tiny; keep them all f32
        or name.endswith(".bias")
    )


def validate_tree(state: dict) -> None:
    expected = expected_tree()
    actual = {k: tuple(v.shape) for k, v in state.items()}
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mis_shaped = sorted(
        k for k in set(expected) & set(actual) if expected[k] != actual[k]
    )
    if missing or unexpected or mis_shaped:
        for k in missing:
            print(f"MISSING    {k}  expected shape {expected[k]}", file=sys.stderr)
        for k in unexpected:
            print(f"UNEXPECTED {k}  shape {actual[k]}", file=sys.stderr)
        for k in mis_shaped:
            print(
                f"MIS-SHAPED {k}  expected {expected[k]}, got {actual[k]}",
                file=sys.stderr,
            )
        sys.exit(
            f"state_dict does not match the expected tree: "
            f"{len(missing)} missing, {len(unexpected)} unexpected, "
            f"{len(mis_shaped)} mis-shaped (doc/02 §3)"
        )
    print(f"tree OK: {len(expected)} tensors, all shapes match")


def apply_bias_mask_rule(state: dict) -> dict[str, int]:
    """doc/02 §2: fold bias_mask into bias, drop the mask.

    Two patterns exist in the released checkpoint:
    - DINO + trunk blocks: mask all zero (the qkv bias is entirely disabled);
    - camera-head blocks: only the K third zeroed.
    Common invariants enforced: values ∈ {0,1}, the K third is always fully
    zero, and each of the Q/V thirds is uniformly 0 or uniformly 1.
    Returns pattern counts.
    """
    masks = [k for k in state if k.endswith(".attn.qkv.bias_mask")]
    counts = {"all_zero": 0, "k_only": 0}
    for mask_key in masks:
        bias_key = mask_key.removesuffix("_mask")
        assert bias_key in state, f"{mask_key} has no sibling {bias_key}"
        mask = state[mask_key].float()
        n = mask.shape[0]
        assert n % 3 == 0, f"{mask_key}: length {n} not divisible by 3"
        dim = n // 3
        values = set(mask.unique().tolist())
        assert values <= {0.0, 1.0}, f"{mask_key}: values {values} not in {{0,1}}"
        q, k, v = mask[:dim], mask[dim : 2 * dim], mask[2 * dim :]
        assert (k == 0.0).all(), f"{mask_key}: K third not fully masked"
        for third_name, third in (("Q", q), ("V", v)):
            uniform = (third == 0.0).all() or (third == 1.0).all()
            assert uniform, f"{mask_key}: {third_name} third is neither all-0 nor all-1"
        assert torch.equal(q, v), f"{mask_key}: Q and V thirds differ"
        counts["all_zero" if (q == 0.0).all() else "k_only"] += 1
        state[bias_key] = state[bias_key] * mask.to(state[bias_key].dtype)
        del state[mask_key]
    assert sum(counts.values()) > 0, "no bias_mask keys found — wrong checkpoint?"
    return counts


def f16_outlier_scan(name: str, tensor: torch.Tensor) -> tuple[bool, dict]:
    """Return (keep_f32, report) for a GEMM-weight tensor (doc/02 §4)."""
    t = tensor.float()
    f16_max = torch.finfo(torch.float16).max
    max_abs = t.abs().max().item()
    overflow = max_abs > f16_max
    roundtrip = t.half().float()
    denom = t.abs().clamp_min(1e-12)
    rel_err = ((roundtrip - t).abs() / denom)
    bad_frac = (rel_err > 1e-2).float().mean().item()
    keep_f32 = overflow or bad_frac > 1e-3
    report = {
        "tensor": name,
        "max_abs": max_abs,
        "rel_err_gt_1e-2_frac": bad_frac,
        "overflow": overflow,
    }
    return keep_f32, report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", type=Path, help="path to vggt_omega_1b_512.pt")
    ap.add_argument(
        "-o", "--out-dir", type=Path, default=Path("fixtures"),
        help="output directory for model.safetensors + manifest.json",
    )
    args = ap.parse_args()

    print(f"hashing {args.checkpoint} ...")
    sha = hashlib.sha256()
    with open(args.checkpoint, "rb") as f:
        while chunk := f.read(1 << 24):
            sha.update(chunk)
    checkpoint_sha256 = sha.hexdigest()

    print("loading checkpoint (weights_only) ...")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    assert isinstance(state, dict), f"expected flat state_dict, got {type(state)}"

    validate_tree(state)
    mask_counts = apply_bias_mask_rule(state)
    print(
        f"bias-mask rule applied: {mask_counts['all_zero']} fully-masked "
        f"(DINO/trunk), {mask_counts['k_only']} K-only-masked (camera head)"
    )
    for k in DROPPED_KEYS:
        del state[k]

    out: dict[str, torch.Tensor] = {}
    flagged: list[dict] = []
    n_f16 = n_f32 = 0
    for name, tensor in state.items():
        tensor = tensor.contiguous()
        if is_always_f32(name) or tensor.dim() == 1:
            out[name] = tensor.float()
            n_f32 += 1
            continue
        keep_f32, report = f16_outlier_scan(name, tensor)
        if keep_f32:
            flagged.append(report)
            out[name] = tensor.float()
            n_f32 += 1
        else:
            out[name] = tensor.half()
            n_f16 += 1

    print(f"dtypes: {n_f16} f16, {n_f32} f32, {len(flagged)} flagged by outlier scan")
    for r in flagged:
        print(
            f"  FLAGGED {r['tensor']}: max_abs={r['max_abs']:.4g} "
            f"bad_frac={r['rel_err_gt_1e-2_frac']:.2e} overflow={r['overflow']}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "model.safetensors"
    save_file(out, model_path)

    manifest = {
        "config": MODEL_CONFIG,
        "checkpoint_sha256": checkpoint_sha256,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "conversion_tool_version": TOOL_VERSION,
        "bias_masks_applied": mask_counts,
        "dropped_keys": sorted(DROPPED_KEYS),
        "f16_outlier_report": flagged,
        "tensors": {
            name: {"shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch.")}
            for name, t in sorted(out.items())
        },
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    size_gb = model_path.stat().st_size / 2**30
    print(f"wrote {model_path} ({size_gb:.2f} GiB) and {manifest_path}")


if __name__ == "__main__":
    main()
