#!/usr/bin/env python3
"""Dump reference-implementation activations for the parity harness.

Runs Meta's reference ``vggt_omega`` package (installed in tools/.venv from
Meta's GitHub — its code never enters this repo) on deterministic synthetic
inputs and captures, via forward hooks, every tensor the parity gates of
doc/02 §5 compare against:

  images                       exact preprocessed input (N,3,H,W) f32 [0,1]
  dino.patch_embed             patch-embed conv output (prefix+patch tokens)
  dino.tokens                  DINO output after final norm (patch tokens)
  trunk.frame.{k:02}           output of aggregator frame block k   (24)
  trunk.inter.{k:02}           output of aggregator inter-frame block k (24;
                               register layers {2,6,9,14,20} carry only the
                               N*17 register+camera tokens — saved as-is)
  cache.{k:02}                 the 4 cached concat tensors (k in 4,11,17,23)
  camera.trunk_out             camera-head trunk output (after block 3)
  pose_enc                     (B,N,9)
  dense.fused                  fused DPT map after refinenet1, chunks concat
  depth, depth_conf            final outputs

Two scenes (doc/02 §5): small = 4 frames 128x96 (WxH), realistic = 8 frames
624x416. Two precision variants: "bf16" reproduces the reference autocast
(trunk bf16, heads f32); "f32" forces autocast off everywhere — the target
for the engine's all-f32 debug mode. On CPU the reference's
``device_type="cuda"`` autocast is inert, so this script rewrites it to
"cpu" for the bf16 variant (recorded in metadata).

Outputs are weights-derived: local artifacts only, never commit/distribute.

Usage:
  dump_reference.py fixtures/../vggt_omega_1b_512.pt -o fixtures/dumps \
      [--scene small realistic] [--variant bf16 f32] [--device cpu|cuda]
"""

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import save_file

SEED = 42
SCENES = {
    # name: (num_frames, width, height)
    "small": (4, 128, 96),
    "realistic": (8, 624, 416),
}
CACHED_LAYERS = [4, 11, 17, 23]


def synthetic_frames(n: int, w: int, h: int) -> torch.Tensor:
    """Deterministic structured frames in [0,1], (N,3,H,W) f32.

    Smooth per-frame-shifting gradients plus a moving inverted disk and a
    dash of seeded noise — arbitrary but reproducible; parity only needs
    determinism, not realism.
    """
    gen = torch.Generator().manual_seed(SEED)
    ys = torch.linspace(0.0, 1.0, h).view(h, 1).expand(h, w)
    xs = torch.linspace(0.0, 1.0, w).view(1, w).expand(h, w)
    tau = 2.0 * math.pi
    frames = []
    for i in range(n):
        phase = i / n
        r = 0.5 + 0.5 * torch.sin(tau * (3.0 * xs + phase))
        g = 0.5 + 0.5 * torch.sin(tau * (2.0 * ys - phase))
        b = 0.5 + 0.5 * torch.sin(tau * (1.5 * (xs + ys) + 2.0 * phase))
        img = torch.stack([r, g, b])
        cx, cy = 0.3 + 0.4 * phase, 0.6 - 0.2 * phase
        disk = ((xs - cx) ** 2 + (ys - cy) ** 2) < 0.03
        img = torch.where(disk, 1.0 - img, img)
        img = img + torch.rand((3, h, w), generator=gen) * 0.02
        frames.append(img.clamp(0.0, 1.0))
    return torch.stack(frames)


class Capture:
    """Collects hook outputs; concatenates when a module fires per chunk."""

    def __init__(self):
        self.store: dict[str, list[torch.Tensor]] = {}

    def hook(self, name: str, extract=None):
        def fn(_module, _inputs, output):
            t = output
            if extract is not None:
                t = extract(t)
            if isinstance(t, tuple):
                t = t[0]
            assert isinstance(t, torch.Tensor), f"{name}: unexpected hook output {type(t)}"
            self.store.setdefault(name, []).append(t.detach().float().cpu())

        return fn

    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            name: parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
            for name, parts in self.store.items()
        }


def patch_autocast(device: str, variant: str):
    """Rewrite the reference's internal ``torch.autocast`` calls.

    - f32 variant: force enabled=False everywhere (doc/02 §5).
    - bf16 on CPU: translate device_type "cuda" -> "cpu" so the trunk
      autocast (bf16) and the heads' enabled=False both take effect.
    """
    real_autocast = torch.autocast

    class Patched:
        def __init__(self, device_type, dtype=None, enabled=True, cache_enabled=None):
            if variant == "f32":
                enabled = False
            if device == "cpu" and device_type == "cuda":
                device_type = "cpu"
                if enabled and dtype is None:
                    dtype = torch.bfloat16
            self.ctx = real_autocast(
                device_type=device_type, dtype=dtype, enabled=enabled,
                cache_enabled=cache_enabled,
            )

        def __enter__(self):
            return self.ctx.__enter__()

        def __exit__(self, *exc):
            return self.ctx.__exit__(*exc)

    torch.autocast = Patched
    return real_autocast


def run_dump(model, images: torch.Tensor, variant: str, device: str) -> dict[str, torch.Tensor]:
    cap = Capture()
    handles = []

    def add(module, name, extract=None):
        handles.append(module.register_forward_hook(cap.hook(name, extract)))

    agg = model.aggregator
    add(agg.patch_embed.patch_embed, "dino.patch_embed")
    add(
        agg.patch_embed,
        "dino.tokens",
        extract=lambda out: out["x_norm_patchtokens"] if isinstance(out, dict) else out,
    )
    for k in range(len(agg.frame_blocks)):
        add(agg.frame_blocks[k], f"trunk.frame.{k:02}")
        add(agg.inter_frame_blocks[k], f"trunk.inter.{k:02}")
    # cached concat tensors from the aggregator's own return value
    add(
        agg,
        "cache",
        extract=lambda out: torch.stack(
            [out[0][k] for k in CACHED_LAYERS]
        ),
    )
    add(model.camera_head.trunk[-1], "camera.trunk_out")
    add(model.dense_head.scratch.refinenet1, "dense.fused")

    real_autocast = patch_autocast(device, variant)
    try:
        with torch.no_grad():
            predictions = model(images.to(device))
    finally:
        torch.autocast = real_autocast
        for h in handles:
            h.remove()

    tensors = cap.tensors()
    stacked_cache = tensors.pop("cache")
    for i, k in enumerate(CACHED_LAYERS):
        tensors[f"cache.{k:02}"] = stacked_cache[i]
    tensors["images"] = images.cpu()
    tensors["pose_enc"] = predictions["pose_enc"].detach().float().cpu()
    tensors["depth"] = predictions["depth"].detach().float().cpu()
    tensors["depth_conf"] = predictions["depth_conf"].detach().float().cpu()
    return {k: v.contiguous() for k, v in tensors.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", type=Path, help="path to vggt_omega_1b_512.pt")
    ap.add_argument("-o", "--out-dir", type=Path, default=Path("fixtures/dumps"))
    ap.add_argument("--scene", nargs="+", choices=sorted(SCENES), default=sorted(SCENES))
    ap.add_argument("--variant", nargs="+", choices=["bf16", "f32"], default=["bf16", "f32"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from vggt_omega.models.vggt_omega import VGGTOmega

    print("loading checkpoint ...")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = VGGTOmega(enable_camera=True, enable_depth=True, enable_alignment=False)
    model.load_state_dict(state)
    model.eval().to(args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for scene in args.scene:
        n, w, h = SCENES[scene]
        images = synthetic_frames(n, w, h)
        for variant in args.variant:
            print(f"running {scene} ({n}x{w}x{h}) {variant} on {args.device} ...")
            tensors = run_dump(model, images, variant, args.device)
            meta = {
                "scene": scene,
                "variant": variant,
                "device": args.device,
                "n_frames": n,
                "width": w,
                "height": h,
                "seed": SEED,
                "torch": torch.__version__,
                "shapes": {k: list(v.shape) for k, v in sorted(tensors.items())},
            }
            path = args.out_dir / f"dump_{scene}_{variant}.safetensors"
            save_file(tensors, path, metadata={"headshot": json.dumps(meta)})
            size_mb = path.stat().st_size / 2**20
            print(f"  wrote {path} ({len(tensors)} tensors, {size_mb:.0f} MiB)")


if __name__ == "__main__":
    main()
