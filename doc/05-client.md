# 05 — Client: Preprocessing, Capture Handling, Visualization

bevy app. Owns everything before the model (media → frames) and everything
after it (points → screen/export). The frames it sends must match the
reference preprocessing exactly (§3) — the model is sensitive to sizing and
color conventions it was trained with.

## 1. Input media

| source | format | handling |
|---|---|---|
| DJI Mavic 2 Pro video | 2.7K H.264/H.265, **D-Log** color | decode → D-Log→Rec.709 LUT → keyframes |
| DJI SRT telemetry | subtitle track / sidecar `.srt` | parse per-frame GPS/alt/exposure (06) |
| Mirrorless photos | 14-bit RAW | develop to sRGB (neutral curve) |
| Smartphone photos | JPEG/HEIC (sRGB/P3) | decode, convert to sRGB |

### 1.1 Tonemapping

The model was trained on internet-style video and photos — i.e. standard
display-referred sRGB/Rec.709. **Never feed log footage raw** (flat contrast
is far out of distribution and wastes code values).

- D-Log: apply DJI's official D-Log → Rec.709 3D LUT (user supplies the
  `.cube`; ship a parametric fallback approximation). Neutral contrast, no
  creative grading, no aggressive saturation.
- RAW: develop with a standard neutral tone curve to sRGB (e.g. via
  `rawloader`/`imagepipe` or pre-converted externally). Avoid HDR tone
  mapping / local contrast — halos around edges are appearance changes the
  model must then explain away.
- Keep exposure differences between shots — the model is trained with heavy
  color jitter and handles them; don't over-normalize.

### 1.2 Capture guidance (documented in the app/UI)

- Overcast is ideal (diffuse light, no moving hard shadows between passes).
- Fast shutter (~1/200 s+): motion blur is the model's #1 failure mode.
- No lens-distortion model exists: prefer the phone's main 1× lens, never
  the ultra-wide; drone gimbal footage is fine.
- Avoid zooming mid-shot (abrupt FoV change is a listed failure mode).
- **Bridge viewpoints**: when mixing drone + ground captures, include a
  low-orbit / descent segment connecting aerial and ground-level views —
  registration relies on covisibility, and nadir-only + ground-only with
  nothing between is the risky case.
- Dynamic objects (people, cars) will ghost in the fused cloud; keep them
  minimal or plan to filter (future work).

## 2. Keyframe selection

Budget-driven: the server cost is quadratic in frame count (03 §5). Default
target **N ≈ 150–250** total across all sources; expose as a slider with a
time estimate.

Pipeline per video:
1. Base temporal sampling ~2 fps (drone speeds) → candidates.
2. Sharpness score per candidate: variance of Laplacian on a ~480px-wide
   grayscale downscale; within each window keep the sharpest.
3. Redundancy pruning: drop a candidate when the drone barely moved —
   use SRT GPS displacement when available (e.g. < 0.5 m and < 5° gimbal yaw
   change), else a cheap global frame-difference threshold.
4. If still over budget: uniform decimation to N, preserving first/last.

Photos: include all (deduplicate near-identical bursts by sharpness).
Frame 0 of the assembled batch = the reference frame; pick a mid-sequence,
scene-overview drone frame (everything is expressed in its camera space).
Record the mapping keyframe → (source file, timestamp/frame index, GPS) —
06 needs it.

## 3. Resize / crop to model resolution (must match reference exactly)

Given a decoded frame of size `w×h`, target resolution `R = 512`, patch 16:

1. If alpha: composite on white.
2. Center-crop the aspect ratio `ar = h/w` into `[0.5, 2.0]`:
   - `ar < 0.5`: crop width to `round(h/0.5)`, centered.
   - `ar > 2.0`: crop height to `round(w·2.0)`, centered.
3. **Balanced sizing** (keeps token count ≈ (R/16)² = 1024 regardless of
   aspect): with `ar = h/w` after cropping,
   `w_p = round(√(1024/ar))`, `h_p = round(1024/w_p)` (each ≥ 1, note h_p
   uses the *unrounded* w_p), target size = `(h_p·16, w_p·16)`.
   Example: 3:2 landscape → 624×416.
4. High-quality resample to target (Lanczos3 or area — the 2.7K source is
   ~5× supersampled, this is where that quality is banked; PIL-bicubic
   fidelity is not required, the model is resample-robust, but avoid nearest
   /bilinear shrink aliasing).
5. Output RGB8.

**All frames in one session must have identical target size.** The reference
pads mismatched sizes with white, but padding breaks the centered-principal-
point assumption — instead, pick one aspect bucket per session (from the
dominant source, e.g. 3:2 for the Mavic) and center-crop every other source
to that aspect before step 3. Warn the user when cropping photos.

The client keeps each preprocessed frame in memory: it is the color source
for the point cloud (pixel-exact same grid as the returned depth map).

## 4. bevy visualization

- Progressive point cloud: one GPU vertex/storage buffer region per depth
  chunk; points carry {position f32×3, color u8×4, conf f16, frame_idx u16}.
- Live controls (no re-inference): confidence percentile slider, depth-edge
  toggle, per-source visibility, point size; camera frusta with per-source
  coloring; click a frustum → show that source frame.
- After GPS alignment (06): optional grid/north/gravity gizmos, metric
  measurement tool.
- Renderer: start with instanced quads / point sprites; consider a compute-
  based visibility pass later if 100M+ points become common (200 frames ×
  260k px ≈ 50M raw points before filtering; expect ~10–30M after).

## 5. Export

- PLY (binary) with per-point color + confidence; optional LAS/LAZ when
  geo-referenced (06).
- Session bundle: keyframe manifest (source, timestamp, GPS), pose_enc,
  intrinsics, Sim(3) transform — everything needed to re-fuse or mesh later
  without re-running inference. Store depth/conf maps too (f16, ~1 MB/frame):
  the future meshing project (TSDF fusion) wants depth maps, not points.
