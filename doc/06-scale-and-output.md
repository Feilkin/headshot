# 06 — Metric Scale, Geo-Alignment, Outputs

The model's reconstruction lives in frame-0 camera coordinates at an
arbitrary scale. This document recovers metric scale (and optionally
geo-referencing + gravity alignment) from DJI telemetry, and defines outputs.

## 1. DJI SRT telemetry

Mavic 2 Pro embeds per-frame(-ish) telemetry as SRT subtitles when video
captions are enabled: GPS latitude/longitude, altitude (GPS and/or barometric
relative), and exposure data; some firmware includes gimbal angles. Parser
notes:
- Formats vary by firmware; write a tolerant line-oriented parser keyed on
  `latitude:`/`longitude:`/`rel_alt:`-style tokens, with unit tests over
  samples from our own aircraft.
- Associate each SRT entry to a video frame index via its timecode span;
  keyframes then inherit telemetry through the keyframe manifest (05 §2).
- Accuracy expectations: consumer GNSS ±2–5 m horizontal absolute (relative
  trajectory shape is smoother); barometric *relative* altitude is
  ~±0.1–0.5 m and smoother than GPS altitude — prefer it for the vertical
  axis (as offset from takeoff, so it is internally consistent per flight).

**Telemetry is post-hoc only.** It is never fed to the model (the model has
no conditioning inputs, and the paper reports auxiliary inputs at training
time were harmful). It is used purely to similarity-transform the finished
reconstruction.

## 2. Sim(3) alignment (Umeyama + RANSAC)

Inputs: predicted camera centers `c_i = −Rᵢᵀ tᵢ` (frame-0 space) for the
drone keyframes, and their GPS positions converted to a local **ENU** frame
(tangent plane at the first fix; use barometric relative altitude for U).

1. Candidate correspondences: all drone keyframes with telemetry (ground
   photos have none — they inherit scale by being in the same reconstruction).
2. RANSAC over minimal sets of 3 correspondences: solve Umeyama
   (least-squares similarity: rotation `S∈SO(3)`, scale `s>0`, translation
   `v`), count inliers by residual `‖s·S·c_i + v − enu_i‖ < τ` with
   `τ ≈ max(2 m, 2× measured GPS σ)`.
3. Final Umeyama on all inliers → `T_metric = (s, S, v)`.
4. Health checks: inlier ratio (< 60% ⇒ warn, likely GPS glitches or bad
   reconstruction), scale consistency across RANSAC winners, and residual
   RMS reported to the UI.

Accuracy expectation: over a flight path of a few hundred meters, ±3 m GPS
noise ⇒ ~1% scale error — ample for a research reconstruction.

Apply `T_metric` to: all points, all camera poses
(`R' = R·Sᵀ, t' = s·t − R·Sᵀ·v` for world→cam matrices — derive carefully
and unit-test round-trips: a transformed camera must still project a
transformed point to the same pixel). Bonus outputs for free: gravity-aligned
"up" (ENU U axis), true north, and absolute geolocation (record the ENU
origin's lat/lon in exports).

Fallback when no telemetry (photos-only session): scale stays arbitrary; the
UI can offer a manual two-point "known distance" scale tool.

## 3. Filtering recap (defaults)

Applied client-side at unprojection time (01 §5.3): finite check;
confidence-percentile threshold (default keep top ~50–80%, slider);
depth-edge filter (3×3 relative jump > 0.03 → drop) — this one removes the
smeared silhouette points that otherwise dominate visual quality; optional
sky removal (defer; the confidence + edge filters already kill most sky).

Dynamic-object ghosting (people, vehicles): out of scope for v1; note in the
session docs. Future option: cross-frame depth-consistency voting during
fusion, which suppresses movers and noise simultaneously.

## 4. Outputs

1. **PLY (binary)** — points: position f32×3 (metric if aligned), color u8×3,
   confidence f32, source frame id u16. Plus cameras as a sidecar JSON.
   *Implemented* (`headshot_shared::ply`; client Export button /
   `--export-ply`, server `reconstruct` bin) — currently frame-0
   coordinates at arbitrary scale, unfiltered with confidence per vertex.
2. **LAS/LAZ** (only when geo-aligned) — with WGS84/UTM georeference from the
   ENU origin.
3. **Session bundle** (directory or zip): keyframe manifest (source file,
   timestamp, GPS, sharpness), preprocessed frame hashes, `pose_enc`,
   derived intrinsics/extrinsics, depth+conf maps (f16), `T_metric`,
   filter settings, model/version hashes. This is the input contract for the
   follow-up meshing project (TSDF fusion + marching cubes want depth maps +
   poses, not points) and for COLMAP-refinement experiments (feed-forward
   output as bundle-adjustment initialization).

## 5. Validation plan for this stage

- Synthetic test: transform a known cloud by a random Sim(3), add Gaussian
  noise to "GPS", check recovery of s within noise bounds.
- Field test: fly a path with a measured ground truth distance (e.g. two
  markers a laser-measured distance apart), reconstruct, compare.
- Regression scene: keep one short drone clip + photo set with frozen
  keyframe selection as an end-to-end golden test (poses and scale stable
  across engine changes within tolerance).
