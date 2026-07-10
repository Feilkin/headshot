# 07 — Capture UI (bevy feathers)

Interactive front-end over the doc/05 pipeline (commit `5934a51`): Setup →
Review → Session. This doc is the map for UI follow-up work (theming pass,
UX fixes); the capture semantics live in doc/05 and `headshot-capture`.

## Architecture

```
crates/headshot-client/src/
  main.rs        app wiring: Args, Screen state selection, viewer systems,
                 Scene resource (point cloud + status), apply_viewer_event
  session.rs     CLI auto-flow (prepare_session) + run_protocol (shared)
  ui/mod.rs      Screen states {Setup, Review, Session}, CaptureUiPlugin,
                 WorkerTx/WorkerRx channels, PlanRes, SessionSettings,
                 pump_worker_events (single event drain for all screens)
  ui/worker.rs   background thread: Discover / Scan / Realize commands →
                 Progress / Discovered / Scanned / PlanReturned / Viewer events
  ui/setup.rs    SetupState (paths, budget, tonemap, discovery tree with
                 excluded_dirs/excluded_videos), path text field (primary —
                 see Wayland note), drag&drop, auto_discover, tree rows
  ui/review.rs   ReviewState (tab, focus), scrubber, preview + crop overlay,
                 include toggle, crop-scale slider, aspect cycling, auto
                 reselect, selected strip, reconstruct
  ui/log.rs      StatusLog resource + scrolling pane (Session state)
```

- All heavy work (decode, scoring, extraction, protocol) runs in
  `ui/worker.rs`; UI systems only mutate `SessionPlan` (pure methods in
  `headshot-capture/src/plan.rs`) and render it.
- Editing model: `SessionPlan.selected` is chronological; the reference is
  promoted to batch 0 only at realize time. `set_included` / `set_crop_scale`
  / `reselect` / `validate` are the whole edit API.
- CLI compatibility: `<media> …` runs the automatic flow (session.rs);
  `--review` opens Setup pre-filled; no args opens Setup; `--headless`
  needs media and never starts bevy.

## Theming (the design-pass target)

- Theme: `UiTheme(create_dark_theme())` in main.rs — replace with a custom
  theme built on `bevy::feathers::tokens::*`; widgets pick colors up from
  theme tokens, screen roots use `ThemeBackgroundColor(tokens::WINDOW_BG)`.
- Layout: `bsn!` trees in setup.rs/review.rs (flexbox Nodes); dynamic rows
  (tree, tabs, strip) are plain `commands.spawn` in rebuild systems —
  keep them cheap, they respawn on every state change.
- Fonts: feathers bundles Fira Sans + Fira Mono (`feathers::constants::fonts`).
- Colors currently hard-coded outside the theme (candidates for tokens):
  crop overlay (1.0, 0.8, 0.2), reference badge (1.0, 0.3, 0.2), tab/strip
  borders, log pane background.

## Constraints the design must keep

- Crop overlay is an honest preview: always centered, aspect fixed to the
  session bucket, zoom-only (doc/05 §3 principal-point assumption).
- One aspect + exact frame size per session (server rejects mixed).
- A non-drag path for adding media must stay primary: winit 0.30 has no
  Wayland file drop (X11/XWayland only, and cross-boundary drags from
  Wayland-native file managers often fail).
- Long operations stream progress lines; never block the UI thread.

## Known rough edges / follow-up list

- Scrubber `SliderValue` isn't reset when switching tabs (thumb position
  goes stale until moved).
- Selected strip re-uploads every thumbnail texture on any plan change
  (fine ≤ ~200 frames; cache per-unit handles if it grows).
- No back-navigation from Session to Review (the worker returns the plan
  via `PlanReturned`; a "back to review" button is prewired data-wise).
- Photos tab scrubs all photos including burst-rejected ones (marked in
  the info line only).
- Exact-frame scrubbing (beyond the ~2 fps candidate grid) would need
  seek-based single-frame ffmpeg extraction (~1–2 s per seek).
- Manual GUI checklist run over the villa dataset still pending (the
  headless CLI path is verified end-to-end against bigboy).
