# 07 — Capture UI (bevy feathers)

Interactive front-end over the doc/05 pipeline: Setup → Review →
Reconstruct, styled after the design reference `headshot_ui_ref.html`
(three static 1600×900 renders on feathers design tokens). This doc is the
map for UI follow-up work; the capture semantics live in doc/05 and
`headshot-capture`.

## Architecture

```
crates/headshot-client/src/
  main.rs           app wiring: Args, Screen state selection, viewer systems,
                    Scene resource (point cloud + filters), apply_viewer_event
  session.rs        CLI auto-flow (prepare_session) + run_protocol (shared)
  ui/mod.rs         Screen states {Setup, Review, Session}, CaptureUiPlugin,
                    WorkerTx/WorkerRx channels, PlanRes, SessionSettings,
                    pump_worker_events (single event drain for all screens)
  ui/worker.rs      background thread: Discover / Scan / Realize commands →
                    Progress / Discovered / Scanned / PlanReturned / Viewer
  ui/theme.rs       app palette (CROP teal, HAZARD, GPS, borders) on top of
                    feathers palette; UiFonts handles; text/badge/checkbox
                    bundles for dynamic rows; t_* span helpers for bsn!;
                    shared header_bar + Setup › Review › Reconstruct crumbs
  ui/log.rs         StatusLog resource (rolling worker/session progress
                    lines; rendered by Setup + Reconstruct panes)
  ui/setup.rs       header · add-media path field (primary — see Wayland
                    note) · discovered tree with checkbox rows + RAW badge ·
                    sidebar: budget slider, tonemap pane, summary, scan
                    button, black activity/log pane
  ui/review.rs      header with source-tab chips + budget meter · preview
                    with teal crop overlay (corner ticks, model-frame tag,
                    file/sharpness/pipeline chips) · transport (step
                    buttons, timecode, tick markers, scrubber) · controls
                    (include toggle, zoom slider) · sidebar (frame-shape
                    option cards, budget pane, hazard reselect card,
                    summary card, Reconstruct) · selected strip with REF
  ui/reconstruct.rs floating panels over the 3D view: Filters (confidence
                    slider, frame-group radios, frusta toggle — two-way
                    sync with Scene + keyboard), server log, point-count
                    stats with Export PLY button (doc/06 §4; writes
                    headshot-cloud-NNN.ply + cameras sidecar to the cwd
                    on a background thread), help bar
  export.rs         binary PLY via headshot_shared::ply + cameras JSON;
                    also behind --export-ply (headless + auto flows)
```

- All heavy work (decode, scoring, extraction, protocol) runs in
  `ui/worker.rs`; UI systems only mutate `SessionPlan` (pure methods in
  `headshot-capture/src/plan.rs`) and render it.
- Editing model: `SessionPlan.selected` is chronological; the reference is
  promoted to batch 0 only at realize time. `set_included` / `set_crop_scale`
  / `reselect` / `validate` are the whole edit API. The frame-shape cards
  set `plan.aspect` directly (Auto + one card per source shape).
- CLI compatibility: `<media> …` runs the automatic flow (session.rs);
  `--review` opens Setup pre-filled; no args opens Setup; `--headless`
  needs media and never starts bevy.

## Theming

- Feathers widgets read `UiTheme(create_dark_theme())`; app chrome uses
  `ui/theme.rs` constants directly (one fixed dark theme, no custom token
  indirection). Fonts are the feathers-embedded Fira Sans/Mono.
- Static screens are `bsn!` trees; text spans that need a specific font go
  through `theme::t_sans/t_bold/t_mono` (wraps `InheritableFont`, whose
  `Handle<Font>` field accepts asset paths in templates — `TextFont`'s
  `FontSource` does not).
- Dynamic rows (tree, tabs, aspect cards, strip, tick markers, group
  radios) are `commands.spawn` bundles rebuilt in refresh systems using
  `theme::sans/mono/badge/check_square` with `UiFonts` handles — keep them
  cheap, they respawn on every state change.
- Gotcha: every `FeathersSlider` needs an explicit `SliderPrecision`
  component — feathers' fill/value-text sync system requires it in its
  query and `Slider` does not auto-require it; without it the slider
  renders its template placeholder forever.

## Constraints the design must keep

- Crop overlay is an honest preview: always centered, aspect fixed to the
  session bucket, zoom-only (doc/05 §3 principal-point assumption). The
  bottom-left readout spells out source → crop → model.
- One aspect + exact frame size per session (server rejects mixed); the
  frame-shape cards say which class gets centre-cropped and by how much.
- A non-drag path for adding media must stay primary: winit 0.30 has no
  Wayland file drop (X11/XWayland only, and cross-boundary drags from
  Wayland-native file managers often fail).
- Long operations stream progress lines (`StatusLog`); never block the UI
  thread.

## Known rough edges / follow-up list

- Selected strip re-uploads every thumbnail texture on any plan or focus
  change (fine ≤ ~200 frames; cache per-unit handles if it grows).
- No back-navigation from Session to Review (the worker returns the plan
  via `PlanReturned`; a "back to review" button is prewired data-wise).
- No cancel for a running reconstruction (needs a worker-side command).
- Photos tab scrubs all photos including burst-rejected ones (marked in
  the top-left chip only).
- Exact-frame scrubbing (beyond the ~2 fps candidate grid) would need
  seek-based single-frame ffmpeg extraction (~1–2 s per seek).
- Reconstruct overlay panels can collide on very narrow windows (the
  reference layout assumes ≥ ~1300 px width).
- The sharpness readout is the raw variance-of-Laplacian score; a
  session-relative scale would read better than absolute numbers.
