# M3 viewer manual checklist

Milestone gate (doc/README M3): "bevy viewer renders the golden scene
progressively with working confidence-percentile and per-source filters
(manual checklist, recorded)."

Setup: `headshot-server` running with converted weights on the GPU box
(`--listen 0.0.0.0:9276` — the box is headless, so the viewer runs on
another machine); then `headshot-client <frames-dir> --server
<box-ip>:9276` on a machine with a display. The frames dir must exist on
the *client* machine (the client preprocesses and uploads). Record date,
scene, and pass/fail per item.

| # | Check | Expected |
|---|---|---|
| 1 | Launch during inference | status line shows preprocessing → upload → DINO → trunk k/24 progress |
| 2 | Camera frusta appear before any points | frusta drawn right after trunk finishes (CamerasMsg precedes chunks); frame 0 red, others cyan/yellow |
| 3 | Progressive cloud | points appear in visible increments as depth chunks stream, not all at once |
| 4 | Orbit + zoom | left-drag orbits around the cloud, wheel zooms, no gimbal flip within ±85° pitch |
| 5 | Confidence percentile `[` / `]` | point count visibly shrinks/grows; status shows quantile + threshold; no re-inference (instant-ish rebuild) |
| 6 | Frame-group filter `G` | all → even → odd cycles both points and frusta; disjoint subsets |
| 7 | Frusta toggle `F` | frusta hide/show |
| 8 | Failure path | stopping the server mid-session shows FAILED status, app stays responsive |

## Recorded runs

| date | scene | result | notes |
|---|---|---|---|
| _(pending)_ | | | |
