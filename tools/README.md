# tools

Standalone Python scripts (M0, doc/02). One-time setup: `just tools-venv`
(creates `tools/.venv` with CPU torch, safetensors, and the reference
`vggt_omega` package installed from Meta's GitHub — the reference code never
enters this repo).

- `convert_weights.py` — converts the user-downloaded `vggt_omega_1b_512.pt`
  to `model.safetensors` + `manifest.json` (`just convert <ckpt>`). Validates
  the exact expected tensor tree (hard error with full diff), applies the
  masked-K-bias rule, runs the f16 outlier scan, records SHA-256 checksums.
  Imports only torch/safetensors/stdlib.
- `dump_reference.py` — runs the reference implementation on deterministic
  synthetic scenes and dumps the activation fixtures for the parity gates
  (`just dump <ckpt>`), in bf16-autocast and f32-forced variants. On CPU the
  realistic scene takes a few minutes per variant.

Then `just parity` runs the fixture-gated Rust tests.

Outputs of both are derived from the FAIR-licensed checkpoint: **local
artifacts only, never committed or distributed** (covered by .gitignore).
