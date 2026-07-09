# `cargo install just` or `pacman -S just`

default: check test

check:
    cargo clippy --workspace --all-targets

test:
    cargo nextest run --workspace

# Parity suite (doc/02 §5). Requires locally generated fixtures (`just
# convert` + `just dump` first). Fixtures are derived from the FAIR-licensed
# checkpoint — never commit or distribute them. Set HEADSHOT_FIXTURES_DIR to
# use a non-default fixture location.
parity:
    cargo nextest run --workspace --run-ignored ignored-only -E 'test(parity)'

# One-time Python env for the two tools below (torch is CPU-build; plenty).
tools-venv:
    uv venv --python 3.12 tools/.venv
    uv pip install --python tools/.venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch
    uv pip install --python tools/.venv/bin/python safetensors "numpy<2" Pillow einops opencv-python-headless
    uv pip install --python tools/.venv/bin/python git+https://github.com/facebookresearch/vggt-omega

# Convert the downloaded checkpoint (doc/02 §§1–4).
convert ckpt out="fixtures":
    tools/.venv/bin/python tools/convert_weights.py {{ckpt}} -o {{out}}

# Dump reference activations for the parity gates (doc/02 §5).
dump ckpt out="fixtures/dumps":
    tools/.venv/bin/python tools/dump_reference.py {{ckpt}} -o {{out}}
