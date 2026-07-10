// Reshape a per-token projection (S·T, H·D) into a head-major tensor
// (S, H, T, D) ready for QK-norm / RoPE / attention. Runs once per q, k, v
// (each projected by its own GEMM, so no fused 3·H·D activation is ever
// materialized — that buffer overflows wgpu's 2 GiB cap at high frame counts).

@if(f16) enable f16;

@if(f16) alias Elem = f16;
@if(!f16) alias Elem = f32;

struct Params {
    s: u32,
    t: u32,
    h: u32,
    d: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<Elem>;
@group(0) @binding(2) var<storage, read_write> out: array<Elem>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let i = (wg.y * nwg.x + wg.x) * 256u + li;
    let total = p.s * p.h * p.t * p.d;
    if i >= total {
        return;
    }
    // decode output index (s, h, t, d)
    let d = i % p.d;
    let t = (i / p.d) % p.t;
    let h = (i / (p.d * p.t)) % p.h;
    let s = i / (p.d * p.t * p.h);

    let hd = p.h * p.d;
    let row = (s * p.t + t) * hd + h * p.d + d;
    out[i] = x[row];
}
