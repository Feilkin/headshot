// Merge attention heads: (S, H, T, D) → (S·T, H·D), the input layout the
// projection GEMM expects.

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
    // decode input index (s, h, t, d)
    let d = i % p.d;
    let t = (i / p.d) % p.t;
    let h = (i / (p.d * p.t)) % p.h;
    let s = i / (p.d * p.t * p.h);
    out[(s * p.t + t) * (p.h * p.d) + h * p.d + d] = x[i];
}
