// Axial 2D RoPE application (doc/01 §2.1): rotate-half over contiguous
// halves, applied to q or k in head-major layout (S, H, T, D), f32 tables
// (P, D). The first `prefix` tokens of every sequence pass through
// unrotated (5 in DINO, 17 in trunk frame blocks).
//
// One thread per rotation pair (d, d+D/2) so the in-place update is
// race-free.

@if(f16) enable f16;

@if(f16) alias Elem = f16;
@if(!f16) alias Elem = f32;

struct Params {
    s: u32,
    t: u32,
    h: u32,
    d: u32,
    prefix: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> x: array<Elem>;
@group(0) @binding(2) var<storage, read> sin_table: array<f32>;
@group(0) @binding(3) var<storage, read> cos_table: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let half = p.d / 2u;
    let i = (wg.y * nwg.x + wg.x) * 256u + li;
    let total = p.s * p.h * p.t * half;
    if i >= total {
        return;
    }
    let dp = i % half; // pair index: channels (dp, dp + half)
    let t = (i / half) % p.t;
    let h = (i / (half * p.t)) % p.h;
    let s = i / (half * p.t * p.h);
    if t < p.prefix {
        return;
    }
    let row = (t - p.prefix) * p.d;

    let base = ((s * p.h + h) * p.t + t) * p.d + dp;
    let lo = f32(x[base]);
    let hi = f32(x[base + half]);
    // rot(x) = concat(−x[half:], x[:half])
    x[base] = Elem(lo * cos_table[row + dp] - hi * sin_table[row + dp]);
    x[base + half] = Elem(hi * cos_table[row + half + dp] + lo * sin_table[row + half + dp]);
}
