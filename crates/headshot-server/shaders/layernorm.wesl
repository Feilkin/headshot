// Row-wise LayerNorm, f32 statistics, learned weight+bias (doc/01: eps 1e-5
// everywhere). Also serves per-head QK-norm (rows = S·H·T, dim = 64).
// One 256-thread workgroup per row.

@if(f16) enable f16;

@if(f16) alias Elem = f16;
@if(!f16) alias Elem = f32;

struct Params {
    rows: u32,
    dim: u32,
    eps: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<Elem>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<Elem>;

var<workgroup> scratch: array<f32, 256>;
var<workgroup> mu: f32;
var<workgroup> inv_sigma: f32;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let row = wg.y * nwg.x + wg.x;
    // uniform per workgroup: safe around barriers
    if row >= p.rows {
        return;
    }
    let base = row * p.dim;

    var partial = 0.0;
    for (var j = li; j < p.dim; j += 256u) {
        partial += f32(x[base + j]);
    }
    scratch[li] = partial;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if li < s {
            scratch[li] += scratch[li + s];
        }
        workgroupBarrier();
    }
    if li == 0u {
        mu = scratch[0] / f32(p.dim);
    }
    workgroupBarrier();

    partial = 0.0;
    for (var j = li; j < p.dim; j += 256u) {
        let d = f32(x[base + j]) - mu;
        partial += d * d;
    }
    scratch[li] = partial;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if li < s {
            scratch[li] += scratch[li + s];
        }
        workgroupBarrier();
    }
    if li == 0u {
        inv_sigma = inverseSqrt(scratch[0] / f32(p.dim) + p.eps);
    }
    workgroupBarrier();

    for (var j = li; j < p.dim; j += 256u) {
        out[base + j] = Elem((f32(x[base + j]) - mu) * inv_sigma * weight[j] + bias[j]);
    }
}
