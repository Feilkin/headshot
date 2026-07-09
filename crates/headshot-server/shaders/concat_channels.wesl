// Channel-dim concat for the cached head inputs (doc/01 §3.3):
// a (rows, c1) ++ b (rows, c2) → out (rows, c1+c2).

struct Params {
    rows: u32,
    c1: u32,
    c2: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let i = (wg.y * nwg.x + wg.x) * 256u + li;
    let c_out = p.c1 + p.c2;
    if i >= p.rows * c_out {
        return;
    }
    let c = i % c_out;
    let row = i / c_out;
    if c < p.c1 {
        out[i] = a[row * p.c1 + c];
    } else {
        out[i] = b[row * p.c2 + (c - p.c1)];
    }
}
