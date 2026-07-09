// Residual add with LayerScale folded in (doc/03 §1 item 4):
// out[r,c] = res[r,c] + gamma[c] * x[r,c].

struct Params {
    total: u32,
    dim: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> res: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> gamma: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let i = (wg.y * nwg.x + wg.x) * 256u + li;
    if i < p.total {
        out[i] = res[i] + gamma[i % p.dim] * x[i];
    }
}
