// Elementwise residual add: acc += addend (doc/03 §1).
// Exercised by tests/gpu_smoke.rs.

struct Params {
    total: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> acc: array<f32>;
@group(0) @binding(2) var<storage, read> addend: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let i = (wg.y * nwg.x + wg.x) * 256u + li;
    if i < p.total {
        acc[i] = acc[i] + addend[i];
    }
}
