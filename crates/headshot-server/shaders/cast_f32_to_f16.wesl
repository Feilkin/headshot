// Elementwise f32 → f16 storage cast (stage-boundary precision change,
// e.g. f32 DINO output → f16 trunk input).

enable f16;

struct Params {
    total: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f16>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let i = (wg.y * nwg.x + wg.x) * 256u + li;
    if i < p.total {
        out[i] = f16(x[i]);
    }
}
