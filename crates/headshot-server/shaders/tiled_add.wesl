// out[i] = x[i] + b[i % block] — adds a per-image table (UV positional
// embedding, doc/01 §4.3, pre-scaled by 0.1 on the CPU) to every image of
// an NHWC batch (block = h·w·c). f32 only.

struct Params {
    total: u32,
    block: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let i = (wg.y * nwg.x + wg.x) * 256u + li;
    if i < p.total {
        out[i] = x[i] + b[i % p.block];
    }
}
