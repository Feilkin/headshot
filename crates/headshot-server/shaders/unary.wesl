// Elementwise unary ops for the dense head (doc/01 §4.2): ReLU (RCU),
// exp (depth), 1+exp (confidence). f32 only — heads run all-f32.

struct Params {
    total: u32,
    op: u32, // 0 = relu, 1 = exp, 2 = 1 + exp
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let i = (wg.y * nwg.x + wg.x) * 256u + li;
    if i >= p.total {
        return;
    }
    let v = x[i];
    switch p.op {
        case 0u: {
            out[i] = max(v, 0.0);
        }
        case 1u: {
            out[i] = exp(v);
        }
        default: {
            out[i] = 1.0 + exp(v);
        }
    }
}
