// Residual add with LayerScale folded in (doc/03 §1 item 4):
// out[r,c] = res[r,c] + gamma[c] * x[r,c].

@if(f16) enable f16;

@if(f16) alias Elem = f16;
@if(!f16) alias Elem = f32;

struct Params {
    total: u32,
    dim: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> res: array<Elem>;
@group(0) @binding(2) var<storage, read> x: array<Elem>;
@group(0) @binding(3) var<storage, read> gamma: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<Elem>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let i = (wg.y * nwg.x + wg.x) * 256u + li;
    if i < p.total {
        out[i] = Elem(f32(res[i]) + gamma[i % p.dim] * f32(x[i]));
    }
}
