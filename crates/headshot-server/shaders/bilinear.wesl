// Bilinear resize with align_corners=True (doc/03 §1 item 8):
// src = dst · (in_size − 1) / (out_size − 1). NHWC, f32 only.

struct Params {
    n: u32,
    ih: u32,
    iw: u32,
    oh: u32,
    ow: u32,
    c: u32,
    _pad0: u32,
    _pad1: u32,
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
    let total = p.n * p.oh * p.ow * p.c;
    if i >= total {
        return;
    }
    let ci = i % p.c;
    let ox = (i / p.c) % p.ow;
    let oy = (i / (p.c * p.ow)) % p.oh;
    let ni = i / (p.c * p.ow * p.oh);

    var sy = 0.0;
    if p.oh > 1u {
        sy = f32(oy) * f32(p.ih - 1u) / f32(p.oh - 1u);
    }
    var sx = 0.0;
    if p.ow > 1u {
        sx = f32(ox) * f32(p.iw - 1u) / f32(p.ow - 1u);
    }
    let y0 = u32(floor(sy));
    let x0 = u32(floor(sx));
    let y1 = min(y0 + 1u, p.ih - 1u);
    let x1 = min(x0 + 1u, p.iw - 1u);
    let fy = sy - f32(y0);
    let fx = sx - f32(x0);

    let base = ni * p.ih * p.iw;
    let v00 = x[((base + y0 * p.iw + x0) * p.c) + ci];
    let v01 = x[((base + y0 * p.iw + x1) * p.c) + ci];
    let v10 = x[((base + y1 * p.iw + x0) * p.c) + ci];
    let v11 = x[((base + y1 * p.iw + x1) * p.c) + ci];
    let top = v00 * (1.0 - fx) + v01 * fx;
    let bot = v10 * (1.0 - fx) + v11 * fx;
    out[i] = top * (1.0 - fy) + bot * fy;
}
