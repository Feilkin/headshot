// 3×3 im2col for NHWC images, pad 1, stride 1 or 2 (doc/03 §1 item 5):
// (n, h, w, c) → rows (n·oh·ow, 9·c), column order (c, ky, kx) matching a
// PyTorch conv weight (oc, c, 3, 3) flattened to (oc, 9·c). The conv
// itself is the `linear` kernel on the result. f32 only.

struct Params {
    n: u32,
    h: u32,
    w: u32,
    c: u32,
    stride: u32,
    oh: u32,
    ow: u32,
    _pad: u32,
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
    let row_len = 9u * p.c;
    let total = p.n * p.oh * p.ow * row_len;
    if i >= total {
        return;
    }
    // decode (n, oy, ox, ci, ky, kx)
    let col = i % row_len;
    let kx = col % 3u;
    let ky = (col / 3u) % 3u;
    let ci = col / 9u;
    let pix = i / row_len;
    let ox = pix % p.ow;
    let oy = (pix / p.ow) % p.oh;
    let ni = pix / (p.ow * p.oh);

    let y = i32(oy * p.stride + ky) - 1;
    let xx = i32(ox * p.stride + kx) - 1;
    if y < 0 || y >= i32(p.h) || xx < 0 || xx >= i32(p.w) {
        out[i] = 0.0;
    } else {
        out[i] = x[((ni * p.h + u32(y)) * p.w + u32(xx)) * p.c + ci];
    }
}
