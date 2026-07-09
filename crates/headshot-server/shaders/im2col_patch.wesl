// Patch embedding front half (doc/03 §1 item 9): fold ImageNet
// normalization (doc/01 §1) and 16×16-stride-16 patch extraction into one
// pass. images (N, 3, H, W) f32 in [0,1] → rows (N·P, 3·16·16) ordered
// (c, ky, kx) to match the conv weight layout (1024, 3, 16, 16); the
// patch-embed GEMM then runs the `linear` kernel.

@if(f16) enable f16;

@if(f16) alias Elem = f16;
@if(!f16) alias Elem = f32;

struct Params {
    n: u32,
    height: u32,
    width: u32,
    _pad: u32,
    mean: vec4<f32>, // xyz used
    inv_std: vec4<f32>,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> images: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<Elem>;

const PATCH: u32 = 16u;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let i = (wg.y * nwg.x + wg.x) * 256u + li;
    let w_p = p.width / PATCH;
    let h_p = p.height / PATCH;
    let row_len = 3u * PATCH * PATCH;
    let total = p.n * h_p * w_p * row_len;
    if i >= total {
        return;
    }
    // decode (n, patch_row, patch_col, c, ky, kx)
    let col = i % row_len;
    let kx = col % PATCH;
    let ky = (col / PATCH) % PATCH;
    let c = col / (PATCH * PATCH);
    let patch_idx = i / row_len;
    let pc = patch_idx % w_p;
    let pr = (patch_idx / w_p) % h_p;
    let n = patch_idx / (w_p * h_p);

    let y = pr * PATCH + ky;
    let x = pc * PATCH + kx;
    let value = images[((n * 3u + c) * p.height + y) * p.width + x];
    out[i] = Elem((value - p.mean[c]) * p.inv_std[c]);
}
