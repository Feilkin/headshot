// Pixel-shuffle expansion (doc/03 §1 items 6–7): rows (n·h·w, k²·oc) →
// NHWC (n, h·k, w·k, oc) with channel index j = oc·k² + dy·k + dx.
// Serves both kernel==stride ConvTranspose (as GEMM + this) and the final
// ×4 pixel shuffle (oc = 1). f32 only.

struct Params {
    n: u32,
    h: u32,
    w: u32,
    k: u32,
    oc: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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
    let hk = p.h * p.k;
    let wk = p.w * p.k;
    let total = p.n * hk * wk * p.oc;
    if i >= total {
        return;
    }
    // decode output (n, y, x, o)
    let o = i % p.oc;
    let xx = (i / p.oc) % wk;
    let y = (i / (p.oc * wk)) % hk;
    let ni = i / (p.oc * wk * hk);

    let yi = y / p.k;
    let dy = y % p.k;
    let xi = xx / p.k;
    let dx = xx % p.k;
    let row = (ni * p.h + yi) * p.w + xi;
    let j = (o * p.k + dy) * p.k + dx;
    out[i] = x[row * (p.k * p.k * p.oc) + j];
}
