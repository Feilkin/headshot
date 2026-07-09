// Tiled f32 GEMM for PyTorch Linear semantics: out (M,N) = x (M,K) @ w^T,
// w stored (N,K) row-major as in the checkpoint, optional bias (N).
// Correctness-first naive tiling; the WMMA path replaces this for f16.

struct Params {
    m: u32,
    k: u32,
    n: u32,
    has_bias: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

const TILE: u32 = 16u;
var<workgroup> xs: array<f32, 256>;
var<workgroup> ws: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) li: vec3<u32>,
) {
    let row = wg.y * TILE + li.y; // m
    let col = wg.x * TILE + li.x; // n
    var acc = 0.0;
    let tiles = (p.k + TILE - 1u) / TILE;
    for (var t = 0u; t < tiles; t++) {
        let kx = t * TILE + li.x;
        if row < p.m && kx < p.k {
            xs[li.y * TILE + li.x] = x[row * p.k + kx];
        } else {
            xs[li.y * TILE + li.x] = 0.0;
        }
        let ky = t * TILE + li.y;
        if col < p.n && ky < p.k {
            ws[li.y * TILE + li.x] = w[col * p.k + ky];
        } else {
            ws[li.y * TILE + li.x] = 0.0;
        }
        workgroupBarrier();
        for (var i = 0u; i < TILE; i++) {
            acc += xs[li.y * TILE + i] * ws[i * TILE + li.x];
        }
        workgroupBarrier();
    }
    if row < p.m && col < p.n {
        if p.has_bias != 0u {
            acc += bias[col];
        }
        out[row * p.n + col] = acc;
    }
}
