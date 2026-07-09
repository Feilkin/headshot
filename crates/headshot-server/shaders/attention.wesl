// Non-causal SDPA with online softmax (doc/03 §1 item 1), head_dim 64.
// q, k, v head-major (S, H, T, D=64) → out same layout. One thread per
// (sequence, head, query): serial pass over keys with running max /
// denominator — flash-attention semantics without tiling; numerically
// f32 throughout. Tiled/WMMA variant comes with the f16 path.

struct Params {
    s: u32,
    t: u32,
    h: u32,
    scale: f32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

const D: u32 = 64u;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let i = (wg.y * nwg.x + wg.x) * 64u + li;
    if i >= p.s * p.h * p.t {
        return;
    }
    let t = i % p.t;
    let h = (i / p.t) % p.h;
    let s = i / (p.t * p.h);

    let seq_base = (s * p.h + h) * p.t * D;
    let q_base = seq_base + t * D;

    var qv: array<f32, D>;
    for (var d = 0u; d < D; d++) {
        qv[d] = q[q_base + d];
    }

    var m = -3.4e38;
    var l = 0.0;
    var acc: array<f32, D>;
    for (var d = 0u; d < D; d++) {
        acc[d] = 0.0;
    }

    for (var kk = 0u; kk < p.t; kk++) {
        let k_base = seq_base + kk * D;
        var score = 0.0;
        for (var d = 0u; d < D; d++) {
            score += qv[d] * k[k_base + d];
        }
        score *= p.scale;

        let m_new = max(m, score);
        let corr = exp(m - m_new);
        let e = exp(score - m_new);
        l = l * corr + e;
        for (var d = 0u; d < D; d++) {
            acc[d] = acc[d] * corr + e * v[k_base + d];
        }
        m = m_new;
    }

    let inv_l = 1.0 / l;
    for (var d = 0u; d < D; d++) {
        out[q_base + d] = acc[d] * inv_l;
    }
}
