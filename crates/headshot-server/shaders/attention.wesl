// Non-causal flash attention (doc/03 §1 item 1), head_dim 64, f32 math.
// q, k, v head-major (S, H, T, D=64) → out same layout, bound as vec4
// arrays (D = 16 vec4s).
//
// One thread per query with online softmax (running max + denominator,
// f32), streaming K/V from global memory. All lanes of a wave read the
// same key row in lockstep, so K/V loads are effectively broadcasts served
// from cache — an LDS-staged variant of this kernel measured 3× *slower*
// (occupancy loss + barriers bought nothing; the kernel is ALU-bound).
// vec4 loads/MADs quarter the instruction count vs the scalar version.
//
// Dispatch grid: x = ceil(T/64) query tiles, y = S·H.

@if(f16) enable f16;

@if(f16) alias EV = vec4<f16>;
@if(!f16) alias EV = vec4<f32>;

struct Params {
    s: u32,
    t: u32,
    h: u32,
    scale: f32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<EV>;
@group(0) @binding(2) var<storage, read> k: array<EV>;
@group(0) @binding(3) var<storage, read> v: array<EV>;
@group(0) @binding(4) var<storage, read_write> out: array<EV>;

const DV: u32 = 16u; // head_dim in vec4s (64 / 4)

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let q_idx = wg.x * 64u + li;
    if q_idx >= p.t {
        return;
    }
    let seq_base = wg.y * p.t * DV; // one (s·h) slice per grid row
    let q_base = seq_base + q_idx * DV;

    var qv: array<vec4<f32>, DV>;
    for (var d = 0u; d < DV; d++) {
        qv[d] = vec4<f32>(q[q_base + d]);
    }

    var m = -3.4e38;
    var l = 0.0;
    var acc: array<vec4<f32>, DV>;
    for (var d = 0u; d < DV; d++) {
        acc[d] = vec4<f32>();
    }

    for (var kk = 0u; kk < p.t; kk++) {
        let k_base = seq_base + kk * DV;
        var score4 = vec4<f32>();
        for (var d = 0u; d < DV; d++) {
            score4 += qv[d] * vec4<f32>(k[k_base + d]);
        }
        let score = (score4.x + score4.y + score4.z + score4.w) * p.scale;

        let m_new = max(m, score);
        let corr = exp(m - m_new);
        let e = exp(score - m_new);
        l = l * corr + e;
        for (var d = 0u; d < DV; d++) {
            acc[d] = acc[d] * corr + e * vec4<f32>(v[k_base + d]);
        }
        m = m_new;
    }

    let inv_l = 1.0 / l;
    for (var d = 0u; d < DV; d++) {
        out[q_base + d] = EV(acc[d] * inv_l);
    }
}
