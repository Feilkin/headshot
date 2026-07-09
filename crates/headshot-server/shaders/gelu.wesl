// Erf-based GELU matching PyTorch's default nn.GELU() (doc/03 §1).
// erf via Abramowitz & Stegun 7.1.26 (max abs error ~1.5e-7 — inside the
// all-f32 parity gates, unlike the tanh approximation).

struct Params {
    total: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

fn erf_approx(v: f32) -> f32 {
    let s = sign(v);
    let a = abs(v);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t * exp(-a * a);
    return s * y;
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let i = (wg.y * nwg.x + wg.x) * 256u + li;
    if i < p.total {
        let v = x[i];
        out[i] = 0.5 * v * (1.0 + erf_approx(v * 0.7071067811865475));
    }
}
