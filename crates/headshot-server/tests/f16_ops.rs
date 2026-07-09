//! Every op's f16 variant vs its f32 variant on identical (f16-rounded)
//! inputs — catches a broken f16 kernel in isolation.

use headshot_server::engine::GpuContext;
use headshot_server::engine::tensor::Dtype;

fn rvec(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            // f16-exact values in [-1, 1): keep the comparison rounding-free
            half::f16::from_f32(((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0).to_f32()
        })
        .collect()
}

fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len());
    let max = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let nonzero = b.iter().filter(|v| **v != 0.0).count();
    assert!(nonzero > 0, "{what}: f32 reference is all zero?!");
    let a_nonzero = a.iter().filter(|v| **v != 0.0).count();
    assert!(a_nonzero > 0, "{what}: f16 output is ALL ZERO");
    assert!(max < tol, "{what}: max diff {max} > {tol}");
}

#[test]
fn every_op_f16_matches_f32() {
    let Some(ctx) = GpuContext::new().ok() else { return };
    if !ctx.f16_supported {
        return;
    }
    let (s, h, t, d) = (2, 4, 21, 64);
    let c = h * d;
    let rows = s * t;

    let x = rvec(1, rows * c);
    let up = |data: &[f32], shape: &[usize], dt| ctx.tensor_from_f32(shape, data, dt);

    // layernorm
    let w = rvec(2, c);
    let b = rvec(3, c);
    let f32_out = ctx.download(&ctx.layernorm(
        &up(&x, &[rows, c], Dtype::F32),
        &up(&w, &[c], Dtype::F32),
        &up(&b, &[c], Dtype::F32),
    ));
    let f16_out = ctx.download(&ctx.layernorm(
        &up(&x, &[rows, c], Dtype::F16),
        &up(&w, &[c], Dtype::F32),
        &up(&b, &[c], Dtype::F32),
    ));
    close(&f16_out, &f32_out, 2e-2, "layernorm");

    // gelu
    let f32_out = ctx.download(&ctx.gelu(&up(&x, &[rows, c], Dtype::F32)));
    let f16_out = ctx.download(&ctx.gelu(&up(&x, &[rows, c], Dtype::F16)));
    close(&f16_out, &f32_out, 2e-3, "gelu");

    // residual_ls
    let r = rvec(4, rows * c);
    let g = rvec(5, c);
    let f32_out = ctx.download(&ctx.residual_ls(
        &up(&r, &[rows, c], Dtype::F32),
        &up(&x, &[rows, c], Dtype::F32),
        &up(&g, &[c], Dtype::F32),
    ));
    let f16_out = ctx.download(&ctx.residual_ls(
        &up(&r, &[rows, c], Dtype::F16),
        &up(&x, &[rows, c], Dtype::F16),
        &up(&g, &[c], Dtype::F32),
    ));
    close(&f16_out, &f32_out, 2e-3, "residual_ls");

    // qkv_split + attn_merge
    let qkv = rvec(6, rows * 3 * c);
    let [q32, _, _] = ctx.qkv_split(&up(&qkv, &[rows, 3 * c], Dtype::F32), s, t, h, d);
    let [q16, _, _] = ctx.qkv_split(&up(&qkv, &[rows, 3 * c], Dtype::F16), s, t, h, d);
    close(&ctx.download(&q16), &ctx.download(&q32), 1e-6, "qkv_split");
    close(
        &ctx.download(&ctx.attn_merge(&q16)),
        &ctx.download(&ctx.attn_merge(&q32)),
        1e-6,
        "attn_merge",
    );

    // rope_apply
    let (sin, cos) = headshot_server::engine::rope::tables(
        &headshot_server::engine::rope::periods(),
        4,
        4,
    );
    let prefix = t - 16;
    let sin_t = up(&sin, &[16, 64], Dtype::F32);
    let cos_t = up(&cos, &[16, 64], Dtype::F32);
    let q = rvec(7, s * h * t * d);
    let g32 = up(&q, &[s, h, t, d], Dtype::F32);
    ctx.rope_apply(&g32, &sin_t, &cos_t, prefix);
    let g16 = up(&q, &[s, h, t, d], Dtype::F16);
    ctx.rope_apply(&g16, &sin_t, &cos_t, prefix);
    close(&ctx.download(&g16), &ctx.download(&g32), 2e-3, "rope_apply");

    // concat_channels
    let a = rvec(8, rows * 8);
    let bb = rvec(9, rows * 24);
    let f32_out = ctx.download(&ctx.concat_channels(
        &up(&a, &[rows, 8], Dtype::F32),
        &up(&bb, &[rows, 24], Dtype::F32),
    ));
    let f16_out = ctx.download(&ctx.concat_channels(
        &up(&a, &[rows, 8], Dtype::F16),
        &up(&bb, &[rows, 24], Dtype::F16),
    ));
    close(&f16_out, &f32_out, 1e-6, "concat_channels");
}
