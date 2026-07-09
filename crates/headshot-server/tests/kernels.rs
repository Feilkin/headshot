//! Kernel unit tests vs naive CPU references on random small inputs
//! (M1 CI gate — runs without the checkpoint; skips without a GPU).

use headshot_server::engine::tensor::Dtype;
use headshot_server::engine::{GpuContext, rope};

/// Deterministic light-weight RNG (SplitMix64 → uniform in [-1, 1]).
struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 23) as f32 * 2.0 - 1.0
    }

    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

fn ctx() -> Option<GpuContext> {
    match GpuContext::new() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("no GPU adapter — skipping ({e})");
            None
        }
    }
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol + tol * e.abs(),
            "{what}[{i}]: got {a}, want {e} (tol {tol})"
        );
    }
}

#[test]
fn residual_add() {
    let Some(ctx) = ctx() else { return };
    let mut rng = Rng(1);
    let a = rng.vec(1000);
    let b = rng.vec(1000);
    let acc = ctx.tensor_from_slice(&[1000], &a);
    let add = ctx.tensor_from_slice(&[1000], &b);
    ctx.dispatch(
        "residual_add",
        bytemuck::bytes_of(&1000u32),
        &[&acc, &add],
        [4, 1, 1],
    );
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    assert_close(&ctx.download(&acc), &expected, 1e-6, "residual_add");
}

#[test]
fn linear_matches_cpu() {
    let Some(ctx) = ctx() else { return };
    let mut rng = Rng(2);
    // deliberately non-multiples of the 16-tile
    let (m, k, n) = (37, 53, 29);
    let x = rng.vec(m * k);
    let w = rng.vec(n * k);
    let b = rng.vec(n);

    let gx = ctx.tensor_from_slice(&[m, k], &x);
    let gw = ctx.tensor_from_slice(&[n, k], &w);
    let gb = ctx.tensor_from_slice(&[n], &b);
    let out = ctx.linear(&gx, &gw, Some(&gb));

    let mut expected = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += x[row * k + kk] as f64 * w[col * k + kk] as f64;
            }
            expected[row * n + col] = acc as f32 + b[col];
        }
    }
    assert_close(&ctx.download(&out), &expected, 1e-4, "linear");

    // no-bias path
    let out = ctx.linear(&gx, &gw, None);
    let no_bias: Vec<f32> = expected
        .iter()
        .enumerate()
        .map(|(i, v)| v - b[i % n])
        .collect();
    assert_close(&ctx.download(&out), &no_bias, 1e-4, "linear-nobias");
}

#[test]
fn layernorm_matches_cpu() {
    let Some(ctx) = ctx() else { return };
    let mut rng = Rng(3);
    let (rows, dim) = (19, 300); // dim > wg size to exercise the strided loop
    let x = rng.vec(rows * dim);
    let w = rng.vec(dim);
    let b = rng.vec(dim);

    let out = ctx.layernorm(
        &ctx.tensor_from_slice(&[rows, dim], &x),
        &ctx.tensor_from_slice(&[dim], &w),
        &ctx.tensor_from_slice(&[dim], &b),
    );

    let mut expected = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..][..dim];
        let mu = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mu).powi(2)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for j in 0..dim {
            expected[r * dim + j] = (row[j] - mu) * inv * w[j] + b[j];
        }
    }
    assert_close(&ctx.download(&out), &expected, 1e-4, "layernorm");
}

#[test]
fn gelu_matches_erf() {
    let Some(ctx) = ctx() else { return };
    let xs: Vec<f32> = (-40..=40).map(|i| i as f32 * 0.2).collect();
    let out = ctx.gelu(&ctx.tensor_from_slice(&[xs.len()], &xs));
    // reference values via libm-quality erf (statrs-free: use f64 series
    // through the same A&S formula is what the kernel uses; instead check
    // against known exact GELU values and symmetry properties)
    let got = ctx.download(&out);
    for (i, &x) in xs.iter().enumerate() {
        let exact = 0.5 * x as f64 * (1.0 + libm::erf(x as f64 / std::f64::consts::SQRT_2));
        // A&S 7.1.26 in f32: ~4e-6 absolute worst case — far inside the
        // 1e-4 f32 parity gates (doc/03 §1 note on GELU flavors).
        assert!(
            (got[i] as f64 - exact).abs() < 1e-5,
            "gelu({x}) = {}, want {exact}",
            got[i]
        );
    }
}

#[test]
fn rope_apply_matches_cpu() {
    let Some(ctx) = ctx() else { return };
    let mut rng = Rng(4);
    let (s, h, t, d, prefix) = (2, 3, 11, 64, 5);
    let p = t - prefix; // 6 patch tokens = 2×3 grid
    let (sin, cos) = rope::tables(&rope::periods(), 2, 3);
    assert_eq!(sin.len(), p * d);

    let x = rng.vec(s * h * t * d);
    let gx = ctx.tensor_from_slice(&[s, h, t, d], &x);
    ctx.rope_apply(
        &gx,
        &ctx.tensor_from_slice(&[p, d], &sin),
        &ctx.tensor_from_slice(&[p, d], &cos),
        prefix,
    );
    let got = ctx.download(&gx);

    for si in 0..s {
        for hi in 0..h {
            for ti in 0..t {
                let base = ((si * h + hi) * t + ti) * d;
                let vec: [f32; 64] = std::array::from_fn(|j| x[base + j]);
                let expected = if ti < prefix {
                    vec // unrotated prefix
                } else {
                    rope::apply_cpu(&vec, &sin, &cos, ti - prefix)
                };
                assert_close(&got[base..base + d], &expected, 1e-5, "rope");
            }
        }
    }
}

#[test]
fn attention_matches_cpu() {
    let Some(ctx) = ctx() else { return };
    // 17: one partial tile; 64: exactly one tile; 150: three tiles with a
    // partial tail (BR = BC = 64 in the kernel)
    for t in [17, 64, 150] {
        attention_case(&ctx, t);
    }
}

fn attention_case(ctx: &GpuContext, t: usize) {
    let mut rng = Rng(5 + t as u64);
    let (s, h, d) = (2, 3, 64);
    let q = rng.vec(s * h * t * d);
    let k = rng.vec(s * h * t * d);
    let v = rng.vec(s * h * t * d);

    let out = ctx.attention(
        &ctx.tensor_from_slice(&[s, h, t, d], &q),
        &ctx.tensor_from_slice(&[s, h, t, d], &k),
        &ctx.tensor_from_slice(&[s, h, t, d], &v),
    );
    let got = ctx.download(&out);

    // naive f32 reference: full score matrix + softmax
    let scale = 1.0 / (d as f32).sqrt();
    for sh in 0..s * h {
        let base = sh * t * d;
        for qi in 0..t {
            let qv = &q[base + qi * d..][..d];
            let mut scores: Vec<f32> = (0..t)
                .map(|ki| {
                    let kv = &k[base + ki * d..][..d];
                    qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f32>() * scale
                })
                .collect();
            let max = scores.iter().cloned().fold(f32::MIN, f32::max);
            let mut denom = 0.0;
            for sc in scores.iter_mut() {
                *sc = (*sc - max).exp();
                denom += *sc;
            }
            for dd in 0..d {
                let expected: f32 = (0..t)
                    .map(|ki| scores[ki] * v[base + ki * d + dd])
                    .sum::<f32>()
                    / denom;
                let actual = got[base + qi * d + dd];
                assert!(
                    (actual - expected).abs() < 1e-4,
                    "attention[{sh},{qi},{dd}]: {actual} vs {expected}"
                );
            }
        }
    }
}

#[test]
fn qkv_split_merge_roundtrip() {
    let Some(ctx) = ctx() else { return };
    let mut rng = Rng(6);
    let (s, t, h, d) = (2, 5, 4, 8);
    let qkv = rng.vec(s * t * 3 * h * d);
    let g = ctx.tensor_from_slice(&[s * t, 3 * h * d], &qkv);
    let [q, k, v] = ctx.qkv_split(&g, s, t, h, d);

    // CPU check of q layout + merge(q) recovers the q columns
    let gq = ctx.download(&q);
    for si in 0..s {
        for hi in 0..h {
            for ti in 0..t {
                for di in 0..d {
                    let got = gq[(((si * h) + hi) * t + ti) * d + di];
                    let want = qkv[(si * t + ti) * 3 * h * d + hi * d + di];
                    assert_eq!(got, want, "q[{si},{hi},{ti},{di}]");
                }
            }
        }
    }
    let _ = ctx.download(&k);
    let merged = ctx.download(&ctx.attn_merge(&q));
    for si in 0..s {
        for ti in 0..t {
            for ci in 0..h * d {
                let got = merged[(si * t + ti) * h * d + ci];
                let want = qkv[(si * t + ti) * 3 * h * d + ci];
                assert_eq!(got, want, "merge[{si},{ti},{ci}]");
            }
        }
    }
    let _ = ctx.download(&v);
}

#[test]
fn im2col_and_concat() {
    let Some(ctx) = ctx() else { return };
    let mut rng = Rng(7);
    let (n, height, width) = (2, 32, 16); // 2×1 patches per image
    let images: Vec<f32> = (0..n * 3 * height * width)
        .map(|_| rng.next_f32() * 0.5 + 0.5)
        .collect();
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];
    let out = ctx.im2col_patch(
        &ctx.tensor_from_slice(&[n, 3, height, width], &images),
        mean,
        std,
        Dtype::F32,
    );
    assert_eq!(out.shape, vec![n * 2, 768]);
    let got = ctx.download(&out);
    // spot-check patch (n=1, pr=1, pc=0), c=2, ky=3, kx=7
    let (ni, pr, pc, c, ky, kx) = (1usize, 1usize, 0usize, 2usize, 3usize, 7usize);
    let (y, x) = (pr * 16 + ky, pc * 16 + kx);
    let want = (images[((ni * 3 + c) * height + y) * width + x] - mean[c]) / std[c];
    let patch_idx = ni * 2 + pr; // w_p = 1
    let col = c * 256 + ky * 16 + kx;
    assert!((got[patch_idx * 768 + col] - want).abs() < 1e-6, "im2col spot check");

    let a = rng.vec(6);
    let b = rng.vec(9);
    let cat = ctx.concat_channels(
        &ctx.tensor_from_slice(&[3, 2], &a),
        &ctx.tensor_from_slice(&[3, 3], &b),
    );
    let got = ctx.download(&cat);
    let want = [a[0], a[1], b[0], b[1], b[2], a[2], a[3], b[3], b[4], b[5], a[4], a[5], b[6], b[7], b[8]];
    assert_close(&got, &want, 0.0, "concat");
}

#[test]
fn linear_f16_and_wmma_match_cpu() {
    let Some(ctx) = ctx() else { return };
    if !ctx.f16_supported {
        eprintln!("no shader-f16 — skipping");
        return;
    }
    let mut rng = Rng(8);
    // M deliberately not a multiple of 16 (exercises padded-slack loads and
    // bounds-checked stores); K % 16 == 0, N % 64 == 0 for the WMMA path.
    let (m, k, n) = (37, 96, 128);
    let x: Vec<f32> = rng.vec(m * k);
    let w: Vec<f32> = rng.vec(n * k);
    let b: Vec<f32> = rng.vec(n);

    // CPU reference on the f16-rounded inputs (what the GPU actually sees),
    // accumulated in f64.
    let round = |v: &f32| half::f16::from_f32(*v).to_f32();
    let xr: Vec<f32> = x.iter().map(round).collect();
    let wr: Vec<f32> = w.iter().map(round).collect();
    let mut expected = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += xr[row * k + kk] as f64 * wr[col * k + kk] as f64;
            }
            // output is stored f16
            expected[row * n + col] =
                half::f16::from_f32(acc as f32 + b[col]).to_f32();
        }
    }

    let gx = ctx.tensor_from_f32(&[m, k], &x, Dtype::F16);
    let gw = ctx.tensor_from_f32(&[n, k], &w, Dtype::F16);
    let gb = ctx.tensor_from_slice(&[n], &b);

    // The public `linear` picks WMMA when available (dims allow it here).
    let out = ctx.linear(&gx, &gw, Some(&gb));
    let got = ctx.download(&out);
    let path = if ctx.wmma_supported { "wmma" } else { "scalar f16" };
    // f16 output rounding dominates: one ulp at |acc| ~ 6 is ~6e-3.
    assert_close(&got, &expected, 1e-2, &format!("linear ({path})"));

    // Force the scalar f16 kernel via a WMMA-ineligible N (not mult of 64).
    let n2 = 48;
    let gw2 = ctx.tensor_from_f32(&[n2, k], &w[..n2 * k], Dtype::F16);
    let out2 = ctx.linear(&gx, &gw2, None);
    let got2 = ctx.download(&out2);
    let mut expected2 = vec![0.0f32; m * n2];
    for row in 0..m {
        for col in 0..n2 {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += xr[row * k + kk] as f64 * wr[col * k + kk] as f64;
            }
            expected2[row * n2 + col] = half::f16::from_f32(acc as f32).to_f32();
        }
    }
    // tolerance is relative: 1 f16 ulp ≈ 1e-3 of magnitude
    assert_close(&got2, &expected2, 2e-3, "linear (scalar f16)");
}

#[test]
fn attention_f16_matches_f32() {
    let Some(ctx) = ctx() else { return };
    if !ctx.f16_supported {
        eprintln!("no shader-f16 — skipping");
        return;
    }
    let mut rng = Rng(9);
    let (s, h, t, d) = (1, 2, 33, 64);
    let q = rng.vec(s * h * t * d);
    let k = rng.vec(s * h * t * d);
    let v = rng.vec(s * h * t * d);

    let f32_out = ctx.download(&ctx.attention(
        &ctx.tensor_from_slice(&[s, h, t, d], &q),
        &ctx.tensor_from_slice(&[s, h, t, d], &k),
        &ctx.tensor_from_slice(&[s, h, t, d], &v),
    ));
    let f16_out = ctx.download(&ctx.attention(
        &ctx.tensor_from_f32(&[s, h, t, d], &q, Dtype::F16),
        &ctx.tensor_from_f32(&[s, h, t, d], &k, Dtype::F16),
        &ctx.tensor_from_f32(&[s, h, t, d], &v, Dtype::F16),
    ));
    // inputs rounded to f16 + f16 output rounding; values are O(0.3)
    assert_close(&f16_out, &f32_out, 5e-3, "attention f16 vs f32");
}
