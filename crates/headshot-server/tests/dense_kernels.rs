//! Dense-head kernel unit tests vs CPU references (CI; skips without GPU).

use headshot_server::engine::GpuContext;
use headshot_server::engine::ops::UnaryOp;

fn rvec(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            (z >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        })
        .collect()
}

fn ctx() -> Option<GpuContext> {
    GpuContext::new()
        .inspect_err(|e| eprintln!("no GPU adapter — skipping ({e})"))
        .ok()
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol + tol * e.abs(),
            "{what}[{i}]: got {a}, want {e}"
        );
    }
}

#[test]
fn unary_ops() {
    let Some(ctx) = ctx() else { return };
    let x = rvec(1, 500);
    let g = ctx.tensor_from_slice(&[500], &x);
    let relu = ctx.download(&ctx.unary(&g, UnaryOp::Relu));
    let exp = ctx.download(&ctx.unary(&g, UnaryOp::Exp));
    let ope = ctx.download(&ctx.unary(&g, UnaryOp::OnePlusExp));
    for i in 0..500 {
        assert_eq!(relu[i], x[i].max(0.0));
        assert!((exp[i] - x[i].exp()).abs() < 1e-6);
        assert!((ope[i] - (1.0 + x[i].exp())).abs() < 1e-6);
    }
}

#[test]
fn tiled_add_broadcasts() {
    let Some(ctx) = ctx() else { return };
    let (n, block) = (3, 40);
    let x = rvec(2, n * block);
    let t = rvec(3, block);
    let got = ctx.download(&ctx.tiled_add(
        &ctx.tensor_from_slice(&[n, block], &x),
        &ctx.tensor_from_slice(&[block], &t),
    ));
    let want: Vec<f32> = x.iter().enumerate().map(|(i, v)| v + t[i % block]).collect();
    assert_close(&got, &want, 1e-6, "tiled_add");
}

/// im2col3x3 + linear vs a direct CPU conv (stride 1 and 2, pad 1).
#[test]
fn conv3x3_matches_cpu() {
    let Some(ctx) = ctx() else { return };
    for stride in [1usize, 2] {
        let (n, h, w, c, oc) = (2, 7, 5, 4, 3);
        let x = rvec(4, n * h * w * c);
        let wt = rvec(5, oc * c * 9); // (oc, c, 3, 3) row-major
        let bias = rvec(6, oc);

        let cols = ctx.im2col3x3(&ctx.tensor_from_slice(&[n, h, w, c], &x), stride);
        let out = ctx.linear(
            &cols,
            &ctx.tensor_from_slice(&[oc, 9 * c], &wt),
            Some(&ctx.tensor_from_slice(&[oc], &bias)),
        );
        let got = ctx.download(&out);

        let (oh, ow) = ((h - 1) / stride + 1, (w - 1) / stride + 1);
        for ni in 0..n {
            for oy in 0..oh {
                for ox in 0..ow {
                    for o in 0..oc {
                        let mut acc = bias[o];
                        for ci in 0..c {
                            for ky in 0..3usize {
                                for kx in 0..3usize {
                                    let y = (oy * stride + ky) as isize - 1;
                                    let xx = (ox * stride + kx) as isize - 1;
                                    if y < 0 || y >= h as isize || xx < 0 || xx >= w as isize {
                                        continue;
                                    }
                                    let v = x[((ni * h + y as usize) * w + xx as usize) * c + ci];
                                    acc += v * wt[o * c * 9 + ci * 9 + ky * 3 + kx];
                                }
                            }
                        }
                        let idx = ((ni * oh + oy) * ow + ox) * oc + o;
                        assert!(
                            (got[idx] - acc).abs() < 1e-4,
                            "conv s{stride} [{ni},{oy},{ox},{o}]: {} vs {acc}",
                            got[idx]
                        );
                    }
                }
            }
        }
    }
}

/// ConvTranspose kernel==stride as GEMM + shuffle_expand vs CPU convT.
#[test]
fn convt_matches_cpu() {
    let Some(ctx) = ctx() else { return };
    let (n, h, w, ic, oc, k) = (2, 3, 4, 5, 3, 2);
    let x = rvec(7, n * h * w * ic);
    // PyTorch ConvTranspose2d weight (in, out, k, k) row-major
    let wt = rvec(8, ic * oc * k * k);
    let bias = rvec(9, oc);

    // CPU-side weight repack (as the dense head does at load):
    // W'(j = (o·k+dy)·k+dx, ic)
    let mut wp = vec![0.0f32; oc * k * k * ic];
    for ci in 0..ic {
        for o in 0..oc {
            for dy in 0..k {
                for dx in 0..k {
                    let j = (o * k + dy) * k + dx;
                    wp[j * ic + ci] = wt[((ci * oc + o) * k + dy) * k + dx];
                }
            }
        }
    }
    // bias per output channel j → repeat per (dy,dx)
    let mut bp = vec![0.0f32; oc * k * k];
    for o in 0..oc {
        for dd in 0..k * k {
            bp[(o * k + dd / k) * k + dd % k] = bias[o];
        }
    }

    let t = ctx.linear(
        &ctx.tensor_from_slice(&[n * h * w, ic], &x),
        &ctx.tensor_from_slice(&[oc * k * k, ic], &wp),
        Some(&ctx.tensor_from_slice(&[oc * k * k], &bp)),
    );
    let got = ctx.download(&ctx.shuffle_expand(&t, n, h, w, k));

    for ni in 0..n {
        for yi in 0..h {
            for xi in 0..w {
                for dy in 0..k {
                    for dx in 0..k {
                        for o in 0..oc {
                            let mut acc = bias[o];
                            for ci in 0..ic {
                                acc += x[((ni * h + yi) * w + xi) * ic + ci]
                                    * wt[((ci * oc + o) * k + dy) * k + dx];
                            }
                            let (y, xx) = (yi * k + dy, xi * k + dx);
                            let idx = ((ni * h * k + y) * (w * k) + xx) * oc + o;
                            assert!(
                                (got[idx] - acc).abs() < 1e-4,
                                "convt [{ni},{y},{xx},{o}]: {} vs {acc}",
                                got[idx]
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn bilinear_align_corners() {
    let Some(ctx) = ctx() else { return };
    let (n, ih, iw, c) = (2, 3, 5, 2);
    let x = rvec(10, n * ih * iw * c);
    let (oh, ow) = (7, 4);
    let got = ctx.download(&ctx.bilinear(&ctx.tensor_from_slice(&[n, ih, iw, c], &x), oh, ow));

    for ni in 0..n {
        for oy in 0..oh {
            for ox in 0..ow {
                for ci in 0..c {
                    let sy = oy as f32 * (ih - 1) as f32 / (oh - 1) as f32;
                    let sx = ox as f32 * (iw - 1) as f32 / (ow - 1) as f32;
                    let (y0, x0) = (sy.floor() as usize, sx.floor() as usize);
                    let (y1, x1) = ((y0 + 1).min(ih - 1), (x0 + 1).min(iw - 1));
                    let (fy, fx) = (sy - y0 as f32, sx - x0 as f32);
                    let at = |y: usize, xx: usize| x[((ni * ih + y) * iw + xx) * c + ci];
                    let want = (at(y0, x0) * (1.0 - fx) + at(y0, x1) * fx) * (1.0 - fy)
                        + (at(y1, x0) * (1.0 - fx) + at(y1, x1) * fx) * fy;
                    let idx = ((ni * oh + oy) * ow + ox) * c + ci;
                    assert!(
                        (got[idx] - want).abs() < 1e-5,
                        "bilinear [{ni},{oy},{ox},{ci}]: {} vs {want}",
                        got[idx]
                    );
                }
            }
        }
    }

    // identity resize is exact
    let same = ctx.download(&ctx.bilinear(&ctx.tensor_from_slice(&[n, ih, iw, c], &x), ih, iw));
    assert_close(&same, &x, 1e-6, "bilinear identity");
}
