//! Axial 2D RoPE tables (doc/01 §2.1), shared by DINO and the aggregator
//! frame blocks. Computed once per session on the CPU in f32; the
//! `rope_apply` kernel consumes the sin/cos tables.

/// `periods[i] = base^(2i / (D_head/2))`, base 100, i = 0..15. The
/// checkpoint ships these as `rope_embed.periods`; loading them is
/// preferred, this is the cross-check / fallback.
pub fn periods() -> [f32; 16] {
    std::array::from_fn(|i| 100.0f32.powf(2.0 * i as f32 / 32.0))
}

/// Sin/cos tables for a patch grid of `h_p × w_p` (H', W'): each `(P, 64)`
/// row-major, P = h_p·w_p in h-outer/w-inner order.
///
/// Channels: `[0..16)` h-angles, `[16..32)` w-angles, `[32..64)` repeat of
/// `[0..32)`. Rotation (in the kernel) is rotate-half over contiguous
/// halves: `x' = x·cos + concat(−x[32:], x[:32])·sin`.
pub fn tables(periods: &[f32], h_p: usize, w_p: usize) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(periods.len(), 16);
    let p = h_p * w_p;
    let side = h_p.max(w_p) as f32;
    let coord = |i: usize| 2.0 * ((i as f32 + 0.5) / side) - 1.0;
    let tau = std::f32::consts::TAU;

    let mut sin = vec![0.0f32; p * 64];
    let mut cos = vec![0.0f32; p * 64];
    for row in 0..h_p {
        for col in 0..w_p {
            let idx = (row * w_p + col) * 64;
            for (i, &period) in periods.iter().enumerate() {
                let angle_h = tau * coord(row) / period;
                let angle_w = tau * coord(col) / period;
                for (offset, angle) in [(0, angle_h), (16, angle_w)] {
                    for half in [0, 32] {
                        sin[idx + half + offset + i] = angle.sin();
                        cos[idx + half + offset + i] = angle.cos();
                    }
                }
            }
        }
    }
    (sin, cos)
}

/// CPU reference of the rotation, for kernel unit tests: `x` is one head
/// vector (64), `table_row` indexes the token's patch position.
pub fn apply_cpu(x: &[f32; 64], sin: &[f32], cos: &[f32], table_row: usize) -> [f32; 64] {
    let s = &sin[table_row * 64..][..64];
    let c = &cos[table_row * 64..][..64];
    std::array::from_fn(|i| {
        let rot = if i < 32 { -x[i + 32] } else { x[i - 32] };
        x[i] * c[i] + rot * s[i]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn periods_formula() {
        let p = periods();
        assert_eq!(p[0], 1.0);
        // i=8: 100^(16/32) = 10
        assert!((p[8] - 10.0).abs() < 1e-4);
        assert!((p[15] - 100.0f32.powf(30.0 / 32.0)).abs() < 1e-3);
    }

    #[test]
    fn table_values() {
        // 2×3 grid: max side 3, coords over the longer axis hit
        // {-2/3, 0, 2/3}; the shorter axis {-1/3 ± ...}.
        let (sin, cos) = tables(&periods(), 2, 3);
        assert_eq!(sin.len(), 6 * 64);

        // token (row 1, col 2) = patch index 5; channel 16+0 is the w-angle
        // at period 1: angle = 2π·coord(2) = 2π·(2·(2.5/3)−1) = 2π·(2/3)
        let angle = std::f32::consts::TAU * (2.0 * (2.5 / 3.0) - 1.0);
        let idx = 5 * 64;
        assert!((sin[idx + 16] - angle.sin()).abs() < 1e-6);
        assert!((cos[idx + 16] - angle.cos()).abs() < 1e-6);
        // channel 0 is the h-angle at period 1 for row 1: coord = 0 → sin 0, cos 1
        assert!((sin[idx] - 0.0).abs() < 1e-6);
        assert!((cos[idx] - 1.0).abs() < 1e-6);
        // halves tile: channel j+32 equals channel j
        for j in 0..32 {
            assert_eq!(sin[idx + j], sin[idx + 32 + j]);
            assert_eq!(cos[idx + j], cos[idx + 32 + j]);
        }
    }

    #[test]
    fn rotation_preserves_norm() {
        // RoPE is a rotation: per-pair norms (i, i+32) are preserved.
        let (sin, cos) = tables(&periods(), 4, 4);
        let x: [f32; 64] = std::array::from_fn(|i| (i as f32 * 0.37).sin());
        let y = apply_cpu(&x, &sin, &cos, 7);
        for i in 0..32 {
            let before = x[i].hypot(x[i + 32]);
            let after = y[i].hypot(y[i + 32]);
            assert!((before - after).abs() < 1e-5, "pair {i}");
        }
    }
}
