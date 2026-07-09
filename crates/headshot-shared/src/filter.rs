//! Point-cloud filtering (doc/01 §5.3, doc/06 §3), applied client-side at
//! unprojection time.

/// Confidence value at quantile `q` ∈ [0, 1] over the finite entries —
/// keep points with `conf >= threshold`.
pub fn confidence_threshold(conf: &[f32], q: f32) -> f32 {
    let mut finite: Vec<f32> = conf.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return f32::NEG_INFINITY;
    }
    let idx = ((finite.len() - 1) as f32 * q.clamp(0.0, 1.0)) as usize;
    let (_, nth, _) = finite.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
    *nth
}

/// Depth-edge mask for one (h, w) depth map: `true` = keep. Drops pixels
/// whose 3×3 neighborhood has a relative depth jump
/// `(max − min) / max(|d|, 1e-6) > 0.03` — removes the smeared silhouette
/// points that otherwise dominate visual quality (doc/01 §5.3). Windows
/// are clipped at the image border.
pub fn depth_edge_mask(depth: &[f32], h: usize, w: usize, threshold: f32) -> Vec<bool> {
    assert_eq!(depth.len(), h * w);
    let mut keep = vec![true; h * w];
    for y in 0..h {
        for x in 0..w {
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for yy in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                for xx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                    let d = depth[yy * w + xx];
                    lo = lo.min(d);
                    hi = hi.max(d);
                }
            }
            let center = depth[y * w + x];
            keep[y * w + x] = (hi - lo) / center.abs().max(1e-6) <= threshold;
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_threshold() {
        let conf: Vec<f32> = (1..=100).map(|v| v as f32).collect();
        let t = confidence_threshold(&conf, 0.5);
        assert!((49.0..=51.0).contains(&t), "median ≈ 50, got {t}");
        assert_eq!(confidence_threshold(&conf, 0.0), 1.0);
        assert_eq!(confidence_threshold(&conf, 1.0), 100.0);
        // NaNs are ignored
        let with_nan = [f32::NAN, 5.0, 1.0, 3.0];
        assert_eq!(confidence_threshold(&with_nan, 1.0), 5.0);
    }

    #[test]
    fn step_edge_is_dropped_flat_is_kept() {
        // 6×6 map: left half depth 1.0, right half depth 2.0
        let (h, w) = (6, 6);
        let depth: Vec<f32> =
            (0..h * w).map(|i| if i % w < 3 { 1.0 } else { 2.0 }).collect();
        let keep = depth_edge_mask(&depth, h, w, 0.03);
        for y in 0..h {
            for x in 0..w {
                let expected = !(2..=3).contains(&x); // columns astride the step
                assert_eq!(keep[y * w + x], expected, "({y},{x})");
            }
        }

        // a gentle ramp (1% per pixel) survives
        let ramp: Vec<f32> = (0..h * w).map(|i| 1.0 + 0.01 * (i % w) as f32).collect();
        assert!(depth_edge_mask(&ramp, h, w, 0.03).iter().all(|k| *k));
    }
}
