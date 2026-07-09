//! Resize/crop math to model resolution (doc/05 §3) — must match the
//! reference exactly; the model is sensitive to the sizing conventions it
//! was trained with. Target resolution R = 512, patch 16, token budget
//! (R/16)² = 1024.

const TOKEN_BUDGET: f32 = 1024.0;

/// Center-crop dimensions clamping aspect `h/w` into [0.5, 2.0]
/// (doc/05 §3 step 2). Returns (crop_w, crop_h).
pub fn clamp_aspect(w: u32, h: u32) -> (u32, u32) {
    let ar = h as f32 / w as f32;
    if ar < 0.5 {
        ((h as f32 / 0.5).round() as u32, h)
    } else if ar > 2.0 {
        (w, (w as f32 * 2.0).round() as u32)
    } else {
        (w, h)
    }
}

/// Balanced sizing (doc/05 §3 step 3): patch-grid (w_p, h_p) for an
/// aspect-clamped crop, keeping w_p·h_p ≈ 1024. `h_p` uses the *unrounded*
/// w_p. Target pixel size is (w_p·16, h_p·16).
pub fn balanced_grid(crop_w: u32, crop_h: u32) -> (u32, u32) {
    let ar = crop_h as f32 / crop_w as f32;
    let w_p_exact = (TOKEN_BUDGET / ar).sqrt();
    let w_p = w_p_exact.round().max(1.0) as u32;
    let h_p = (TOKEN_BUDGET / w_p_exact).round().max(1.0) as u32;
    (w_p, h_p)
}

/// Full pipeline for one source size: returns (crop_w, crop_h, target_w,
/// target_h) with target dims divisible by 16.
pub fn target_size(w: u32, h: u32) -> (u32, u32, u32, u32) {
    let (cw, ch) = clamp_aspect(w, h);
    let (w_p, h_p) = balanced_grid(cw, ch);
    (cw, ch, w_p * 16, h_p * 16)
}

/// Centered crop of `(w, h)` to match a session aspect `target_h/target_w`
/// (doc/05 §3: one aspect bucket per session; other sources are cropped to
/// it). Returns (crop_w, crop_h).
pub fn crop_to_aspect(w: u32, h: u32, aspect_h_over_w: f32) -> (u32, u32) {
    let ar = h as f32 / w as f32;
    if ar > aspect_h_over_w {
        (w, (w as f32 * aspect_h_over_w).round() as u32)
    } else {
        ((h as f32 / aspect_h_over_w).round() as u32, h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worked_example_3_2_landscape() {
        // doc/05 §3: 3:2 landscape → 624×416
        let (_, _, tw, th) = target_size(3000, 2000);
        assert_eq!((tw, th), (624, 416));
        // 2.7K DJI 16:9 video
        let (_, _, tw, th) = target_size(2688, 1512);
        assert_eq!(tw % 16, 0);
        assert_eq!(th % 16, 0);
    }

    #[test]
    fn properties_hold_over_many_sizes() {
        for (w, h) in [
            (4000u32, 3000u32),
            (3000, 4000),
            (1920, 1080),
            (1080, 1920),
            (5472, 3648),
            (100, 900),  // aspect 9 → clamped
            (900, 100),  // aspect 1/9 → clamped
            (512, 512),
            (517, 293),
        ] {
            let (cw, ch, tw, th) = target_size(w, h);
            let ar = ch as f32 / cw as f32;
            assert!((0.499..=2.001).contains(&ar), "{w}x{h}: crop aspect {ar}");
            assert_eq!(tw % 16, 0, "{w}x{h}");
            assert_eq!(th % 16, 0, "{w}x{h}");
            let tokens = (tw / 16) * (th / 16);
            assert!(
                (990..=1058).contains(&tokens),
                "{w}x{h}: {tokens} tokens (target {tw}x{th})"
            );
        }
    }

    #[test]
    fn session_aspect_crop() {
        // portrait photo cropped into a landscape session bucket
        let (cw, ch) = crop_to_aspect(3000, 4000, 416.0 / 624.0);
        assert_eq!(cw, 3000);
        assert_eq!(ch, 2000);
        // same-aspect input unchanged
        let (cw, ch) = crop_to_aspect(624, 416, 416.0 / 624.0);
        assert_eq!((cw, ch), (624, 416));
    }
}
