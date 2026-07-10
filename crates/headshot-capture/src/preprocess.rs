//! Pixel path to model resolution (doc/05 §3): center-crop to the session
//! aspect, tonemap in f32 (video only — before resize, so Lanczos runs in
//! display-referred space like the training data), Lanczos3 resample,
//! quantize RGB8. The sizing *math* lives in `headshot_shared::sizing`;
//! this module only applies it to pixels.

use headshot_shared::sizing;
use image::{Rgb, Rgb32FImage, RgbImage};

use crate::lut::Tonemap;
use crate::video::Rgb48Frame;

/// Centered crop rect `(x, y, w, h)` for session aspect `th/tw`, shrunk by
/// `scale` (a centered zoom-in — legal because the principal point stays
/// centered and the camera head estimates FoV per frame). Public so the
/// review UI can draw the exact rect it will get.
pub fn centered_crop(w: u32, h: u32, tw: u32, th: u32, scale: f32) -> [u32; 4] {
    let (cw, ch) = sizing::crop_to_aspect(w, h, th as f32 / tw as f32);
    let s = scale.clamp(0.05, 1.0);
    let cw2 = ((cw as f32 * s).round() as u32).clamp(16, cw);
    // derive height from the scaled width so the crop aspect stays exact
    // up to rounding
    let ch2 = ((cw2 as f32 * ch as f32 / cw as f32).round() as u32).clamp(1, ch);
    [(w - cw2) / 2, (h - ch2) / 2, cw2, ch2]
}

/// Fraction of source pixels lost by `crop` (for the doc/05 §3 warning).
pub fn crop_loss(src_w: u32, src_h: u32, crop: [u32; 4]) -> f64 {
    1.0 - f64::from(crop[2]) * f64::from(crop[3]) / (f64::from(src_w) * f64::from(src_h))
}

/// rgb48 video frame → (RGB8 at `tw×th`, crop rect on the source).
pub fn preprocess_rgb48(
    f: &Rgb48Frame,
    tw: u32,
    th: u32,
    tone: &Tonemap,
    crop_scale: f32,
) -> (Vec<u8>, [u32; 4]) {
    let crop = centered_crop(f.width, f.height, tw, th, crop_scale);
    let [cx, cy, cw, ch] = crop;
    let mut float = Rgb32FImage::new(cw, ch);
    for y in 0..ch {
        let row = (cy + y) as usize * f.width as usize;
        for x in 0..cw {
            let i = (row + (cx + x) as usize) * 3;
            let px = [
                f32::from(f.data[i]) / 65535.0,
                f32::from(f.data[i + 1]) / 65535.0,
                f32::from(f.data[i + 2]) / 65535.0,
            ];
            float.put_pixel(x, y, Rgb(tone.apply_px(px)));
        }
    }
    let resized = image::imageops::resize(&float, tw, th, image::imageops::FilterType::Lanczos3);
    let rgb8 = resized.into_raw().iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect();
    (rgb8, crop)
}

/// Display-referred photo → (RGB8 at `tw×th`, crop rect). No tonemap
/// (doc/05 §1.1: photos are already sRGB).
pub fn preprocess_rgb8(img: &RgbImage, tw: u32, th: u32, crop_scale: f32) -> (Vec<u8>, [u32; 4]) {
    let crop = centered_crop(img.width(), img.height(), tw, th, crop_scale);
    let [cx, cy, cw, ch] = crop;
    let view = image::imageops::crop_imm(img, cx, cy, cw, ch).to_image();
    let resized = image::imageops::resize(&view, tw, th, image::imageops::FilterType::Lanczos3);
    (resized.into_raw(), crop)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb48_constant_frame_tonemaps_and_sizes() {
        // constant mid-grey D-Log frame; 16:9-ish source into a 3:2 session
        let (w, h) = (960, 540);
        let code = (0.584555f32 * 65535.0) as u16; // D-Log white
        let f = Rgb48Frame { width: w, height: h, data: vec![code; (w * h * 3) as usize] };
        let (rgb8, crop) = preprocess_rgb48(&f, 624, 416, &Tonemap::Parametric, 1.0);
        assert_eq!(rgb8.len(), 624 * 416 * 3);
        // 3:2 crop of a 960×540 source: width crops to 810, centered
        assert_eq!(crop, [75, 0, 810, 540]);
        // D-Log white → display white everywhere (constant image: resample
        // can't ring)
        assert!(rgb8.iter().all(|&v| v >= 254), "min {:?}", rgb8.iter().min());

        // None tonemap passes the code value through linearly
        let (rgb8, _) = preprocess_rgb48(&f, 624, 416, &Tonemap::None, 1.0);
        let expected = (0.584555f32 * 255.0).round() as u8;
        assert!(rgb8.iter().all(|&v| v.abs_diff(expected) <= 1));
    }

    #[test]
    fn rgb8_photo_crop_and_dims() {
        // portrait 3:4 photo into a landscape 3:2 session bucket
        let img = RgbImage::from_fn(600, 800, |x, _| {
            Rgb([(x % 256) as u8, 128, 40])
        });
        let (rgb8, crop) = preprocess_rgb8(&img, 624, 416, 1.0);
        assert_eq!(rgb8.len(), 624 * 416 * 3);
        // aspect 416/624 = 2/3 → height crops to 400, centered
        assert_eq!(crop, [0, 200, 600, 400]);
        assert!(crop_loss(600, 800, crop) > 0.49 && crop_loss(600, 800, crop) < 0.51);

        // same-aspect input loses nothing
        let img = RgbImage::new(624, 416);
        let (_, crop) = preprocess_rgb8(&img, 624, 416, 1.0);
        assert_eq!(crop, [0, 0, 624, 416]);
        assert_eq!(crop_loss(624, 416, crop), 0.0);
    }

    #[test]
    fn scaled_crop_stays_centered_and_aspect_true() {
        for (w, h) in [(2688u32, 1512u32), (600, 800), (3000, 2000), (640, 640)] {
            let full = centered_crop(w, h, 624, 416, 1.0);
            for scale in [0.3f32, 0.5, 0.77, 1.0] {
                let [x, y, cw, ch] = centered_crop(w, h, 624, 416, scale);
                // centered within ±1 px of the source center
                assert!(
                    (2 * x + cw).abs_diff(w) <= 1 && (2 * y + ch).abs_diff(h) <= 1,
                    "{w}x{h}@{scale}: off-center [{x},{y},{cw},{ch}]"
                );
                // aspect matches the unscaled crop up to rounding
                let want = f64::from(full[3]) / f64::from(full[2]);
                let got = f64::from(ch) / f64::from(cw);
                assert!((want - got).abs() < 0.01, "{w}x{h}@{scale}: aspect {got} vs {want}");
                // shrinks monotonically, never exceeds the full crop
                assert!(cw <= full[2] && ch <= full[3]);
            }
            // output dims are always the target regardless of scale
            let img = RgbImage::new(w, h);
            let (rgb8, _) = preprocess_rgb8(&img, 624, 416, 0.4);
            assert_eq!(rgb8.len(), 624 * 416 * 3);
        }
    }
}
