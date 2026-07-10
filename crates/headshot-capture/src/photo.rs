//! Photo handling (doc/05 §§1–2): EXIF metadata (orientation — which the
//! `image` crate does NOT auto-apply — capture time, GPS) and decode
//! routing: JPEG/PNG via `image`, HEIC via the ffmpeg backend, RAW via
//! rawler.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use exif::{In, Tag, Value};
use image::RgbImage;

use crate::error::CaptureError;
use crate::srt::GpsFix;
use crate::video::VideoBackend;

#[derive(Debug, Clone, Default)]
pub struct PhotoMeta {
    /// `DateTimeOriginal` normalized to `YYYY-MM-DDTHH:MM:SS`.
    pub capture_time: Option<String>,
    /// EXIF GPS (altitude is absolute/ellipsoidal — photos carry no
    /// takeoff-relative altitude).
    pub gps: Option<GpsFix>,
    /// EXIF orientation 1–8 (1 = upright).
    pub orientation: u32,
}

/// Read EXIF tolerantly: a photo with no (or broken) EXIF gets defaults,
/// never an error.
pub fn read_photo_meta(path: &Path) -> PhotoMeta {
    let mut meta = PhotoMeta { orientation: 1, ..Default::default() };
    let Ok(file) = File::open(path) else { return meta };
    let Ok(exif) = exif::Reader::new().read_from_container(&mut BufReader::new(file)) else {
        return meta;
    };

    if let Some(f) = exif.get_field(Tag::Orientation, In::PRIMARY)
        && let Some(o) = f.value.get_uint(0)
        && (1..=8).contains(&o)
    {
        meta.orientation = o;
    }
    if let Some(f) = exif.get_field(Tag::DateTimeOriginal, In::PRIMARY) {
        meta.capture_time = normalize_exif_datetime(&f.display_value().to_string());
    }

    let coord = |tag: Tag, ref_tag: Tag, neg: &str| -> Option<f64> {
        let f = exif.get_field(tag, In::PRIMARY)?;
        let Value::Rational(parts) = &f.value else { return None };
        let dms: Vec<f64> = parts.iter().take(3).map(|r| r.to_f64()).collect();
        let deg = dms.first()? + dms.get(1).copied().unwrap_or(0.0) / 60.0
            + dms.get(2).copied().unwrap_or(0.0) / 3600.0;
        let sign = exif
            .get_field(ref_tag, In::PRIMARY)
            .map(|r| r.display_value().to_string())
            .is_some_and(|r| r.trim().eq_ignore_ascii_case(neg));
        Some(if sign { -deg } else { deg })
    };
    if let (Some(lat), Some(lon)) = (
        coord(Tag::GPSLatitude, Tag::GPSLatitudeRef, "S"),
        coord(Tag::GPSLongitude, Tag::GPSLongitudeRef, "W"),
    ) && (lat.abs() > 1e-6 || lon.abs() > 1e-6)
    {
        let alt = exif
            .get_field(Tag::GPSAltitude, In::PRIMARY)
            .and_then(|f| match &f.value {
                Value::Rational(r) => r.first().map(|r| r.to_f64()),
                _ => None,
            })
            .map(|a| {
                let below = exif
                    .get_field(Tag::GPSAltitudeRef, In::PRIMARY)
                    .and_then(|f| f.value.get_uint(0))
                    == Some(1);
                if below { -a } else { a }
            });
        meta.gps = Some(GpsFix { lat, lon, rel_alt_m: None, abs_alt_m: alt });
    }
    meta
}

/// `"2025:06:23 08:25:01"` → `"2025-06-23T08:25:01"`.
fn normalize_exif_datetime(s: &str) -> Option<String> {
    let s = s.trim();
    let (date, time) = s.split_once([' ', 'T'])?;
    let date = date.replace(':', "-");
    if date.len() != 10 || time.len() < 8 {
        return None;
    }
    Some(format!("{date}T{time}"))
}

/// Apply EXIF orientation 1–8 (doc/05 §1: phones and mirrorless bodies
/// store sensor-native pixels + a rotation flag).
pub fn apply_orientation(img: RgbImage, orientation: u32) -> RgbImage {
    use image::imageops;
    match orientation {
        2 => imageops::flip_horizontal(&img),
        3 => imageops::rotate180(&img),
        4 => imageops::flip_vertical(&img),
        5 => imageops::flip_horizontal(&imageops::rotate90(&img)),
        6 => imageops::rotate90(&img),
        7 => imageops::flip_horizontal(&imageops::rotate270(&img)),
        8 => imageops::rotate270(&img),
        _ => img,
    }
}

/// Decode any supported photo to upright RGB8: JPEG/PNG/TIFF via `image`,
/// RAW via rawler (already upright), HEIC via the ffmpeg backend.
pub fn decode_photo(path: &Path, backend: &dyn VideoBackend) -> Result<RgbImage, CaptureError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    if crate::raw::is_raw_ext(&ext) {
        return crate::raw::develop_raw(path);
    }
    let meta = read_photo_meta(path);
    let img = match ext.as_str() {
        "heic" | "heif" => backend.decode_image_rgb8(path)?,
        _ => image::open(path)?.to_rgb8(),
    };
    Ok(apply_orientation(img, meta.orientation))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exif_datetime_normalization() {
        assert_eq!(
            normalize_exif_datetime("2025:06:23 08:25:01").as_deref(),
            Some("2025-06-23T08:25:01")
        );
        assert_eq!(
            normalize_exif_datetime("2025-06-23 08:25:01").as_deref(),
            Some("2025-06-23T08:25:01")
        );
        assert_eq!(normalize_exif_datetime("garbage"), None);
        assert_eq!(normalize_exif_datetime(""), None);
    }

    #[test]
    fn orientation_transforms() {
        // 2×3 image with a unique corner marker at (0, 0)
        let mut img = RgbImage::new(2, 3);
        img.put_pixel(0, 0, image::Rgb([255, 0, 0]));

        let upright = apply_orientation(img.clone(), 1);
        assert_eq!(upright.dimensions(), (2, 3));
        assert_eq!(upright.get_pixel(0, 0)[0], 255);

        // rotate90 CW: (0,0) → (h-1, 0) in the new (3×2) frame
        let r90 = apply_orientation(img.clone(), 6);
        assert_eq!(r90.dimensions(), (3, 2));
        assert_eq!(r90.get_pixel(2, 0)[0], 255);

        let r180 = apply_orientation(img.clone(), 3);
        assert_eq!(r180.dimensions(), (2, 3));
        assert_eq!(r180.get_pixel(1, 2)[0], 255);

        let r270 = apply_orientation(img.clone(), 8);
        assert_eq!(r270.dimensions(), (3, 2));
        assert_eq!(r270.get_pixel(0, 1)[0], 255);

        // transpose (5): (x,y) → (y,x), marker stays at origin
        let t = apply_orientation(img.clone(), 5);
        assert_eq!(t.dimensions(), (3, 2));
        assert_eq!(t.get_pixel(0, 0)[0], 255);

        // mirrored (2): marker moves to the right edge
        let m = apply_orientation(img, 2);
        assert_eq!(m.dimensions(), (2, 3));
        assert_eq!(m.get_pixel(1, 0)[0], 255);
    }

    #[test]
    fn missing_exif_yields_defaults() {
        let meta = read_photo_meta(Path::new("/nonexistent/photo.jpg"));
        assert_eq!(meta.orientation, 1);
        assert!(meta.gps.is_none());
        assert!(meta.capture_time.is_none());
    }
}
