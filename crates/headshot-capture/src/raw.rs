//! RAW development (doc/05 §1.1): rawler decode + neutral develop to
//! display-referred sRGB — rescale → demosaic → white balance → calibrate
//! → sRGB, no HDR tonemapping or local contrast. All rawler types stay
//! inside this module so an unsupported-camera swap stays one-module.

use std::path::Path;

use image::RgbImage;

use crate::error::CaptureError;

/// Extensions routed through rawler.
pub const RAW_EXTENSIONS: &[&str] =
    &["dng", "arw", "nef", "nrw", "cr2", "cr3", "raf", "orf", "rw2", "pef", "srw", "iiq"];

pub fn is_raw_ext(ext: &str) -> bool {
    RAW_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str())
}

/// Develop a camera RAW to sRGB RGB8 with rawler's neutral default
/// pipeline. Unsupported cameras and undecodable files surface as
/// `RawUnsupported`, whose message points at external DNG conversion.
pub fn develop_raw(path: &Path) -> Result<RgbImage, CaptureError> {
    let unsupported = |detail: String| {
        CaptureError::RawUnsupported(format!("{}: {detail}", path.display()))
    };
    let raw = rawler::decode_file(path).map_err(|e| unsupported(e.to_string()))?;
    let developed = rawler::imgop::develop::RawDevelop::default()
        .develop_intermediate(&raw)
        .map_err(|e| unsupported(e.to_string()))?
        .to_dynamic_image()
        .ok_or_else(|| unsupported("develop produced no image".into()))?;
    Ok(developed.to_rgb8())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_extensions_recognized() {
        assert!(is_raw_ext("ARW"));
        assert!(is_raw_ext("dng"));
        assert!(!is_raw_ext("jpg"));
        assert!(!is_raw_ext("mp4"));
    }

    #[test]
    fn garbage_input_is_raw_unsupported_with_dng_hint() {
        let path = std::env::temp_dir()
            .join(format!("headshot-capture-garbage-{}.arw", std::process::id()));
        std::fs::write(&path, b"definitely not a raw file").unwrap();
        let err = develop_raw(&path).expect_err("garbage must not develop");
        std::fs::remove_file(&path).ok();
        assert!(matches!(err, CaptureError::RawUnsupported(_)));
        assert!(err.to_string().contains("convert to DNG"), "{err}");
    }
}
