//! Point-cloud export (doc/06 §4): binary PLY via the shared writer
//! (`headshot_shared::ply`, same format the server's `reconstruct` bin
//! emits) plus a cameras sidecar JSON with the same field names. All
//! points are written unfiltered; confidence rides along so downstream
//! tools can re-filter without re-inference. Coordinates are frame-0
//! camera space at arbitrary scale until Sim(3) alignment lands.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use headshot_shared::ply::{PlyPoint, write_ply_iter};
use headshot_shared::pose::Camera;

use crate::session::ChunkPoints;

pub struct ExportStats {
    pub points: usize,
    pub ply: PathBuf,
    pub cameras: PathBuf,
}

/// First free `headshot-cloud-NNN.ply` in `dir`.
pub fn next_free_path(dir: &Path) -> PathBuf {
    (1..)
        .map(|i| dir.join(format!("headshot-cloud-{i:03}.ply")))
        .find(|p| !p.exists())
        .expect("some export slot below usize::MAX is free")
}

/// Write `chunks` as binary PLY to `ply_path` and the cameras to a
/// `.cameras.json` sidecar next to it.
pub fn export_ply(
    chunks: &[Arc<ChunkPoints>],
    cameras: &[Camera],
    ply_path: &Path,
) -> Result<ExportStats> {
    let points: usize = chunks.iter().map(|c| c.positions.len()).sum();
    let iter = chunks.iter().flat_map(|c| {
        (0..c.positions.len()).map(move |i| PlyPoint {
            pos: c.positions[i],
            color: [0, 1, 2].map(|k| (c.colors[i][k] * 255.0 + 0.5).clamp(0.0, 255.0) as u8),
            conf: c.conf[i],
            frame: c.frame[i],
        })
    });
    write_ply_iter(ply_path, points, iter)
        .with_context(|| format!("writing {}", ply_path.display()))?;

    // same schema as the server reconstruct bin's sidecar, minus the
    // per-frame source/pose_enc it has and the session doesn't keep
    let cameras_path = ply_path.with_extension("cameras.json");
    let cams_json: Vec<serde_json::Value> = cameras
        .iter()
        .map(|cam| {
            serde_json::json!({
                "extrinsic_r": cam.r,
                "extrinsic_t": cam.t,
                "intrinsics": { "fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy },
                "center": cam.center(),
            })
        })
        .collect();
    let sidecar = serde_json::json!({
        "convention": "world-to-camera extrinsics (row-major R, t) + pinhole intrinsics \
                       (pixels), OpenCV axes, frame-0 world, arbitrary scale",
        "cameras": cams_json,
    });
    std::fs::write(&cameras_path, serde_json::to_string_pretty(&sidecar)?)
        .with_context(|| format!("writing {}", cameras_path.display()))?;

    Ok(ExportStats { points, ply: ply_path.to_owned(), cameras: cameras_path })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(n: usize, frame: u16) -> Arc<ChunkPoints> {
        Arc::new(ChunkPoints {
            positions: (0..n).map(|i| [i as f32, 2.0 * i as f32, -0.5]).collect(),
            colors: (0..n).map(|i| [i as f32 / n as f32, 0.5, 1.0, 1.0]).collect(),
            conf: (0..n).map(|i| i as f32 * 0.1).collect(),
            frame: vec![frame; n],
        })
    }

    #[test]
    fn ply_roundtrip() {
        let dir = std::env::temp_dir().join(format!("headshot-export-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = next_free_path(&dir);
        let cams = vec![Camera {
            r: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t: [0.1, 0.2, 0.3],
            fx: 500.0,
            fy: 501.0,
            cx: 344.0,
            cy: 192.0,
        }];
        let stats = export_ply(&[chunk(3, 0), chunk(2, 7)], &cams, &path).unwrap();
        assert_eq!(stats.points, 5);

        let bytes = std::fs::read(&path).unwrap();
        let header_end = bytes.windows(11).position(|w| w == b"end_header\n").unwrap() + 11;
        let header = std::str::from_utf8(&bytes[..header_end]).unwrap();
        assert!(header.starts_with("ply\nformat binary_little_endian 1.0\n"));
        assert!(header.contains("element vertex 5\n"));
        assert_eq!(bytes.len() - header_end, 5 * 21);

        // second record of the second chunk: position (1, 2, -0.5), frame 7
        let rec = &bytes[header_end + 4 * 21..][..21];
        assert_eq!(f32::from_le_bytes(rec[0..4].try_into().unwrap()), 1.0);
        assert_eq!(f32::from_le_bytes(rec[4..8].try_into().unwrap()), 2.0);
        assert_eq!(f32::from_le_bytes(rec[8..12].try_into().unwrap()), -0.5);
        assert_eq!(rec[13], 128); // g = 0.5
        assert_eq!(u16::from_le_bytes(rec[19..21].try_into().unwrap()), 7);

        let sidecar: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(&stats.cameras).unwrap()).unwrap();
        assert_eq!(sidecar["cameras"][0]["intrinsics"]["fx"], 500.0);
        assert_eq!(sidecar["cameras"].as_array().unwrap().len(), 1);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn next_free_skips_existing() {
        let dir = std::env::temp_dir().join(format!("headshot-nextfree-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(next_free_path(&dir).ends_with("headshot-cloud-001.ply"));
        std::fs::write(dir.join("headshot-cloud-001.ply"), b"x").unwrap();
        assert!(next_free_path(&dir).ends_with("headshot-cloud-002.ply"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
