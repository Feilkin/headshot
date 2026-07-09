//! Binary PLY export (doc/06 §4): position, color, confidence, source
//! frame id.

use std::io::{self, BufWriter, Write};
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub struct PlyPoint {
    pub pos: [f32; 3],
    pub color: [u8; 3],
    pub conf: f32,
    pub frame: u16,
}

pub fn write_ply(path: &Path, points: &[PlyPoint]) -> io::Result<()> {
    let mut out = BufWriter::new(std::fs::File::create(path)?);
    write!(
        out,
        "ply\nformat binary_little_endian 1.0\ncomment headshot reconstruction\n\
         element vertex {}\n\
         property float x\nproperty float y\nproperty float z\n\
         property uchar red\nproperty uchar green\nproperty uchar blue\n\
         property float confidence\nproperty ushort frame\nend_header\n",
        points.len()
    )?;
    for p in points {
        for v in p.pos {
            out.write_all(&v.to_le_bytes())?;
        }
        out.write_all(&p.color)?;
        out.write_all(&p.conf.to_le_bytes())?;
        out.write_all(&p.frame.to_le_bytes())?;
    }
    out.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_and_size() {
        let dir = std::env::temp_dir().join(format!("headshot-ply-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.ply");
        let points = [
            PlyPoint { pos: [1.0, 2.0, 3.0], color: [10, 20, 30], conf: 0.5, frame: 7 },
            PlyPoint { pos: [-1.0, 0.0, 4.5], color: [0, 0, 255], conf: 2.0, frame: 0 },
        ];
        write_ply(&path, &points).unwrap();
        let data = std::fs::read(&path).unwrap();
        let header_end = data.windows(11).position(|w| w == b"end_header\n").unwrap() + 11;
        assert!(data.starts_with(b"ply\nformat binary_little_endian 1.0\n"));
        assert_eq!(data.len() - header_end, 2 * (12 + 3 + 4 + 2));
        // first float of first vertex
        assert_eq!(f32::from_le_bytes(data[header_end..][..4].try_into().unwrap()), 1.0);
        std::fs::remove_dir_all(&dir).ok();
    }
}
