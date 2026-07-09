//! Client/server message schema (doc/04 §3): length-prefixed binary over
//! one duplex stream (plain TCP), little-endian throughout.
//!
//! Wire format per message: `u32` payload length, then payload =
//! `u8` tag + fields. Big blobs (RGB frames, depth/conf maps) are raw
//! little-endian arrays; depth and confidence travel as f16.

use std::io::{self, Read, Write};

/// Frame-count cap the server enforces by default (compute guard,
/// quadratic attention; doc/04 §6).
pub const DEFAULT_FRAME_CAP: u32 = 512;

/// Hard cap on a single message's payload (largest legal message is a
/// frame upload or an 8-frame depth chunk; 64 MiB bounds both).
pub const MAX_PAYLOAD: u32 = 64 << 20;

#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    // client → server
    OpenSession {
        session_id: u64,
        n_frames: u32,
        width: u32,
        height: u32,
        draft: bool,
        model: String,
    },
    Frame {
        session_id: u64,
        frame_idx: u32,
        rgb8: Vec<u8>,
    },
    Reconstruct {
        session_id: u64,
    },
    Cancel {
        session_id: u64,
    },

    // server → client
    FrameAck {
        frame_idx: u32,
        s1_done: bool,
    },
    Progress {
        stage: Stage,
        done: u32,
        total: u32,
    },
    Cameras {
        /// N × 9 pose encodings (doc/01 §0).
        pose_enc: Vec<f32>,
    },
    DepthChunk {
        first_frame_idx: u32,
        n_frames: u32,
        px_per_frame: u32,
        /// n·px f16 z-depths (raw bits).
        depth: Vec<u16>,
        /// n·px f16 confidences (raw bits).
        conf: Vec<u16>,
    },
    Done {
        s1_secs: f32,
        s2_secs: f32,
        s3_secs: f32,
        s4_secs: f32,
        peak_mem_bytes: u64,
        model_hash: String,
    },
    Error {
        code: ErrorCode,
        message: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    S1Dino = 1,
    S2Trunk = 2,
    S4Depth = 4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    Protocol = 1,
    Validation = 2,
    Cancelled = 3,
    Internal = 4,
}

#[derive(Debug, thiserror::Error)]
pub enum ProtocolError {
    #[error("io: {0}")]
    Io(#[from] io::Error),
    #[error("payload length {0} exceeds MAX_PAYLOAD")]
    TooLarge(u32),
    #[error("truncated or malformed message (tag {tag})")]
    Malformed { tag: u8 },
    #[error("unknown message tag {0}")]
    UnknownTag(u8),
}

impl Message {
    fn tag(&self) -> u8 {
        match self {
            Message::OpenSession { .. } => 1,
            Message::Frame { .. } => 2,
            Message::Reconstruct { .. } => 3,
            Message::Cancel { .. } => 4,
            Message::FrameAck { .. } => 128,
            Message::Progress { .. } => 129,
            Message::Cameras { .. } => 130,
            Message::DepthChunk { .. } => 131,
            Message::Done { .. } => 132,
            Message::Error { .. } => 133,
        }
    }

    /// Write one length-prefixed message.
    pub fn write_to(&self, w: &mut impl Write) -> Result<(), ProtocolError> {
        let mut p = vec![self.tag()];
        match self {
            Message::OpenSession { session_id, n_frames, width, height, draft, model } => {
                p.extend(session_id.to_le_bytes());
                p.extend(n_frames.to_le_bytes());
                p.extend(width.to_le_bytes());
                p.extend(height.to_le_bytes());
                p.push(*draft as u8);
                put_str(&mut p, model);
            }
            Message::Frame { session_id, frame_idx, rgb8 } => {
                p.extend(session_id.to_le_bytes());
                p.extend(frame_idx.to_le_bytes());
                p.extend_from_slice(rgb8);
            }
            Message::Reconstruct { session_id } | Message::Cancel { session_id } => {
                p.extend(session_id.to_le_bytes());
            }
            Message::FrameAck { frame_idx, s1_done } => {
                p.extend(frame_idx.to_le_bytes());
                p.push(*s1_done as u8);
            }
            Message::Progress { stage, done, total } => {
                p.push(*stage as u8);
                p.extend(done.to_le_bytes());
                p.extend(total.to_le_bytes());
            }
            Message::Cameras { pose_enc } => {
                p.extend((pose_enc.len() as u32).to_le_bytes());
                for v in pose_enc {
                    p.extend(v.to_le_bytes());
                }
            }
            Message::DepthChunk { first_frame_idx, n_frames, px_per_frame, depth, conf } => {
                p.extend(first_frame_idx.to_le_bytes());
                p.extend(n_frames.to_le_bytes());
                p.extend(px_per_frame.to_le_bytes());
                for v in depth {
                    p.extend(v.to_le_bytes());
                }
                for v in conf {
                    p.extend(v.to_le_bytes());
                }
            }
            Message::Done { s1_secs, s2_secs, s3_secs, s4_secs, peak_mem_bytes, model_hash } => {
                for v in [s1_secs, s2_secs, s3_secs, s4_secs] {
                    p.extend(v.to_le_bytes());
                }
                p.extend(peak_mem_bytes.to_le_bytes());
                put_str(&mut p, model_hash);
            }
            Message::Error { code, message } => {
                p.push(*code as u8);
                put_str(&mut p, message);
            }
        }
        let len = p.len() as u32;
        if len > MAX_PAYLOAD {
            return Err(ProtocolError::TooLarge(len));
        }
        w.write_all(&len.to_le_bytes())?;
        w.write_all(&p)?;
        w.flush()?;
        Ok(())
    }

    /// Read one length-prefixed message. `Ok(None)` on clean EOF at a
    /// message boundary.
    pub fn read_from(r: &mut impl Read) -> Result<Option<Self>, ProtocolError> {
        let mut len_buf = [0u8; 4];
        match r.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e.into()),
        }
        let len = u32::from_le_bytes(len_buf);
        if len > MAX_PAYLOAD || len == 0 {
            return Err(ProtocolError::TooLarge(len));
        }
        let mut p = vec![0u8; len as usize];
        r.read_exact(&mut p)?;

        let tag = p[0];
        let mut c = Cursor { buf: &p[1..], pos: 0, tag };
        let msg = match tag {
            1 => Message::OpenSession {
                session_id: c.u64()?,
                n_frames: c.u32()?,
                width: c.u32()?,
                height: c.u32()?,
                draft: c.u8()? != 0,
                model: c.string()?,
            },
            2 => Message::Frame {
                session_id: c.u64()?,
                frame_idx: c.u32()?,
                rgb8: c.rest(),
            },
            3 => Message::Reconstruct { session_id: c.u64()? },
            4 => Message::Cancel { session_id: c.u64()? },
            128 => Message::FrameAck { frame_idx: c.u32()?, s1_done: c.u8()? != 0 },
            129 => Message::Progress {
                stage: match c.u8()? {
                    1 => Stage::S1Dino,
                    2 => Stage::S2Trunk,
                    4 => Stage::S4Depth,
                    _ => return Err(ProtocolError::Malformed { tag }),
                },
                done: c.u32()?,
                total: c.u32()?,
            },
            130 => {
                let n = c.u32()? as usize;
                let mut pose_enc = Vec::with_capacity(n);
                for _ in 0..n {
                    pose_enc.push(c.f32()?);
                }
                Message::Cameras { pose_enc }
            }
            131 => {
                let first_frame_idx = c.u32()?;
                let n_frames = c.u32()?;
                let px_per_frame = c.u32()?;
                let count = n_frames as usize * px_per_frame as usize;
                let mut depth = Vec::with_capacity(count);
                for _ in 0..count {
                    depth.push(c.u16()?);
                }
                let mut conf = Vec::with_capacity(count);
                for _ in 0..count {
                    conf.push(c.u16()?);
                }
                Message::DepthChunk { first_frame_idx, n_frames, px_per_frame, depth, conf }
            }
            132 => Message::Done {
                s1_secs: c.f32()?,
                s2_secs: c.f32()?,
                s3_secs: c.f32()?,
                s4_secs: c.f32()?,
                peak_mem_bytes: c.u64()?,
                model_hash: c.string()?,
            },
            133 => Message::Error {
                code: match c.u8()? {
                    1 => ErrorCode::Protocol,
                    2 => ErrorCode::Validation,
                    3 => ErrorCode::Cancelled,
                    4 => ErrorCode::Internal,
                    _ => return Err(ProtocolError::Malformed { tag }),
                },
                message: c.string()?,
            },
            other => return Err(ProtocolError::UnknownTag(other)),
        };
        Ok(Some(msg))
    }
}

fn put_str(p: &mut Vec<u8>, s: &str) {
    p.extend((s.len() as u32).to_le_bytes());
    p.extend_from_slice(s.as_bytes());
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
    tag: u8,
}

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], ProtocolError> {
        let end = self.pos + n;
        if end > self.buf.len() {
            return Err(ProtocolError::Malformed { tag: self.tag });
        }
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, ProtocolError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, ProtocolError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, ProtocolError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, ProtocolError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, ProtocolError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, ProtocolError> {
        let len = self.u32()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| ProtocolError::Malformed { tag: self.tag })
    }
    fn rest(&mut self) -> Vec<u8> {
        let out = self.buf[self.pos..].to_vec();
        self.pos = self.buf.len();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_every_message() {
        let messages = vec![
            Message::OpenSession {
                session_id: 7,
                n_frames: 100,
                width: 624,
                height: 416,
                draft: false,
                model: "vggt-omega-1b-512".into(),
            },
            Message::Frame { session_id: 7, frame_idx: 3, rgb8: vec![1, 2, 3, 255, 0, 17] },
            Message::Reconstruct { session_id: 7 },
            Message::Cancel { session_id: 7 },
            Message::FrameAck { frame_idx: 3, s1_done: true },
            Message::Progress { stage: Stage::S2Trunk, done: 12, total: 24 },
            Message::Cameras { pose_enc: vec![0.5, -1.25, 3.0, 0.0, 0.0, 0.0, 1.0, 0.8, 0.9] },
            Message::DepthChunk {
                first_frame_idx: 8,
                n_frames: 2,
                px_per_frame: 3,
                depth: vec![0x3C00, 0x4000, 0x4200, 1, 2, 3],
                conf: vec![0x3C00; 6],
            },
            Message::Done {
                s1_secs: 1.5,
                s2_secs: 20.25,
                s3_secs: 0.01,
                s4_secs: 4.0,
                peak_mem_bytes: 37 << 30,
                model_hash: "abc123".into(),
            },
            Message::Error { code: ErrorCode::Cancelled, message: "cancelled".into() },
        ];

        // one stream carrying all of them in order
        let mut wire = Vec::new();
        for m in &messages {
            m.write_to(&mut wire).unwrap();
        }
        let mut r = wire.as_slice();
        for expected in &messages {
            let got = Message::read_from(&mut r).unwrap().expect("message");
            assert_eq!(&got, expected);
        }
        assert!(Message::read_from(&mut r).unwrap().is_none(), "clean EOF");
    }

    #[test]
    fn rejects_garbage() {
        // absurd length prefix
        let mut r: &[u8] = &[0xFF, 0xFF, 0xFF, 0xFF, 0, 0];
        assert!(matches!(Message::read_from(&mut r), Err(ProtocolError::TooLarge(_))));
        // unknown tag
        let mut wire = Vec::new();
        wire.extend(2u32.to_le_bytes());
        wire.extend([200u8, 0]);
        let mut r = wire.as_slice();
        assert!(matches!(Message::read_from(&mut r), Err(ProtocolError::UnknownTag(200))));
        // truncated payload for the tag
        let mut wire = Vec::new();
        wire.extend(3u32.to_le_bytes());
        wire.extend([3u8, 1, 2]); // Reconstruct needs 8 bytes of session_id
        let mut r = wire.as_slice();
        assert!(matches!(Message::read_from(&mut r), Err(ProtocolError::Malformed { tag: 3 })));
    }
}
