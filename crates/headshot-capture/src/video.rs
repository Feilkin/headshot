//! Video decode via ffmpeg/ffprobe subprocesses (doc/05 §1): metadata
//! probe, pass-1 grayscale candidate sampling, pass-2 full-resolution
//! rgb48le keyframe extraction, embedded-SRT extraction, HEIC decode.
//! `VideoBackend` is the seam that lets the assembly pipeline run in CI on
//! synthetic frames without ffmpeg; the helpers around it are pure and
//! CI-tested. Counts always derive from pipe EOF, never from `nb_frames`.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::error::CaptureError;
use crate::keyframe::RgbFrame;

/// Stream metadata from `ffprobe -show_format -show_streams`.
#[derive(Debug, Clone)]
pub struct VideoMeta {
    pub width: u32,
    pub height: u32,
    /// `avg_frame_rate` as a ratio (e.g. 30000/1001 → 29.97…).
    pub fps: f64,
    /// `nb_frames` when present — unreliable, never used for allocation.
    pub n_frames: Option<u64>,
    pub duration_s: Option<f64>,
    /// `"tv"` / `"pc"`; `None` when untagged (DJI footage usually is —
    /// pass 2 then forces limited range).
    pub color_range: Option<String>,
    /// Warn on `"arib-std-b67"` (HLG) / `"smpte2084"` (PQ).
    pub color_transfer: Option<String>,
    /// First subtitle stream index (DJI embeds telemetry as `mov_text`).
    pub subtitle_stream: Option<u32>,
}

/// Full-resolution rgb48le frame: `width·height·3` little-endian u16s.
pub struct Rgb48Frame {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u16>,
}

/// Decode seam. `FfmpegCli` is the real implementation; tests drive the
/// pipeline with synthetic in-memory frames. `Sync` so photo scoring can
/// fan out across threads.
pub trait VideoBackend: Sync {
    fn probe(&self, path: &Path) -> Result<VideoMeta, CaptureError>;

    /// One sequential decode of every `step`-th frame, RGB8, `out_w` wide
    /// (scoring gray is derived in-crate; the color survives as the UI
    /// thumbnail). Candidate `i` ↔ source frame `i·step` (exact,
    /// deterministic).
    fn sample_thumbs(
        &self,
        path: &Path,
        meta: &VideoMeta,
        step: u32,
        out_w: u32,
    ) -> Result<Vec<RgbFrame>, CaptureError>;

    /// One sequential decode emitting exactly the sorted source frame
    /// indices `frames`, full resolution, rgb48le, streamed one frame at a
    /// time into `sink` (a full-res frame is ~24 MB — collecting a whole
    /// selection would be GBs).
    fn extract_rgb48(
        &self,
        path: &Path,
        meta: &VideoMeta,
        frames: &[u32],
        sink: &mut dyn FnMut(Rgb48Frame) -> Result<(), CaptureError>,
    ) -> Result<(), CaptureError>;

    /// Embedded telemetry subtitle track as SRT text, if the container has
    /// one and it transcodes cleanly. Sidecar `.srt` files always win over
    /// this (doc/06 §1).
    fn extract_embedded_srt(&self, path: &Path) -> Result<Option<String>, CaptureError>;

    /// Fallback decode for formats the `image` crate can't read (HEIC).
    fn decode_image_rgb8(&self, path: &Path) -> Result<image::RgbImage, CaptureError>;
}

/// Candidate sampling stride for ~2 fps (doc/05 §2 step 1).
pub fn candidate_step(fps: f64) -> u32 {
    ((fps / 2.0).round() as u32).max(1)
}

/// Body of an ffmpeg `select` expression matching exactly `frames`; the
/// caller wraps it in `select='…'` (ffmpeg-level quotes protect the
/// commas from the filtergraph parser).
pub fn select_expr(frames: &[u32]) -> String {
    let terms: Vec<String> = frames.iter().map(|n| format!("eq(n,{n})")).collect();
    terms.join("+")
}

/// Parse `ffprobe -print_format json -show_format -show_streams` output.
pub fn parse_ffprobe_json(path: &Path, json: &str) -> Result<VideoMeta, CaptureError> {
    let err = |reason: String| CaptureError::Probe { path: path.to_owned(), reason };
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| err(format!("not JSON: {e}")))?;
    let streams = v["streams"].as_array().ok_or_else(|| err("no streams array".into()))?;
    let video = streams
        .iter()
        .find(|s| s["codec_type"] == "video")
        .ok_or_else(|| err("no video stream".into()))?;

    let dim = |k: &str| {
        video[k]
            .as_u64()
            .and_then(|d| u32::try_from(d).ok())
            .filter(|&d| d > 0)
            .ok_or_else(|| err(format!("bad {k}")))
    };
    let fps_str = video["avg_frame_rate"].as_str().unwrap_or_default();
    let fps = parse_ratio(fps_str)
        .or_else(|| parse_ratio(video["r_frame_rate"].as_str().unwrap_or_default()))
        .ok_or_else(|| err(format!("bad avg_frame_rate {fps_str:?}")))?;

    Ok(VideoMeta {
        width: dim("width")?,
        height: dim("height")?,
        fps,
        n_frames: video["nb_frames"].as_str().and_then(|s| s.parse().ok()),
        duration_s: v["format"]["duration"].as_str().and_then(|s| s.parse().ok()),
        color_range: video["color_range"].as_str().map(str::to_owned),
        color_transfer: video["color_transfer"].as_str().map(str::to_owned),
        subtitle_stream: streams
            .iter()
            .find(|s| s["codec_type"] == "subtitle")
            .and_then(|s| s["index"].as_u64())
            .and_then(|i| u32::try_from(i).ok()),
    })
}

fn parse_ratio(s: &str) -> Option<f64> {
    let (num, den) = s.split_once('/')?;
    let (num, den) = (num.parse::<f64>().ok()?, den.parse::<f64>().ok()?);
    (den > 0.0 && num > 0.0).then(|| num / den)
}

/// ffmpeg/ffprobe found on PATH (or explicit binary paths).
pub struct FfmpegCli {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
}

impl Default for FfmpegCli {
    fn default() -> Self {
        Self { ffmpeg: "ffmpeg".into(), ffprobe: "ffprobe".into() }
    }
}

/// A spawned tool with its stderr draining on a side thread (a full pipe
/// would deadlock the decode).
struct Running {
    tool: &'static str,
    child: std::process::Child,
    stderr: std::thread::JoinHandle<Vec<u8>>,
}

impl Running {
    /// Wait and turn a non-zero exit into `CaptureError::Ffmpeg` with the
    /// stderr tail.
    fn finish(mut self, src: &Path) -> Result<(), CaptureError> {
        let status = self.child.wait()?;
        let stderr_buf = self.stderr.join().unwrap_or_default();
        if !status.success() {
            let text = String::from_utf8_lossy(&stderr_buf);
            let mut tail: Vec<&str> = text.lines().rev().take(8).collect();
            tail.reverse();
            return Err(CaptureError::Ffmpeg {
                tool: self.tool,
                status: status.code().unwrap_or(-1),
                path: src.to_owned(),
                stderr_tail: tail.join("\n"),
            });
        }
        Ok(())
    }

    /// Kill and reap without status interpretation (early-bail path).
    fn abort(mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = self.stderr.join();
    }
}

impl FfmpegCli {
    fn spawn(
        tool: &'static str,
        bin: &Path,
        args: &[&str],
    ) -> Result<(Running, std::process::ChildStdout), CaptureError> {
        let mut child = Command::new(bin)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    CaptureError::FfmpegMissing(e)
                } else {
                    CaptureError::Io(e)
                }
            })?;
        let mut stderr = child.stderr.take().expect("piped");
        let drain = std::thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stderr.read_to_end(&mut buf);
            buf
        });
        let stdout = child.stdout.take().expect("piped");
        Ok((Running { tool, child, stderr: drain }, stdout))
    }

    /// Run to completion with stdout collected.
    fn run(
        &self,
        tool: &'static str,
        bin: PathBuf,
        args: &[&str],
        src: &Path,
    ) -> Result<Vec<u8>, CaptureError> {
        let (running, mut stdout) = Self::spawn(tool, &bin, args)?;
        let mut out = Vec::new();
        stdout.read_to_end(&mut out)?;
        running.finish(src)?;
        Ok(out)
    }

    fn decode(&self, args: &[&str], src: &Path) -> Result<Vec<u8>, CaptureError> {
        self.run("ffmpeg", self.ffmpeg.clone(), args, src)
    }
}

/// Fill `buf` fully (`Ok(true)`), or report a clean EOF at a frame
/// boundary (`Ok(false)`); EOF mid-frame is an error.
fn read_exact_or_eof(r: &mut impl Read, buf: &mut [u8]) -> std::io::Result<bool> {
    let mut filled = 0;
    while filled < buf.len() {
        let n = r.read(&mut buf[filled..])?;
        if n == 0 {
            return if filled == 0 {
                Ok(false)
            } else {
                Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!("partial frame: {filled} of {} bytes", buf.len()),
                ))
            };
        }
        filled += n;
    }
    Ok(true)
}

/// Even-rounded proportional height for a `scale=w:h` with known source
/// dims (computed ourselves so frame byte counts are unambiguous).
fn scaled_even_height(meta: &VideoMeta, out_w: u32) -> u32 {
    let h = f64::from(meta.height) * f64::from(out_w) / f64::from(meta.width);
    (((h / 2.0).round() as u32) * 2).max(2)
}

impl VideoBackend for FfmpegCli {
    fn probe(&self, path: &Path) -> Result<VideoMeta, CaptureError> {
        let p = path.to_string_lossy();
        let args =
            ["-v", "error", "-print_format", "json", "-show_format", "-show_streams", p.as_ref()];
        let out = self.run("ffprobe", self.ffprobe.clone(), &args, path)?;
        parse_ffprobe_json(path, &String::from_utf8_lossy(&out))
    }

    fn sample_thumbs(
        &self,
        path: &Path,
        meta: &VideoMeta,
        step: u32,
        out_w: u32,
    ) -> Result<Vec<RgbFrame>, CaptureError> {
        let (w, h) = (out_w, scaled_even_height(meta, out_w));
        let vf = format!("select='not(mod(n,{step}))',scale={w}:{h}:flags=area");
        let p = path.to_string_lossy();
        #[rustfmt::skip]
        let args = [
            "-v", "error", "-nostdin", "-i", p.as_ref(), "-map", "0:v:0",
            "-vf", &vf, "-fps_mode", "vfr", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ];
        let bytes = self.decode(&args, path)?;
        let px = (w * h * 3) as usize;
        if bytes.is_empty() || !bytes.len().is_multiple_of(px) {
            return Err(CaptureError::Decode {
                path: path.to_owned(),
                reason: format!(
                    "{} bytes is not a whole number of {w}x{h} rgb24 frames",
                    bytes.len()
                ),
            });
        }
        Ok(bytes
            .chunks_exact(px)
            .map(|c| RgbFrame { width: w, height: h, data: c.to_vec() })
            .collect())
    }

    fn extract_rgb48(
        &self,
        path: &Path,
        meta: &VideoMeta,
        frames: &[u32],
        sink: &mut dyn FnMut(Rgb48Frame) -> Result<(), CaptureError>,
    ) -> Result<(), CaptureError> {
        assert!(frames.windows(2).all(|w| w[0] < w[1]), "frame list must be sorted unique");
        // untagged range: force limited (DJI records tv range; a full-range
        // guess would crush the D-Log code values the LUT expects)
        let range_unknown =
            meta.color_range.as_deref().is_none_or(|r| r.eq_ignore_ascii_case("unknown"));
        let setparams = if range_unknown { "setparams=range=tv," } else { "" };
        let vf = format!("{setparams}select='{}'", select_expr(frames));
        let p = path.to_string_lossy();
        #[rustfmt::skip]
        let args = [
            "-v", "error", "-nostdin", "-i", p.as_ref(), "-map", "0:v:0",
            "-vf", &vf, "-fps_mode", "vfr",
            "-sws_flags", "full_chroma_int+accurate_rnd+bitexact",
            "-f", "rawvideo", "-pix_fmt", "rgb48le", "pipe:1",
        ];
        let (running, mut stdout) = Self::spawn("ffmpeg", &self.ffmpeg, &args)?;
        let (w, h) = (meta.width, meta.height);
        let frame_bytes = (w as usize) * (h as usize) * 6;
        let mut buf = vec![0u8; frame_bytes];
        let mut delivered = 0usize;
        // an early bail must kill the child (finish() would block on a
        // decoder writing into a dead pipe and mask the real error)
        let stream_result: Result<(), CaptureError> = loop {
            match read_exact_or_eof(&mut stdout, &mut buf) {
                Ok(false) => break Ok(()),
                Ok(true) => {
                    delivered += 1;
                    if delivered > frames.len() {
                        break Err(CaptureError::Decode {
                            path: path.to_owned(),
                            reason: format!("more than {} frames decoded", frames.len()),
                        });
                    }
                    let frame = Rgb48Frame {
                        width: w,
                        height: h,
                        data: buf
                            .chunks_exact(2)
                            .map(|b| u16::from_le_bytes([b[0], b[1]]))
                            .collect(),
                    };
                    if let Err(e) = sink(frame) {
                        break Err(e);
                    }
                }
                Err(e) => {
                    break Err(CaptureError::Decode {
                        path: path.to_owned(),
                        reason: e.to_string(),
                    });
                }
            }
        };
        if let Err(e) = stream_result {
            running.abort();
            return Err(e);
        }
        running.finish(path)?;
        if delivered != frames.len() {
            return Err(CaptureError::Decode {
                path: path.to_owned(),
                reason: format!("expected {} rgb48 frames, decoded {delivered}", frames.len()),
            });
        }
        Ok(())
    }

    fn extract_embedded_srt(&self, path: &Path) -> Result<Option<String>, CaptureError> {
        let p = path.to_string_lossy();
        #[rustfmt::skip]
        let args = [
            "-v", "error", "-nostdin", "-i", p.as_ref(),
            "-map", "0:s:0", "-c:s", "srt", "-f", "srt", "pipe:1",
        ];
        // a data-only track or transcode failure is "no telemetry", not an
        // error — the sidecar path (or none) takes over
        match self.decode(&args, path) {
            Ok(out) if !out.is_empty() => Ok(Some(String::from_utf8_lossy(&out).into_owned())),
            Ok(_) => Ok(None),
            Err(CaptureError::Ffmpeg { .. }) => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn decode_image_rgb8(&self, path: &Path) -> Result<image::RgbImage, CaptureError> {
        let meta = self.probe(path)?;
        let p = path.to_string_lossy();
        #[rustfmt::skip]
        let args = [
            "-v", "error", "-nostdin", "-i", p.as_ref(),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ];
        let bytes = self.decode(&args, path)?;
        let expected = (meta.width * meta.height * 3) as usize;
        if bytes.len() != expected {
            return Err(CaptureError::Decode {
                path: path.to_owned(),
                reason: format!("expected {expected} rgb24 bytes, got {}", bytes.len()),
            });
        }
        image::RgbImage::from_raw(meta.width, meta.height, bytes).ok_or_else(|| {
            CaptureError::Decode { path: path.to_owned(), reason: "container mismatch".into() }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffprobe_json_mavic_fixture() {
        // hand-written representative capture; to be regenerated from a
        // real clip once local samples land (C4 samples_* tests)
        let json = include_str!("../tests/data/ffprobe/mavic2pro-2688x1512.json");
        let m = parse_ffprobe_json(Path::new("clip.mp4"), json).unwrap();
        assert_eq!((m.width, m.height), (2688, 1512));
        assert!((m.fps - 29.970_029_97).abs() < 1e-6, "{}", m.fps);
        assert_eq!(m.n_frames, Some(8991));
        assert_eq!(m.color_range.as_deref(), Some("tv"));
        assert_eq!(m.color_transfer.as_deref(), Some("bt709"));
        assert_eq!(m.subtitle_stream, Some(2));
        assert!((m.duration_s.unwrap() - 299.966333).abs() < 1e-6);
    }

    #[test]
    fn ffprobe_json_errors() {
        let bad = |json: &str| parse_ffprobe_json(Path::new("x"), json).expect_err("should fail");
        bad("not json");
        bad("{}");
        bad(r#"{"streams": []}"#);
        bad(r#"{"streams": [{"codec_type": "video", "width": 0, "height": 100, "avg_frame_rate": "25/1"}]}"#);
        bad(r#"{"streams": [{"codec_type": "video", "width": 100, "height": 100, "avg_frame_rate": "0/0"}]}"#);
    }

    #[test]
    fn candidate_step_table() {
        assert_eq!(candidate_step(29.97), 15);
        assert_eq!(candidate_step(30.0), 15);
        assert_eq!(candidate_step(25.0), 13);
        assert_eq!(candidate_step(24.0), 12);
        assert_eq!(candidate_step(60.0), 30);
        assert_eq!(candidate_step(2.0), 1);
        assert_eq!(candidate_step(0.5), 1);
    }

    #[test]
    fn select_expr_string() {
        assert_eq!(select_expr(&[12, 90, 1234]), "eq(n,12)+eq(n,90)+eq(n,1234)");
        assert_eq!(select_expr(&[0]), "eq(n,0)");
        // ~250 keyframes stays far under the argv limit
        let big: Vec<u32> = (0..250).map(|i| i * 37).collect();
        assert!(select_expr(&big).len() < 4096);
    }

    #[test]
    fn scaled_height_even_and_proportional() {
        let meta = |w, h| VideoMeta {
            width: w,
            height: h,
            fps: 30.0,
            n_frames: None,
            duration_s: None,
            color_range: None,
            color_transfer: None,
            subtitle_stream: None,
        };
        assert_eq!(scaled_even_height(&meta(2688, 1512), 480), 270);
        assert_eq!(scaled_even_height(&meta(1920, 1080), 480), 270);
        assert_eq!(scaled_even_height(&meta(3000, 2000), 480), 320);
        assert_eq!(scaled_even_height(&meta(4000, 3000), 480), 360);
        // portrait and tiny sources stay even and ≥ 2
        assert_eq!(scaled_even_height(&meta(1080, 1920), 480) % 2, 0);
        assert!(scaled_even_height(&meta(5000, 20), 480) >= 2);
    }
}
