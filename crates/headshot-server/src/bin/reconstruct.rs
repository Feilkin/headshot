//! M2 CLI (doc/README): frames dir in → PLY out. Plain JPEG/PNG input with
//! the resize math of doc/05 §3; no tonemapping (that's M4a). The
//! client/server split (M3) replaces this as the user path; it stays as
//! the offline research tool.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use headshot_server::engine::GpuContext;
use headshot_server::engine::tensor::Dtype;
use headshot_server::model::{
    camera_head::CameraHead, dense_head::DenseHead, dino::Dino, trunk::Trunk,
};
use headshot_server::parity::fixtures_dir;
use headshot_server::weights::Weights;
use headshot_shared::filter::{confidence_threshold, depth_edge_mask};
use headshot_shared::ply::{PlyPoint, write_ply};
use headshot_shared::pose::Camera;
use headshot_shared::sizing;

/// Reconstruct a point cloud from a directory of photos (VGGT-Omega).
#[derive(Parser)]
#[command(version)]
struct Args {
    /// Directory of JPEG/PNG frames (sorted by filename; frame 0 is the
    /// reference camera).
    frames_dir: PathBuf,

    /// Output PLY path (a cameras JSON sidecar lands next to it).
    #[arg(short, long, default_value = "reconstruction.ply")]
    output: PathBuf,

    /// Directory with converted model.safetensors + manifest.json.
    #[arg(long)]
    weights: Option<PathBuf>,

    /// Engine precision for DINO+trunk (heads always run f32).
    #[arg(long, default_value = "f16")]
    precision: String,

    /// Confidence quantile to drop (0.3 keeps the top 70%).
    #[arg(long, default_value_t = 0.3)]
    conf_quantile: f32,

    /// 3×3 relative depth-jump threshold (doc/01 §5.3).
    #[arg(long, default_value_t = 0.03)]
    edge_threshold: f32,

    /// Cap on frame count (compute guard; quadratic in frames).
    #[arg(long, default_value_t = 64)]
    max_frames: usize,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let args = Args::parse();
    let precision = match args.precision.as_str() {
        "f16" => Dtype::F16,
        "f32" => Dtype::F32,
        other => anyhow::bail!("--precision must be f16 or f32, got {other}"),
    };

    // ---- load + preprocess frames (doc/05 §3) ----
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&args.frames_dir)
        .with_context(|| format!("reading {}", args.frames_dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            matches!(
                p.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref(),
                Some("jpg" | "jpeg" | "png")
            )
        })
        .collect();
    paths.sort();
    anyhow::ensure!(!paths.is_empty(), "no JPEG/PNG frames in {}", args.frames_dir.display());
    paths.truncate(args.max_frames);
    let n = paths.len();

    // session aspect bucket from the first frame
    let first = image::open(&paths[0]).with_context(|| paths[0].display().to_string())?;
    let (_, _, tw, th) = sizing::target_size(first.width(), first.height());
    let (width, height) = (tw as usize, th as usize);
    let session_aspect = th as f32 / tw as f32;
    tracing::info!("{n} frames → {tw}x{th} ({} tokens/frame)", (tw / 16) * (th / 16));

    let mut images = Vec::with_capacity(n * 3 * height * width);
    let mut rgb_frames: Vec<Vec<u8>> = Vec::with_capacity(n); // color source
    for (i, path) in paths.iter().enumerate() {
        let img = if i == 0 { first.clone() } else { image::open(path)? };
        let (cw, ch) = sizing::crop_to_aspect(img.width(), img.height(), session_aspect);
        let cropped = img.crop_imm((img.width() - cw) / 2, (img.height() - ch) / 2, cw, ch);
        let resized =
            image::imageops::resize(&cropped.to_rgb8(), tw, th, image::imageops::FilterType::Lanczos3);
        // (3, H, W) f32 planes in [0, 1]
        for c in 0..3 {
            images.extend(resized.pixels().map(|p| p.0[c] as f32 / 255.0));
        }
        rgb_frames.push(resized.into_raw());
    }

    // ---- inference ----
    let start = std::time::Instant::now();
    let ctx = GpuContext::new()?;
    let weights_dir = args.weights.unwrap_or_else(fixtures_dir);
    let weights = Weights::open(&weights_dir)?;
    let dino = Dino::load(&ctx, &weights, precision)?;
    let trunk = Trunk::load(&ctx, &weights, precision)?;
    let camera_head = CameraHead::load(&ctx, &weights)?;
    let dense_head = DenseHead::load(&ctx, &weights)?;
    tracing::info!("weights loaded in {:.1?}", start.elapsed());

    let (h_p, w_p) = (height / 16, width / 16);
    let images = ctx.tensor_from_slice(&[n, 3, height, width], &images);
    let t0 = std::time::Instant::now();
    let tokens = dino.forward(&ctx, &images, None);
    let caches = trunk.forward(&ctx, &tokens, n, h_p, w_p, None);
    let pose_enc = camera_head.forward(&ctx, &caches[3], n, None);
    let dense = dense_head.forward(&ctx, &caches, n, h_p, w_p, None);
    ctx.sync();
    tracing::info!("inference: {:.1?}", t0.elapsed());

    // ---- unproject + filter (doc/01 §5, doc/06 §3) ----
    let conf_gate = confidence_threshold(&dense.conf, args.conf_quantile);
    tracing::info!("confidence threshold (q={}): {conf_gate:.3}", args.conf_quantile);

    let cameras: Vec<Camera> = pose_enc
        .chunks(9)
        .map(|enc| Camera::from_pose_enc(enc, tw, th))
        .collect();

    let px_per_frame = height * width;
    let mut points = Vec::new();
    for (frame, cam) in cameras.iter().enumerate() {
        let depth = &dense.depth[frame * px_per_frame..][..px_per_frame];
        let conf = &dense.conf[frame * px_per_frame..][..px_per_frame];
        let keep = depth_edge_mask(depth, height, width, args.edge_threshold);
        let rgb = &rgb_frames[frame];
        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;
                let (d, c) = (depth[i], conf[i]);
                if !d.is_finite() || !c.is_finite() || c < conf_gate || !keep[i] {
                    continue;
                }
                points.push(PlyPoint {
                    pos: cam.unproject(x as u32, y as u32, d),
                    color: [rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]],
                    conf: c,
                    frame: frame as u16,
                });
            }
        }
    }
    tracing::info!(
        "{} points after filtering ({:.0}% of {})",
        points.len(),
        100.0 * points.len() as f32 / (n * px_per_frame) as f32,
        n * px_per_frame
    );

    write_ply(&args.output, &points)?;
    let sidecar = args.output.with_extension("cameras.json");
    let cams_json: Vec<serde_json::Value> = cameras
        .iter()
        .zip(pose_enc.chunks(9))
        .zip(&paths)
        .map(|((cam, enc), path)| {
            serde_json::json!({
                "source": path.file_name().and_then(|s| s.to_str()),
                "pose_enc": enc,
                "extrinsic_r": cam.r,
                "extrinsic_t": cam.t,
                "intrinsics": { "fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy },
                "center": cam.center(),
            })
        })
        .collect();
    std::fs::write(
        &sidecar,
        serde_json::to_string_pretty(&serde_json::json!({
            "width": tw, "height": th, "model": "vggt-omega-1b-512",
            "cameras": cams_json,
        }))?,
    )?;
    tracing::info!("wrote {} and {}", args.output.display(), sidecar.display());
    Ok(())
}
