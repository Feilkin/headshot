//! VGGT-Omega inference server (doc/03, doc/04): loads weights once,
//! serves reconstruction sessions over the length-prefixed TCP protocol.
//! `--probe` reports GPU capabilities and exits.

use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use headshot_server::engine::tensor::Dtype;
use headshot_server::server::{Engine, serve};
use tracing::info;

/// VGGT-Omega 1B inference server.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// Listen address.
    #[arg(long, default_value = "127.0.0.1:9276")]
    listen: String,

    /// Directory with converted model.safetensors + manifest.json.
    #[arg(long)]
    weights: Option<PathBuf>,

    /// Engine precision for DINO+trunk (heads always run f32).
    #[arg(long, default_value = "f16")]
    precision: String,

    /// Maximum frames per session (compute guard; doc/04 §6).
    #[arg(long, default_value_t = headshot_shared::protocol::DEFAULT_FRAME_CAP)]
    frame_cap: u32,

    /// Probe GPU features and exit.
    #[arg(long)]
    probe: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let args = Args::parse();

    if args.probe {
        return probe();
    }

    let precision = match args.precision.as_str() {
        "f16" => Dtype::F16,
        "f32" => Dtype::F32,
        other => anyhow::bail!("--precision must be f16 or f32, got {other}"),
    };
    let weights_dir = args
        .weights
        .unwrap_or_else(headshot_server::parity::fixtures_dir);
    let start = std::time::Instant::now();
    let engine = Arc::new(Engine::load(&weights_dir, precision, args.frame_cap)?);
    info!("engine loaded in {:.1?} (precision {precision:?})", start.elapsed());

    let listener = TcpListener::bind(&args.listen)
        .with_context(|| format!("binding {}", args.listen))?;
    serve(listener, engine)
}

fn probe() -> Result<()> {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = headshot_server::engine::pollster_block(instance.request_adapter(
        &wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        },
    ))
    .context("no usable GPU adapter")?;

    let adapter_info = adapter.get_info();
    info!(
        name = adapter_info.name,
        backend = ?adapter_info.backend,
        driver = adapter_info.driver_info,
        "adapter"
    );

    let features = adapter.features();
    for (feature, name) in [
        (wgpu::Features::SHADER_F16, "shader-f16"),
        (
            wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX,
            "cooperative-matrix (WMMA)",
        ),
        (wgpu::Features::TIMESTAMP_QUERY, "timestamp-query"),
    ] {
        let status = if features.contains(feature) {
            "available"
        } else {
            "MISSING"
        };
        info!("{name}: {status}");
    }
    for config in adapter.cooperative_matrix_properties() {
        info!(
            "cooperative-matrix config: {}x{}x{} {:?} -> {:?}",
            config.m_size, config.n_size, config.k_size, config.ab_type, config.cr_type
        );
    }
    Ok(())
}
