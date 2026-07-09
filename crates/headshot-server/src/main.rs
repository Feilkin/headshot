//! VGGT-Omega inference server (doc/03, doc/04).
//!
//! M0 skeleton: probes the GPU and reports whether the features the engine
//! needs (shader-f16, cooperative matrix) are available, then exits. The
//! session protocol lands in M3.

use anyhow::{Context, Result};
use clap::Parser;
use tracing::info;

/// VGGT-Omega 1B inference server.
#[derive(Parser)]
#[command(version)]
struct Args {}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    Args::parse();

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
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

    Ok(())
}
