//! Model stages assembled from engine ops (doc/01).

pub mod block;
pub mod dino;
pub mod trunk;

use anyhow::Result;
use half::f16;

use crate::engine::{GpuContext, tensor::GpuTensor};
use crate::weights::Weights;

/// Upload a checkpoint tensor as f32, converting stored f16 on the way
/// (the all-f32 engine; the f16 fast path will upload natively).
pub fn upload_f32(ctx: &GpuContext, weights: &Weights, name: &str) -> Result<GpuTensor> {
    let view = weights.tensor(name)?;
    let shape: Vec<usize> = view.shape().to_vec();
    let data: Vec<f32> = match view.dtype() {
        safetensors::Dtype::F32 => bytemuck::cast_slice(view.data()).to_vec(),
        safetensors::Dtype::F16 => bytemuck::cast_slice::<u8, f16>(view.data())
            .iter()
            .map(|v| v.to_f32())
            .collect(),
        other => anyhow::bail!("{name}: unsupported dtype {other:?}"),
    };
    Ok(ctx.tensor_from_slice(&shape, &data))
}

/// Read a small f32 tensor to the CPU (rope periods etc.).
pub fn read_f32(weights: &Weights, name: &str) -> Result<Vec<f32>> {
    let view = weights.tensor(name)?;
    anyhow::ensure!(view.dtype() == safetensors::Dtype::F32, "{name}: expected f32");
    Ok(bytemuck::cast_slice(view.data()).to_vec())
}
