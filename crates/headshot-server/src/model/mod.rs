//! Model stages assembled from engine ops (doc/01).
//!
//! Every stage takes a [`Dtype`] precision: `F32` is the all-f32 debug
//! engine (parity baseline vs the f32-forced dumps); `F16` is the fast path
//! (f16 storage / f32 math, WMMA GEMMs) validated against the bf16 dumps.

pub mod block;
pub mod camera_head;
pub mod dense_head;
pub mod dino;
pub mod trunk;

use anyhow::Result;
use half::f16;

use crate::engine::tensor::{Dtype, GpuTensor};
use crate::engine::GpuContext;
use crate::weights::Weights;

/// Upload a checkpoint tensor as f32 regardless of stored dtype
/// (norm weights, biases, gammas — always-f32 parameters).
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

/// Upload a GEMM weight / token constant at the engine precision.
///
/// In F16 mode, tensors the converter kept f32 (the outlier-flagged ones)
/// are still cast to f16 here: the reference runs those same GEMMs under
/// bf16 autocast, which has *less* mantissa than f16, so this stays strictly
/// closer to the reference than the reference is to its own f32 weights.
/// The f32 flag benefits the all-f32 debug path.
pub fn upload_at(
    ctx: &GpuContext,
    weights: &Weights,
    name: &str,
    precision: Dtype,
) -> Result<GpuTensor> {
    let view = weights.tensor(name)?;
    let shape: Vec<usize> = view.shape().to_vec();
    match (precision, view.dtype()) {
        (Dtype::F16, safetensors::Dtype::F16) => {
            Ok(ctx.tensor_from_f16_bits(&shape, bytemuck::cast_slice(view.data())))
        }
        (Dtype::F16, safetensors::Dtype::F32) => {
            let data: &[f32] = bytemuck::cast_slice(view.data());
            Ok(ctx.tensor_from_f32(&shape, data, Dtype::F16))
        }
        (Dtype::F32, _) => upload_f32(ctx, weights, name),
        (_, other) => anyhow::bail!("{name}: unsupported dtype {other:?}"),
    }
}

/// Read a small f32 tensor to the CPU (rope periods etc.).
pub fn read_f32(weights: &Weights, name: &str) -> Result<Vec<f32>> {
    let view = weights.tensor(name)?;
    anyhow::ensure!(view.dtype() == safetensors::Dtype::F32, "{name}: expected f32");
    Ok(bytemuck::cast_slice(view.data()).to_vec())
}
