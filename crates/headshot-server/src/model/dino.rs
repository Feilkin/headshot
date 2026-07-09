//! Stage 1 — DINOv3 ViT-L/16 patch embedding, run per frame (doc/01 §2).

use anyhow::Result;

use super::block::{Block, Rope};
use super::{read_f32, upload_at, upload_f32};
use crate::engine::tensor::{Dtype, GpuTensor};
use crate::engine::{GpuContext, rope};
use crate::weights::Weights;

const PREFIX: usize = 5; // 1 cls + 4 storage tokens
const DIM: usize = 1024;

pub struct Dino {
    proj_w: GpuTensor, // (1024, 768) flattened conv weight
    proj_b: GpuTensor,
    cls_token: GpuTensor,      // (1, 1024)
    storage_tokens: GpuTensor, // (4, 1024)
    blocks: Vec<Block>,
    norm_w: GpuTensor,
    norm_b: GpuTensor,
    periods: Vec<f32>,
    precision: Dtype,
}

/// Optional observer for intermediate activations (parity tests).
pub type Tap<'a> = &'a mut dyn FnMut(&str, &GpuTensor);

impl Dino {
    /// DINO always runs f32 regardless of the requested engine precision:
    /// its residual stream carries per-frame "massive activations" of
    /// ~1.4e5 (measured on the real checkpoint) which overflow f16's 65504
    /// max — the reference survives only because bf16 keeps f32's exponent
    /// range. The final norm squashes them (output max ~2), so the trunk
    /// can still run f16; the caller casts at the stage boundary
    /// ([`crate::model::trunk::Trunk::forward`] does it automatically).
    pub fn load(ctx: &GpuContext, weights: &Weights, precision: Dtype) -> Result<Self> {
        let _ = precision;
        let precision = Dtype::F32;
        let p = "aggregator.patch_embed";
        let proj_w = upload_at(ctx, weights, &format!("{p}.patch_embed.proj.weight"), precision)?
            .reshaped(&[DIM, 768]);
        let blocks = (0..24)
            .map(|i| Block::load(ctx, weights, &format!("{p}.blocks.{i}"), 16, false, precision))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            proj_w,
            proj_b: upload_f32(ctx, weights, &format!("{p}.patch_embed.proj.bias"))?,
            cls_token: upload_at(ctx, weights, &format!("{p}.cls_token"), precision)?
                .reshaped(&[1, DIM]),
            storage_tokens: upload_at(ctx, weights, &format!("{p}.storage_tokens"), precision)?
                .reshaped(&[4, DIM]),
            blocks,
            norm_w: upload_f32(ctx, weights, &format!("{p}.norm.weight"))?,
            norm_b: upload_f32(ctx, weights, &format!("{p}.norm.bias"))?,
            periods: read_f32(weights, &format!("{p}.rope_embed.periods"))?,
            precision,
        })
    }

    /// images (N, 3, H, W) f32 in [0,1] → patch tokens (N·P, 1024).
    /// `tap` observes "dino.patch_embed" (N·P, 1024) and "dino.tokens".
    /// Bails with [`crate::engine::Cancelled`] between blocks on request.
    pub fn forward(
        &self,
        ctx: &GpuContext,
        images: &GpuTensor,
        mut tap: Option<Tap<'_>>,
    ) -> Result<GpuTensor> {
        let [n, _, height, width] = images.shape[..] else { panic!("images (N,3,H,W)") };
        let (h_p, w_p) = (height / 16, width / 16);
        let p = h_p * w_p;
        let t = p + PREFIX;

        // patch embed: fused normalize+im2col, then the conv-as-GEMM
        let cols = ctx.im2col_patch(
            images,
            headshot_shared::model::IMAGENET_MEAN,
            headshot_shared::model::IMAGENET_STD,
            self.precision,
        );
        let embedded = ctx.linear(&cols, &self.proj_w, Some(&self.proj_b));
        if let Some(tap) = tap.as_deref_mut() {
            tap("dino.patch_embed", &embedded);
        }

        // assemble (N, 5+P, 1024): cls + storage + patches per frame
        let mut x = ctx.empty_typed(&[n * t, DIM], self.precision);
        for i in 0..n {
            ctx.copy_rows(&self.cls_token, 0, &x, i * t, 1);
            ctx.copy_rows(&self.storage_tokens, 0, &x, i * t + 1, 4);
            ctx.copy_rows(&embedded, i * p, &x, i * t + PREFIX, p);
        }

        let (sin, cos) = rope::tables(&self.periods, h_p, w_p);
        let sin = ctx.tensor_from_slice(&[p, 64], &sin);
        let cos = ctx.tensor_from_slice(&[p, 64], &cos);

        for (i, block) in self.blocks.iter().enumerate() {
            ctx.check_cancelled()?;
            x = block.forward(ctx, &x, n, t, Some(Rope { sin: &sin, cos: &cos, prefix: PREFIX }));
            if let Some(tap) = tap.as_deref_mut() {
                tap(&format!("dino.block.{i:02}"), &x);
            }
        }

        let normed = ctx.layernorm(&x, &self.norm_w, &self.norm_b);

        // keep only the P patch tokens per frame
        let out = ctx.empty_typed(&[n * p, DIM], self.precision);
        for i in 0..n {
            ctx.copy_rows(&normed, i * t + PREFIX, &out, i * p, p);
        }
        if let Some(tap) = tap {
            tap("dino.tokens", &out);
        }
        ctx.flush();
        Ok(out)
    }
}
