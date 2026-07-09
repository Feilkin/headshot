//! Stage 1 — DINOv3 ViT-L/16 patch embedding, run per frame (doc/01 §2).

use anyhow::Result;

use super::block::{Block, Rope};
use super::{read_f32, upload_f32};
use crate::engine::{GpuContext, rope, tensor::GpuTensor};
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
}

/// Optional observer for intermediate activations (parity tests).
pub type Tap<'a> = &'a mut dyn FnMut(&str, &GpuTensor);

impl Dino {
    pub fn load(ctx: &GpuContext, weights: &Weights) -> Result<Self> {
        let p = "aggregator.patch_embed";
        let proj_w =
            upload_f32(ctx, weights, &format!("{p}.patch_embed.proj.weight"))?.reshaped(&[DIM, 768]);
        let blocks = (0..24)
            .map(|i| Block::load(ctx, weights, &format!("{p}.blocks.{i}"), 16, false))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            proj_w,
            proj_b: upload_f32(ctx, weights, &format!("{p}.patch_embed.proj.bias"))?,
            cls_token: upload_f32(ctx, weights, &format!("{p}.cls_token"))?.reshaped(&[1, DIM]),
            storage_tokens: upload_f32(ctx, weights, &format!("{p}.storage_tokens"))?
                .reshaped(&[4, DIM]),
            blocks,
            norm_w: upload_f32(ctx, weights, &format!("{p}.norm.weight"))?,
            norm_b: upload_f32(ctx, weights, &format!("{p}.norm.bias"))?,
            periods: read_f32(weights, &format!("{p}.rope_embed.periods"))?,
        })
    }

    /// images (N, 3, H, W) f32 in [0,1] → patch tokens (N·P, 1024).
    /// `tap` observes "dino.patch_embed" (N·P, 1024) and "dino.tokens".
    pub fn forward(
        &self,
        ctx: &GpuContext,
        images: &GpuTensor,
        mut tap: Option<Tap<'_>>,
    ) -> GpuTensor {
        let [n, _, height, width] = images.shape[..] else { panic!("images (N,3,H,W)") };
        let (h_p, w_p) = (height / 16, width / 16);
        let p = h_p * w_p;
        let t = p + PREFIX;

        // patch embed: fused normalize+im2col, then the conv-as-GEMM
        let cols = ctx.im2col_patch(
            images,
            headshot_shared::model::IMAGENET_MEAN,
            headshot_shared::model::IMAGENET_STD,
        );
        let embedded = ctx.linear(&cols, &self.proj_w, Some(&self.proj_b));
        if let Some(tap) = tap.as_deref_mut() {
            tap("dino.patch_embed", &embedded);
        }

        // assemble (N, 5+P, 1024): cls + storage + patches per frame
        let mut x = ctx.empty(&[n * t, DIM]);
        let row = (DIM * 4) as u64;
        for i in 0..n {
            let dst = (i * t) as u64 * row;
            ctx.copy(&self.cls_token, 0, &x, dst, row);
            ctx.copy(&self.storage_tokens, 0, &x, dst + row, 4 * row);
            ctx.copy(&embedded, (i * p) as u64 * row, &x, dst + 5 * row, (p as u64) * row);
        }

        let (sin, cos) = rope::tables(&self.periods, h_p, w_p);
        let sin = ctx.tensor_from_slice(&[p, 64], &sin);
        let cos = ctx.tensor_from_slice(&[p, 64], &cos);

        for block in &self.blocks {
            x = block.forward(ctx, &x, n, t, Some(Rope { sin: &sin, cos: &cos, prefix: PREFIX }));
        }

        let normed = ctx.layernorm(&x, &self.norm_w, &self.norm_b);

        // keep only the P patch tokens per frame
        let out = ctx.empty(&[n * p, DIM]);
        for i in 0..n {
            ctx.copy(
                &normed,
                ((i * t + PREFIX) * DIM * 4) as u64,
                &out,
                (i * p * DIM * 4) as u64,
                (p * DIM * 4) as u64,
            );
        }
        if let Some(tap) = tap {
            tap("dino.tokens", &out);
        }
        ctx.flush();
        out
    }
}
