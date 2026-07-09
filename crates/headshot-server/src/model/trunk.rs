//! Stage 2 — aggregator trunk (doc/01 §3): 24 frame-block / inter-frame-
//! block pairs over camera+register+patch tokens; caches four concatenated
//! layer outputs for the heads.

use anyhow::Result;

use super::block::{Block, Rope};
use super::dino::Tap;
use super::{read_f32, upload_f32};
use crate::engine::{GpuContext, rope, tensor::GpuTensor};
use crate::weights::Weights;

const PREFIX: usize = 17; // 1 camera + 16 register tokens
const DIM: usize = 1024;

pub struct Trunk {
    camera_token: GpuTensor,   // (2, 1, 1024) — variant 0: frame 0, 1: rest
    register_token: GpuTensor, // (2, 16, 1024)
    frame_blocks: Vec<Block>,
    inter_blocks: Vec<Block>,
    periods: Vec<f32>,
}

impl Trunk {
    pub fn load(ctx: &GpuContext, weights: &Weights) -> Result<Self> {
        let load_blocks = |name: &str| -> Result<Vec<Block>> {
            (0..24)
                .map(|i| Block::load(ctx, weights, &format!("aggregator.{name}.{i}"), 16, true))
                .collect()
        };
        Ok(Self {
            camera_token: upload_f32(ctx, weights, "aggregator.camera_token")?,
            register_token: upload_f32(ctx, weights, "aggregator.register_token")?,
            frame_blocks: load_blocks("frame_blocks")?,
            inter_blocks: load_blocks("inter_frame_blocks")?,
            periods: read_f32(weights, "aggregator.rope_embed.periods")?,
        })
    }

    /// dino_tokens (N·P, 1024) → the four cached tensors (N·T, 2048),
    /// T = P + 17 (doc/01 §3.3).
    ///
    /// `tap` observes "trunk.frame.NN" (N·T, 1024), "trunk.inter.NN"
    /// (N·T or N·17 rows — register layers carry only the gathered prefix
    /// tokens, matching the reference hook shapes) and "cache.NN".
    pub fn forward(
        &self,
        ctx: &GpuContext,
        dino_tokens: &GpuTensor,
        n: usize,
        h_p: usize,
        w_p: usize,
        mut tap: Option<Tap<'_>>,
    ) -> Vec<GpuTensor> {
        let p = h_p * w_p;
        assert_eq!(dino_tokens.len(), n * p * DIM);
        let t = p + PREFIX;
        let row = (DIM * 4) as u64;

        // assemble (N·T, 1024): camera variant + register variant + patches
        let mut x = ctx.empty(&[n * t, DIM]);
        for i in 0..n {
            let variant = (i != 0) as usize;
            let dst = (i * t) as u64 * row;
            ctx.copy(&self.camera_token, (variant * DIM * 4) as u64, &x, dst, row);
            ctx.copy(
                &self.register_token,
                (variant * 16 * DIM * 4) as u64,
                &x,
                dst + row,
                16 * row,
            );
            ctx.copy(dino_tokens, (i * p) as u64 * row, &x, dst + 17 * row, (p as u64) * row);
        }

        let (sin, cos) = rope::tables(&self.periods, h_p, w_p);
        let sin = ctx.tensor_from_slice(&[p, 64], &sin);
        let cos = ctx.tensor_from_slice(&[p, 64], &cos);

        let mut caches = Vec::new();
        for k in 0..24 {
            let frame_out = self.frame_blocks[k].forward(
                ctx,
                &x,
                n,
                t,
                Some(Rope { sin: &sin, cos: &cos, prefix: PREFIX }),
            );
            if let Some(tap) = tap.as_deref_mut() {
                tap(&format!("trunk.frame.{k:02}"), &frame_out);
            }

            let register_layer =
                headshot_shared::model::REGISTER_ATTENTION_LAYERS.contains(&k);
            if register_layer {
                // gather the 17 prefix tokens of every frame (contiguous
                // per frame), run the block over one sequence of N·17,
                // scatter back; patch tokens pass through unchanged.
                let reg = ctx.empty(&[n * PREFIX, DIM]);
                for i in 0..n {
                    ctx.copy(&frame_out, (i * t) as u64 * row, &reg, (i * PREFIX) as u64 * row, PREFIX as u64 * row);
                }
                let reg_out = self.inter_blocks[k].forward(ctx, &reg, 1, n * PREFIX, None);
                if let Some(tap) = tap.as_deref_mut() {
                    tap(&format!("trunk.inter.{k:02}"), &reg_out);
                }
                for i in 0..n {
                    ctx.copy(&reg_out, (i * PREFIX) as u64 * row, &frame_out, (i * t) as u64 * row, PREFIX as u64 * row);
                }
                x = frame_out;
            } else {
                // global attention: one sequence of N·T tokens, no RoPE
                let inter_out = self.inter_blocks[k].forward(ctx, &frame_out, 1, n * t, None);
                if let Some(tap) = tap.as_deref_mut() {
                    tap(&format!("trunk.inter.{k:02}"), &inter_out);
                }
                if headshot_shared::model::CACHED_LAYERS.contains(&k) {
                    let cache = ctx.concat_channels(&frame_out, &inter_out);
                    if let Some(tap) = tap.as_deref_mut() {
                        tap(&format!("cache.{k:02}"), &cache);
                    }
                    caches.push(cache);
                }
                x = inter_out;
            }
            ctx.flush();
        }
        caches
    }
}
