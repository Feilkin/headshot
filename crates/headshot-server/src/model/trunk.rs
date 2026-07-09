//! Stage 2 — aggregator trunk (doc/01 §3): 24 frame-block / inter-frame-
//! block pairs over camera+register+patch tokens; caches four concatenated
//! layer outputs for the heads.

use anyhow::Result;

use super::block::{Block, Rope};
use super::dino::Tap;
use super::{read_f32, upload_at};
use crate::engine::tensor::{Dtype, GpuTensor};
use crate::engine::{GpuContext, rope};
use crate::weights::Weights;

const PREFIX: usize = 17; // 1 camera + 16 register tokens
const DIM: usize = 1024;

pub struct Trunk {
    camera_token: GpuTensor,   // (2, 1024) — variant 0: frame 0, 1: rest
    register_token: GpuTensor, // (2·16, 1024)
    frame_blocks: Vec<Block>,
    inter_blocks: Vec<Block>,
    periods: Vec<f32>,
    precision: Dtype,
}

impl Trunk {
    pub fn load(ctx: &GpuContext, weights: &Weights, precision: Dtype) -> Result<Self> {
        let load_blocks = |name: &str| -> Result<Vec<Block>> {
            (0..24)
                .map(|i| {
                    Block::load(ctx, weights, &format!("aggregator.{name}.{i}"), 16, true, precision)
                })
                .collect()
        };
        Ok(Self {
            camera_token: upload_at(ctx, weights, "aggregator.camera_token", precision)?
                .reshaped(&[2, DIM]),
            register_token: upload_at(ctx, weights, "aggregator.register_token", precision)?
                .reshaped(&[32, DIM]),
            frame_blocks: load_blocks("frame_blocks")?,
            inter_blocks: load_blocks("inter_frame_blocks")?,
            periods: read_f32(weights, "aggregator.rope_embed.periods")?,
            precision,
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
        // DINO always emits f32 (massive activations; see Dino::load) —
        // cast down at the stage boundary when the trunk runs f16.
        let cast_storage;
        let dino_tokens = if dino_tokens.dtype != self.precision {
            assert_eq!(self.precision, Dtype::F16, "unexpected dtype combination");
            cast_storage = ctx.cast_to_f16(dino_tokens);
            &cast_storage
        } else {
            dino_tokens
        };
        let t = p + PREFIX;

        // assemble (N·T, 1024): camera variant + register variant + patches
        let mut x = ctx.empty_typed(&[n * t, DIM], self.precision);
        for i in 0..n {
            let variant = (i != 0) as usize;
            ctx.copy_rows(&self.camera_token, variant, &x, i * t, 1);
            ctx.copy_rows(&self.register_token, variant * 16, &x, i * t + 1, 16);
            ctx.copy_rows(dino_tokens, i * p, &x, i * t + PREFIX, p);
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
                let reg = ctx.empty_typed(&[n * PREFIX, DIM], self.precision);
                for i in 0..n {
                    ctx.copy_rows(&frame_out, i * t, &reg, i * PREFIX, PREFIX);
                }
                let reg_out = self.inter_blocks[k].forward(ctx, &reg, 1, n * PREFIX, None);
                if let Some(tap) = tap.as_deref_mut() {
                    tap(&format!("trunk.inter.{k:02}"), &reg_out);
                }
                for i in 0..n {
                    ctx.copy_rows(&reg_out, i * PREFIX, &frame_out, i * t, PREFIX);
                }
                x = frame_out;
            } else {
                // global attention: one sequence of N·T tokens, no RoPE
                let inter_out = self.inter_blocks[k].forward(ctx, &frame_out, 1, n * t, None);
                if let Some(tap) = tap.as_deref_mut() {
                    tap(&format!("trunk.inter.{k:02}"), &inter_out);
                }
                if headshot_shared::model::CACHED_LAYERS.contains(&k) {
                    // cached head inputs are f32 (doc/01 §3.3) — the heads'
                    // autocast boundary
                    let cache = ctx.concat_channels(&frame_out, &inter_out);
                    let cache = if cache.dtype == Dtype::F16 {
                        ctx.cast_to_f32(&cache)
                    } else {
                        cache
                    };
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
