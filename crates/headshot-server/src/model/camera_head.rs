//! Stage 3a — camera head (doc/01 §4.1): cached layer 23's prefix tokens →
//! per-frame pose encoding. All f32 (the reference's autocast boundary).

use anyhow::Result;

use super::block::Block;
use super::dino::Tap;
use super::{upload_f32};
use crate::engine::GpuContext;
use crate::engine::tensor::{Dtype, GpuTensor};
use crate::weights::Weights;

const PREFIX: usize = 17;
const DIM: usize = 2048;

pub struct CameraHead {
    token_norm_w: GpuTensor,
    token_norm_b: GpuTensor,
    trunk: Vec<Block>,
    trunk_norm_w: GpuTensor,
    trunk_norm_b: GpuTensor,
    branch0_w: GpuTensor,
    branch0_b: GpuTensor,
    branch2_w: GpuTensor,
    branch2_b: GpuTensor,
}

impl CameraHead {
    pub fn load(ctx: &GpuContext, weights: &Weights) -> Result<Self> {
        let f32t = |name: &str| upload_f32(ctx, weights, &format!("camera_head.{name}"));
        let trunk = (0..4)
            .map(|i| {
                Block::load(ctx, weights, &format!("camera_head.trunk.{i}"), 16, false, Dtype::F32)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            token_norm_w: f32t("token_norm.weight")?,
            token_norm_b: f32t("token_norm.bias")?,
            trunk,
            trunk_norm_w: f32t("trunk_norm.weight")?,
            trunk_norm_b: f32t("trunk_norm.bias")?,
            branch0_w: f32t("camera_branch.0.weight")?,
            branch0_b: f32t("camera_branch.0.bias")?,
            branch2_w: f32t("camera_branch.2.weight")?,
            branch2_b: f32t("camera_branch.2.bias")?,
        })
    }

    /// `cache23` is the layer-23 cached tensor (N·T, 2048) at any engine
    /// precision (cast to f32 here — the head's autocast boundary).
    /// Returns `pose_enc` (N, 9): t(3), quat xyzw(4), fov_h, fov_w —
    /// with the `relu(x)+0.01` FoV activation already applied.
    ///
    /// `tap` observes "camera.trunk_out" (N·17, 2048).
    pub fn forward(
        &self,
        ctx: &GpuContext,
        cache23: &GpuTensor,
        n: usize,
        mut tap: Option<Tap<'_>>,
    ) -> Vec<f32> {
        let t_full = cache23.len() / DIM / n;

        // gather the 17 prefix tokens of every frame, as f32
        let gathered = ctx.empty(&[n * PREFIX, DIM]);
        assert_eq!(cache23.dtype, Dtype::F32, "cached head inputs are f32 (doc/01 §3.3)");
        for i in 0..n {
            ctx.copy_rows(cache23, i * t_full, &gathered, i * PREFIX, PREFIX);
        }

        // one sequence over all frames' prefix tokens, no RoPE, no QK-norm
        let mut x = ctx.layernorm(&gathered, &self.token_norm_w, &self.token_norm_b);
        for block in &self.trunk {
            x = block.forward(ctx, &x, 1, n * PREFIX, None);
        }
        if let Some(tap) = tap.as_mut() {
            tap("camera.trunk_out", &x);
        }

        // each frame's camera token = position 0 of its 17-group
        let camera_tokens = ctx.empty(&[n, DIM]);
        for i in 0..n {
            ctx.copy_rows(&x, i * PREFIX, &camera_tokens, i, 1);
        }
        let normed = ctx.layernorm(&camera_tokens, &self.trunk_norm_w, &self.trunk_norm_b);
        let hidden = ctx.linear(&normed, &self.branch0_w, Some(&self.branch0_b));
        let act = ctx.gelu(&hidden);
        let raw = ctx.linear(&act, &self.branch2_w, Some(&self.branch2_b));

        // tiny (N×9): FoV activation on the CPU
        let mut pose_enc = ctx.download(&raw);
        for frame in pose_enc.chunks_mut(9) {
            for fov in &mut frame[7..9] {
                *fov = fov.max(0.0) + 0.01;
            }
        }
        pose_enc
    }
}
