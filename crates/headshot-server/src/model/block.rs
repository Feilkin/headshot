//! The shared pre-norm transformer block (doc/01 §2 step 3, §3.2):
//! `x += LS1(Attn(LN(x)))` ; `x += LS2(MLP(LN(x)))`, with optional QK-norm
//! (trunk) and optional RoPE (DINO + trunk frame blocks).

use anyhow::Result;

use super::{upload_at, upload_f32};
use crate::engine::tensor::{Dtype, GpuTensor};
use crate::engine::GpuContext;
use crate::weights::Weights;

pub struct Block {
    norm1_w: GpuTensor,
    norm1_b: GpuTensor,
    qkv_w: GpuTensor,
    qkv_b: GpuTensor,
    proj_w: GpuTensor,
    proj_b: GpuTensor,
    ls1: GpuTensor,
    norm2_w: GpuTensor,
    norm2_b: GpuTensor,
    fc1_w: GpuTensor,
    fc1_b: GpuTensor,
    fc2_w: GpuTensor,
    fc2_b: GpuTensor,
    ls2: GpuTensor,
    qk_norm: Option<QkNorm>,
    heads: usize,
    head_dim: usize,
}

struct QkNorm {
    q_w: GpuTensor,
    q_b: GpuTensor,
    k_w: GpuTensor,
    k_b: GpuTensor,
}

/// RoPE tables + unrotated prefix length for one attention call.
pub struct Rope<'a> {
    pub sin: &'a GpuTensor,
    pub cos: &'a GpuTensor,
    pub prefix: usize,
}

impl Block {
    /// Load `{prefix}.{norm1,attn.qkv,...}` from the converted checkpoint.
    /// GEMM weights go to `precision`; norms, biases, gammas stay f32.
    pub fn load(
        ctx: &GpuContext,
        weights: &Weights,
        prefix: &str,
        heads: usize,
        qk_norm: bool,
        precision: Dtype,
    ) -> Result<Self> {
        let f32t = |suffix: &str| upload_f32(ctx, weights, &format!("{prefix}.{suffix}"));
        let gemm = |suffix: &str| upload_at(ctx, weights, &format!("{prefix}.{suffix}"), precision);
        let dim = weights.tensor(&format!("{prefix}.norm1.weight"))?.shape()[0];
        Ok(Self {
            norm1_w: f32t("norm1.weight")?,
            norm1_b: f32t("norm1.bias")?,
            qkv_w: gemm("attn.qkv.weight")?,
            qkv_b: f32t("attn.qkv.bias")?,
            proj_w: gemm("attn.proj.weight")?,
            proj_b: f32t("attn.proj.bias")?,
            ls1: f32t("ls1.gamma")?,
            norm2_w: f32t("norm2.weight")?,
            norm2_b: f32t("norm2.bias")?,
            fc1_w: gemm("mlp.fc1.weight")?,
            fc1_b: f32t("mlp.fc1.bias")?,
            fc2_w: gemm("mlp.fc2.weight")?,
            fc2_b: f32t("mlp.fc2.bias")?,
            ls2: f32t("ls2.gamma")?,
            qk_norm: qk_norm
                .then(|| -> Result<QkNorm> {
                    Ok(QkNorm {
                        q_w: f32t("attn.q_norm.weight")?,
                        q_b: f32t("attn.q_norm.bias")?,
                        k_w: f32t("attn.k_norm.weight")?,
                        k_b: f32t("attn.k_norm.bias")?,
                    })
                })
                .transpose()?,
            heads,
            head_dim: dim / heads,
        })
    }

    /// `x` is (S·T, C); attention runs as S sequences of length T.
    /// Returns the block output, same shape. Flushes at the end so buffer
    /// lifetimes stay bounded per block.
    pub fn forward(
        &self,
        ctx: &GpuContext,
        x: &GpuTensor,
        s: usize,
        t: usize,
        rope: Option<Rope<'_>>,
    ) -> GpuTensor {
        let (h, d) = (self.heads, self.head_dim);

        let normed = ctx.layernorm(x, &self.norm1_w, &self.norm1_b);
        let qkv = ctx.linear(&normed, &self.qkv_w, Some(&self.qkv_b));
        let [q, k, v] = ctx.qkv_split(&qkv, s, t, h, d);

        // QK-norm (LayerNorm over head_dim), before RoPE (doc/01 §3.2)
        let (q, k) = match &self.qk_norm {
            Some(n) => (
                ctx.layernorm(&q, &n.q_w, &n.q_b),
                ctx.layernorm(&k, &n.k_w, &n.k_b),
            ),
            None => (q, k),
        };
        if let Some(rope) = rope {
            ctx.rope_apply(&q, rope.sin, rope.cos, rope.prefix);
            ctx.rope_apply(&k, rope.sin, rope.cos, rope.prefix);
        }

        let attn = ctx.attention(&q, &k, &v);
        let merged = ctx.attn_merge(&attn);
        let proj = ctx.linear(&merged, &self.proj_w, Some(&self.proj_b));
        let x1 = ctx.residual_ls(x, &proj, &self.ls1);

        let normed2 = ctx.layernorm(&x1, &self.norm2_w, &self.norm2_b);
        let hidden = ctx.linear(&normed2, &self.fc1_w, Some(&self.fc1_b));
        let act = ctx.gelu(&hidden);
        let mlp = ctx.linear(&act, &self.fc2_w, Some(&self.fc2_b));
        let out = ctx.residual_ls(&x1, &mlp, &self.ls2);

        ctx.flush();
        out
    }
}
