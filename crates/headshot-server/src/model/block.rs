//! The shared pre-norm transformer block (doc/01 §2 step 3, §3.2):
//! `x += LS1(Attn(LN(x)))` ; `x += LS2(MLP(LN(x)))`, with optional QK-norm
//! (trunk) and optional RoPE (DINO + trunk frame blocks).

use anyhow::Result;

use super::upload_f32;
use crate::engine::{GpuContext, tensor::GpuTensor};
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
    pub fn load(
        ctx: &GpuContext,
        weights: &Weights,
        prefix: &str,
        heads: usize,
        qk_norm: bool,
    ) -> Result<Self> {
        let get = |suffix: &str| upload_f32(ctx, weights, &format!("{prefix}.{suffix}"));
        let dim = weights.tensor(&format!("{prefix}.norm1.weight"))?.shape()[0];
        Ok(Self {
            norm1_w: get("norm1.weight")?,
            norm1_b: get("norm1.bias")?,
            qkv_w: get("attn.qkv.weight")?,
            qkv_b: get("attn.qkv.bias")?,
            proj_w: get("attn.proj.weight")?,
            proj_b: get("attn.proj.bias")?,
            ls1: get("ls1.gamma")?,
            norm2_w: get("norm2.weight")?,
            norm2_b: get("norm2.bias")?,
            fc1_w: get("mlp.fc1.weight")?,
            fc1_b: get("mlp.fc1.bias")?,
            fc2_w: get("mlp.fc2.weight")?,
            fc2_b: get("mlp.fc2.bias")?,
            ls2: get("ls2.gamma")?,
            qk_norm: qk_norm
                .then(|| -> Result<QkNorm> {
                    Ok(QkNorm {
                        q_w: get("attn.q_norm.weight")?,
                        q_b: get("attn.q_norm.bias")?,
                        k_w: get("attn.k_norm.weight")?,
                        k_b: get("attn.k_norm.bias")?,
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
