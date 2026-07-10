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
    // qkv projection split per-output (q, k, v): three (H·D, C) GEMMs instead
    // of one fused (3·H·D, C), so the fused activation — which overflows
    // wgpu's 2 GiB buffer cap at high frame counts — is never materialized.
    qkv_w: [GpuTensor; 3],
    qkv_b: [GpuTensor; 3],
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

/// Copy leading-dimension rows `[start, start + count)` of `src` into a fresh
/// tensor of the same row width and dtype. Recorded into the open encoder —
/// callers flush before the source drops.
fn slice_leading(ctx: &GpuContext, src: &GpuTensor, start: usize, count: usize) -> GpuTensor {
    let cols = *src.shape.last().unwrap();
    let out = ctx.empty_typed(&[count, cols], src.dtype);
    ctx.copy_rows(src, start, &out, 0, count);
    out
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

        // Split the fused qkv projection into independent q/k/v GEMMs. The
        // checkpoint stores weight (3·H·D, C) and bias (3·H·D) as q|k|v
        // stacked along the output rows, so each third is a contiguous slice.
        let qkv_w_fused = gemm("attn.qkv.weight")?;
        let qkv_b_fused = f32t("attn.qkv.bias")?.reshaped(&[3, dim]);
        let qkv_w = std::array::from_fn(|i| slice_leading(ctx, &qkv_w_fused, i * dim, dim));
        let qkv_b = std::array::from_fn(|i| slice_leading(ctx, &qkv_b_fused, i, 1).reshaped(&[dim]));
        // The slices are recorded copies; submit them before the fused source
        // buffers drop at the end of this function.
        ctx.flush();

        Ok(Self {
            norm1_w: f32t("norm1.weight")?,
            norm1_b: f32t("norm1.bias")?,
            qkv_w,
            qkv_b,
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
        // project q, k, v independently, then reshape each to head-major
        let [q, k, v] = std::array::from_fn(|i| {
            let proj = ctx.linear(&normed, &self.qkv_w[i], Some(&self.qkv_b[i]));
            ctx.split_heads(&proj, s, t, h, d)
        });

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
        let mlp = self.mlp(ctx, &normed2);
        let out = ctx.residual_ls(&x1, &mlp, &self.ls2);

        ctx.flush();
        out
    }

    /// `fc2(gelu(fc1(x)))`. The hidden activation `(rows, 4·C)` grows with the
    /// token count and is the widest buffer in the network; at high frame
    /// counts it exceeds wgpu's `max_buffer_size` (2 GiB on many drivers). The
    /// MLP is row-independent, so stream the tokens through it in chunks small
    /// enough that every intermediate fits — numerically identical to the
    /// unchunked path. Small inputs take the single-shot path unchanged.
    fn mlp(&self, ctx: &GpuContext, normed: &GpuTensor) -> GpuTensor {
        let c = *normed.shape.last().unwrap();
        let rows = normed.len() / c;
        let hidden_dim = self.fc1_w.shape[0];
        let chunk = ctx.max_rows(hidden_dim, normed.dtype);
        if rows <= chunk {
            let hidden = ctx.linear(normed, &self.fc1_w, Some(&self.fc1_b));
            let act = ctx.gelu(&hidden);
            return ctx.linear(&act, &self.fc2_w, Some(&self.fc2_b));
        }
        let mlp = ctx.empty_typed(&[rows, c], normed.dtype);
        let mut r0 = 0;
        while r0 < rows {
            let n = chunk.min(rows - r0);
            let normed_chunk = ctx.empty_typed(&[n, c], normed.dtype);
            ctx.copy_rows(normed, r0, &normed_chunk, 0, n);
            let hidden = ctx.linear(&normed_chunk, &self.fc1_w, Some(&self.fc1_b));
            let act = ctx.gelu(&hidden);
            let mlp_chunk = ctx.linear(&act, &self.fc2_w, Some(&self.fc2_b));
            ctx.copy_rows(&mlp_chunk, 0, &mlp, r0, n);
            // Bound peak memory to one chunk's transients (with the default
            // serialized backpressure this frees them before the next chunk).
            ctx.flush();
            r0 += n;
        }
        mlp
    }
}
