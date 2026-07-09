//! High-level ops over the WGSL kernels. Shapes follow the kernel docs:
//! token matrices are (rows, channels) row-major; attention tensors are
//! head-major (S, H, T, D).

use bytemuck::{Pod, Zeroable};

use super::{GpuContext, grid_2d, tensor::GpuTensor};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LinearParams {
    m: u32,
    k: u32,
    n: u32,
    has_bias: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LayerNormParams {
    rows: u32,
    dim: u32,
    eps: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FourU32 {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RopeParams {
    s: u32,
    t: u32,
    h: u32,
    d: u32,
    prefix: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AttentionParams {
    s: u32,
    t: u32,
    h: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Im2colParams {
    n: u32,
    height: u32,
    width: u32,
    _pad: u32,
    mean: [f32; 4],
    inv_std: [f32; 4],
}

impl GpuContext {
    /// `out (m, n) = x (m, k) @ weight (n, k)^T [+ bias (n)]`.
    pub fn linear(&self, x: &GpuTensor, weight: &GpuTensor, bias: Option<&GpuTensor>) -> GpuTensor {
        let k = *x.shape.last().unwrap();
        let m = x.len() / k;
        assert_eq!(weight.shape[1], k, "weight k mismatch");
        let n = weight.shape[0];
        if let Some(b) = bias {
            assert_eq!(b.len(), n, "bias len");
        }
        let out = self.empty(&[m, n]);
        let params = LinearParams {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            has_bias: bias.is_some() as u32,
        };
        // bias binding must exist even when unused
        let dummy;
        let bias_ref = match bias {
            Some(b) => b,
            None => {
                dummy = self.empty(&[1]);
                &dummy
            }
        };
        self.dispatch(
            "linear",
            bytemuck::bytes_of(&params),
            &[x, weight, bias_ref, &out],
            [n.div_ceil(16) as u32, m.div_ceil(16) as u32, 1],
        );
        out
    }

    /// Row-wise LayerNorm over the last dim, eps 1e-5.
    pub fn layernorm(&self, x: &GpuTensor, weight: &GpuTensor, bias: &GpuTensor) -> GpuTensor {
        let dim = *x.shape.last().unwrap();
        let rows = x.len() / dim;
        assert_eq!(weight.len(), dim);
        assert_eq!(bias.len(), dim);
        let out = self.empty(&x.shape);
        let params = LayerNormParams { rows: rows as u32, dim: dim as u32, eps: 1e-5, _pad: 0 };
        let wgs = rows as u64;
        let grid = if wgs <= 65535 {
            [wgs as u32, 1, 1]
        } else {
            [256, wgs.div_ceil(256) as u32, 1]
        };
        self.dispatch("layernorm", bytemuck::bytes_of(&params), &[x, weight, bias, &out], grid);
        out
    }

    pub fn gelu(&self, x: &GpuTensor) -> GpuTensor {
        let out = self.empty(&x.shape);
        let total = x.len() as u32;
        self.dispatch(
            "gelu",
            bytemuck::bytes_of(&total),
            &[x, &out],
            grid_2d(x.len() as u64, 256),
        );
        out
    }

    /// `out = res + gamma ⊙ x` (LayerScale + residual).
    pub fn residual_ls(&self, res: &GpuTensor, x: &GpuTensor, gamma: &GpuTensor) -> GpuTensor {
        assert_eq!(res.len(), x.len());
        let dim = gamma.len();
        assert_eq!(x.len() % dim, 0);
        let out = self.empty(&res.shape);
        let params = [x.len() as u32, dim as u32];
        self.dispatch(
            "residual_ls",
            bytemuck::cast_slice(&params),
            &[res, x, gamma, &out],
            grid_2d(x.len() as u64, 256),
        );
        out
    }

    /// (S·T, 3·H·D) → q, k, v each (S, H, T, D).
    pub fn qkv_split(
        &self,
        qkv: &GpuTensor,
        s: usize,
        t: usize,
        h: usize,
        d: usize,
    ) -> [GpuTensor; 3] {
        assert_eq!(qkv.len(), s * t * 3 * h * d);
        let shape = [s, h, t, d];
        let out = [self.empty(&shape), self.empty(&shape), self.empty(&shape)];
        let params =
            FourU32 { a: s as u32, b: t as u32, c: h as u32, d: d as u32 };
        self.dispatch(
            "qkv_split",
            bytemuck::bytes_of(&params),
            &[qkv, &out[0], &out[1], &out[2]],
            grid_2d((s * h * t * d) as u64, 256),
        );
        out
    }

    /// In-place RoPE on a head-major (S, H, T, D) tensor; the first
    /// `prefix` tokens pass through. Tables are (P, D) with P = T − prefix.
    pub fn rope_apply(&self, x: &GpuTensor, sin: &GpuTensor, cos: &GpuTensor, prefix: usize) {
        let [s, h, t, d] = x.shape[..] else { panic!("rope_apply wants (S,H,T,D)") };
        assert_eq!(sin.len(), (t - prefix) * d, "table size");
        let params = RopeParams {
            s: s as u32,
            t: t as u32,
            h: h as u32,
            d: d as u32,
            prefix: prefix as u32,
            _pad: [0; 3],
        };
        self.dispatch(
            "rope_apply",
            bytemuck::bytes_of(&params),
            &[x, sin, cos],
            grid_2d((s * h * t * (d / 2)) as u64, 256),
        );
    }

    /// Non-causal SDPA over head-major q, k, v (S, H, T, 64) → same layout.
    pub fn attention(&self, q: &GpuTensor, k: &GpuTensor, v: &GpuTensor) -> GpuTensor {
        let [s, h, t, d] = q.shape[..] else { panic!("attention wants (S,H,T,D)") };
        assert_eq!(d, 64, "attention kernel is specialized to head_dim 64");
        assert_eq!(q.shape, k.shape);
        assert_eq!(q.shape, v.shape);
        let out = self.empty(&q.shape);
        let params = AttentionParams {
            s: s as u32,
            t: t as u32,
            h: h as u32,
            scale: 1.0 / (d as f32).sqrt(),
        };
        self.dispatch(
            "attention",
            bytemuck::bytes_of(&params),
            &[q, k, v, &out],
            grid_2d((s * h * t) as u64, 64),
        );
        out
    }

    /// (S, H, T, D) → (S·T, H·D).
    pub fn attn_merge(&self, x: &GpuTensor) -> GpuTensor {
        let [s, h, t, d] = x.shape[..] else { panic!("attn_merge wants (S,H,T,D)") };
        let out = self.empty(&[s * t, h * d]);
        let params = FourU32 { a: s as u32, b: t as u32, c: h as u32, d: d as u32 };
        self.dispatch(
            "attn_merge",
            bytemuck::bytes_of(&params),
            &[x, &out],
            grid_2d(x.len() as u64, 256),
        );
        out
    }

    /// images (N, 3, H, W) in [0,1] → ImageNet-normalized patch rows
    /// (N·P, 768) matching the conv-weight layout.
    pub fn im2col_patch(&self, images: &GpuTensor, mean: [f32; 3], std: [f32; 3]) -> GpuTensor {
        let [n, c, height, width] = images.shape[..] else { panic!("images (N,3,H,W)") };
        assert_eq!(c, 3);
        assert_eq!(height % 16, 0);
        assert_eq!(width % 16, 0);
        let p = (height / 16) * (width / 16);
        let out = self.empty(&[n * p, 768]);
        let params = Im2colParams {
            n: n as u32,
            height: height as u32,
            width: width as u32,
            _pad: 0,
            mean: [mean[0], mean[1], mean[2], 0.0],
            inv_std: [1.0 / std[0], 1.0 / std[1], 1.0 / std[2], 0.0],
        };
        self.dispatch(
            "im2col_patch",
            bytemuck::bytes_of(&params),
            &[images, &out],
            grid_2d(out.len() as u64, 256),
        );
        out
    }

    /// a (rows, c1) ++ b (rows, c2) → (rows, c1+c2).
    pub fn concat_channels(&self, a: &GpuTensor, b: &GpuTensor) -> GpuTensor {
        let c1 = *a.shape.last().unwrap();
        let c2 = *b.shape.last().unwrap();
        let rows = a.len() / c1;
        assert_eq!(rows, b.len() / c2, "row mismatch");
        let out = self.empty(&[rows, c1 + c2]);
        let params = FourU32 { a: rows as u32, b: c1 as u32, c: c2 as u32, d: 0 };
        self.dispatch(
            "concat_channels",
            bytemuck::bytes_of(&params),
            &[a, b, &out],
            grid_2d(out.len() as u64, 256),
        );
        out
    }
}
