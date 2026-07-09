//! Stage 3b — dense (depth) head (doc/01 §4.2): DPT-style pyramid over the
//! four cached layers' patch tokens → per-pixel depth + confidence.
//! All f32; frames processed in independent chunks of 8 (the streaming
//! unit of doc/04).
//!
//! Internal layout is NHWC (channels contiguous) so 1×1 convs are plain
//! GEMMs; the kernel-==-stride ConvTranspose is GEMM + pixel-shuffle
//! expansion; 3×3 convs are im2col + GEMM.

use anyhow::Result;

use super::dino::Tap;
use super::upload_f32;
use crate::engine::GpuContext;
use crate::engine::ops::UnaryOp;
use crate::engine::tensor::{Dtype, GpuTensor};
use crate::weights::Weights;

const PREFIX: usize = 17;
const DIM: usize = 2048;
const FEAT: usize = 256;
const OC: [usize; 4] = [256, 512, 1024, 1024];
const CHUNK: usize = headshot_shared::model::DEPTH_CHUNK_FRAMES;

pub struct DenseHead {
    norm_w: GpuTensor,
    norm_b: GpuTensor,
    projects: [(GpuTensor, GpuTensor); 4], // 1×1: (oc, 2048) + bias
    resize0: (GpuTensor, GpuTensor),       // convT 4×4s4 repacked (oc·16, ic) + tiled bias
    resize1: (GpuTensor, GpuTensor),       // convT 2×2s2 repacked (oc·4, ic) + tiled bias
    resize3: (GpuTensor, GpuTensor),       // conv 3×3 s2 (oc, 9·ic) + bias
    layer_rn: [GpuTensor; 4],              // 3×3 no-bias (256, 9·oc_j)
    refinenets: [Refinenet; 4],
    proj: (GpuTensor, GpuTensor),      // (16, 256) + bias
    proj_conf: (GpuTensor, GpuTensor), // (16, 256) + bias
}

struct Rcu {
    conv1_w: GpuTensor,
    conv1_b: GpuTensor,
    conv2_w: GpuTensor,
    conv2_b: GpuTensor,
}

struct Refinenet {
    rcu_skip: Option<Rcu>, // resConfUnit1; absent on refinenet4
    rcu_main: Rcu,         // resConfUnit2
    out_conv_w: GpuTensor, // 1×1 (256, 256)
    out_conv_b: GpuTensor,
}

/// The per-frame outputs of one forward pass.
pub struct DenseOutput {
    /// (N, H, W) row-major, depth > 0.
    pub depth: Vec<f32>,
    /// (N, H, W) row-major, confidence ≥ 1.
    pub conf: Vec<f32>,
}

impl DenseHead {
    pub fn load(ctx: &GpuContext, weights: &Weights) -> Result<Self> {
        let get = |name: &str| upload_f32(ctx, weights, &format!("dense_head.{name}"));
        let get_cpu = |name: &str| -> Result<(Vec<usize>, Vec<f32>)> {
            let view = weights.tensor(&format!("dense_head.{name}"))?;
            anyhow::ensure!(view.dtype() == safetensors::Dtype::F32, "{name}: expected f32");
            Ok((view.shape().to_vec(), bytemuck::cast_slice(view.data()).to_vec()))
        };
        // f16-stored GEMM weights also occur here; upload_f32 handles both.
        let get_any_cpu = |name: &str| -> Result<(Vec<usize>, Vec<f32>)> {
            let view = weights.tensor(&format!("dense_head.{name}"))?;
            let data: Vec<f32> = match view.dtype() {
                safetensors::Dtype::F32 => bytemuck::cast_slice(view.data()).to_vec(),
                safetensors::Dtype::F16 => {
                    bytemuck::cast_slice::<u8, half::f16>(view.data())
                        .iter()
                        .map(|v| v.to_f32())
                        .collect()
                }
                other => anyhow::bail!("{name}: unsupported dtype {other:?}"),
            };
            Ok((view.shape().to_vec(), data))
        };

        // ConvTranspose (in, out, k, k) → GEMM weight W'(j=(o·k+dy)·k+dx, in),
        // bias (out) → tiled (out·k²) matching j.
        let repack_convt = |idx: usize, k: usize| -> Result<(GpuTensor, GpuTensor)> {
            let (shape, w) = get_any_cpu(&format!("resize_layers.{idx}.weight"))?;
            let (ic, oc) = (shape[0], shape[1]);
            assert_eq!(&shape[2..], &[k, k]);
            let mut wp = vec![0.0f32; oc * k * k * ic];
            for ci in 0..ic {
                for o in 0..oc {
                    for dy in 0..k {
                        for dx in 0..k {
                            let j = (o * k + dy) * k + dx;
                            wp[j * ic + ci] = w[((ci * oc + o) * k + dy) * k + dx];
                        }
                    }
                }
            }
            let (_, b) = get_cpu(&format!("resize_layers.{idx}.bias"))?;
            let mut bp = vec![0.0f32; oc * k * k];
            for (j, slot) in bp.iter_mut().enumerate() {
                *slot = b[j / (k * k)];
            }
            Ok((
                ctx.tensor_from_slice(&[oc * k * k, ic], &wp),
                ctx.tensor_from_slice(&[oc * k * k], &bp),
            ))
        };

        let conv_weight = |name: &str| -> Result<GpuTensor> {
            // (oc, ic, kh, kw) row-major reshapes directly to (oc, ic·kh·kw)
            let (shape, w) = get_any_cpu(&format!("{name}.weight"))?;
            let oc = shape[0];
            let k: usize = shape[1..].iter().product();
            Ok(ctx.tensor_from_slice(&[oc, k], &w))
        };

        let rcu = |prefix: &str| -> Result<Rcu> {
            Ok(Rcu {
                conv1_w: conv_weight(&format!("{prefix}.conv1"))?,
                conv1_b: get(&format!("{prefix}.conv1.bias"))?,
                conv2_w: conv_weight(&format!("{prefix}.conv2"))?,
                conv2_b: get(&format!("{prefix}.conv2.bias"))?,
            })
        };

        let refinenet = |k: usize| -> Result<Refinenet> {
            let p = format!("scratch.refinenet{k}");
            Ok(Refinenet {
                rcu_skip: (k != 4).then(|| rcu(&format!("{p}.resConfUnit1"))).transpose()?,
                rcu_main: rcu(&format!("{p}.resConfUnit2"))?,
                out_conv_w: conv_weight(&format!("{p}.out_conv"))?,
                out_conv_b: get(&format!("{p}.out_conv.bias"))?,
            })
        };

        Ok(Self {
            norm_w: get("norm.weight")?,
            norm_b: get("norm.bias")?,
            projects: [
                (conv_weight("projects.0")?, get("projects.0.bias")?),
                (conv_weight("projects.1")?, get("projects.1.bias")?),
                (conv_weight("projects.2")?, get("projects.2.bias")?),
                (conv_weight("projects.3")?, get("projects.3.bias")?),
            ],
            resize0: repack_convt(0, 4)?,
            resize1: repack_convt(1, 2)?,
            resize3: (conv_weight("resize_layers.3")?, get("resize_layers.3.bias")?),
            layer_rn: [
                conv_weight("scratch.layer1_rn")?,
                conv_weight("scratch.layer2_rn")?,
                conv_weight("scratch.layer3_rn")?,
                conv_weight("scratch.layer4_rn")?,
            ],
            refinenets: [refinenet(1)?, refinenet(2)?, refinenet(3)?, refinenet(4)?],
            proj: (conv_weight("proj")?, get("proj.bias")?),
            proj_conf: (conv_weight("proj_conf")?, get("proj_conf.bias")?),
        })
    }

    /// caches: the four cached tensors (N·T, 2048) f32 (layers 4, 11, 17,
    /// 23). Returns depth/conf at full input resolution (H = 16·h_p).
    ///
    /// `tap` observes "dense.fused" (n, h4, w4, 256) NHWC per chunk.
    pub fn forward(
        &self,
        ctx: &GpuContext,
        caches: &[GpuTensor],
        n: usize,
        h_p: usize,
        w_p: usize,
        mut tap: Option<Tap<'_>>,
    ) -> DenseOutput {
        assert_eq!(caches.len(), 4);
        let p = h_p * w_p;
        let t = p + PREFIX;
        let (height, width) = (16 * h_p, 16 * w_p);
        let aspect = width as f32 / height as f32;

        // UV positional embeddings (doc/01 §4.3), ×0.1 baked in
        let uv_levels: Vec<GpuTensor> = OC
            .iter()
            .map(|&oc| ctx.tensor_from_slice(&[h_p * w_p * oc], &uv_embed(h_p, w_p, oc, aspect)))
            .collect();
        let uv_fused = ctx.tensor_from_slice(
            &[4 * h_p * 4 * w_p * FEAT],
            &uv_embed(4 * h_p, 4 * w_p, FEAT, aspect),
        );

        let mut depth = Vec::with_capacity(n * height * width);
        let mut conf = Vec::with_capacity(n * height * width);

        for chunk0 in (0..n).step_by(CHUNK) {
            let nc = CHUNK.min(n - chunk0);

            // per level: patch tokens → LN → 1×1 conv → +UV → resize
            let mut levels: Vec<GpuTensor> = Vec::with_capacity(4);
            for (j, cache) in caches.iter().enumerate() {
                assert_eq!(cache.dtype, Dtype::F32, "cached head inputs are f32");
                let tokens = ctx.empty(&[nc * p, DIM]);
                for i in 0..nc {
                    ctx.copy_rows(cache, (chunk0 + i) * t + PREFIX, &tokens, i * p, p);
                }
                let normed = ctx.layernorm(&tokens, &self.norm_w, &self.norm_b);
                let projected = ctx.linear(&normed, &self.projects[j].0, Some(&self.projects[j].1));
                let with_uv = ctx.tiled_add(&projected, &uv_levels[j]);
                let img = with_uv.reshaped(&[nc, h_p, w_p, OC[j]]);

                let resized = match j {
                    0 => {
                        let g = ctx.linear(&img, &self.resize0.0, Some(&self.resize0.1));
                        ctx.shuffle_expand(&g, nc, h_p, w_p, 4)
                    }
                    1 => {
                        let g = ctx.linear(&img, &self.resize1.0, Some(&self.resize1.1));
                        ctx.shuffle_expand(&g, nc, h_p, w_p, 2)
                    }
                    2 => img,
                    _ => {
                        let cols = ctx.im2col3x3(&img, 2);
                        let out = ctx.linear(&cols, &self.resize3.0, Some(&self.resize3.1));
                        out.reshaped(&[nc, h_p.div_ceil(2), w_p.div_ceil(2), OC[3]])
                    }
                };
                // 3×3 rn conv (no bias) → 256 channels
                let cols = ctx.im2col3x3(&resized, 1);
                let rn = ctx.linear(&cols, &self.layer_rn[j], None);
                let [_, rh, rw, _] = resized.shape[..] else { unreachable!() };
                levels.push(rn.reshaped(&[nc, rh, rw, FEAT]));
                ctx.flush();
            }

            // refinement cascade, deepest first (doc/01 §4.2 step 3):
            // out = RN4(l4) → RN3(out, l3) → RN2(out, l2) → RN1(out, l1);
            // levels[j] holds l{j+1}, so RN4 consumes levels[3] and each
            // RNk resizes to the next level's spatial size (RN1: no-op).
            let size = |t: &GpuTensor| (t.shape[1], t.shape[2]);
            let mut x = self.refine(ctx, 3, &levels[3], None, size(&levels[2]));
            x = self.refine(ctx, 2, &x, Some(&levels[2]), size(&levels[1]));
            x = self.refine(ctx, 1, &x, Some(&levels[1]), size(&levels[0]));
            let fused = self.refine(ctx, 0, &x, Some(&levels[0]), size(&levels[0]));
            if let Some(tap) = tap.as_deref_mut() {
                tap("dense.fused", &fused);
            }

            let with_uv = ctx.tiled_add(&fused, &uv_fused);
            for (proj, sink, op) in [
                (&self.proj, &mut depth, UnaryOp::Exp),
                (&self.proj_conf, &mut conf, UnaryOp::OnePlusExp),
            ] {
                let logits = ctx.linear(&with_uv, &proj.0, Some(&proj.1));
                let full = ctx.shuffle_expand(&logits, nc, 4 * h_p, 4 * w_p, 4);
                let activated = ctx.unary(&full, op);
                sink.extend(ctx.download(&activated));
            }
            ctx.flush();
        }

        DenseOutput { depth, conf }
    }

    /// One fusion block RNk (doc/01 §4.2 step 3), 0-based `idx` for
    /// refinenet{idx+1}: optional `x += RCU_a(skip)`, then RCU_b, bilinear
    /// resize (align_corners) to `(oh, ow)`, 1×1 out_conv.
    fn refine(
        &self,
        ctx: &GpuContext,
        idx: usize,
        x: &GpuTensor,
        skip: Option<&GpuTensor>,
        (oh, ow): (usize, usize),
    ) -> GpuTensor {
        let rn = &self.refinenets[idx];
        let merged;
        let x = match (skip, &rn.rcu_skip) {
            (Some(skip), Some(rcu)) => {
                let s = self.rcu(ctx, rcu, skip);
                merged = add(ctx, x, &s);
                &merged
            }
            (None, None) => x,
            _ => panic!("refinenet{}: skip/rcu mismatch", idx + 1),
        };
        let x = self.rcu(ctx, &rn.rcu_main, x);
        let [n, h, w, _] = x.shape[..] else { panic!() };
        let resized = if (h, w) == (oh, ow) { x } else { ctx.bilinear(&x, oh, ow) };
        ctx.linear(&resized, &rn.out_conv_w, Some(&rn.out_conv_b))
            .reshaped(&[n, oh, ow, FEAT])
    }

    /// `x + conv2(relu(conv1(relu(x))))`, 3×3 biased convs (doc/01 §4.2).
    fn rcu(&self, ctx: &GpuContext, rcu: &Rcu, x: &GpuTensor) -> GpuTensor {
        let [n, h, w, _] = x.shape[..] else { panic!("rcu wants NHWC") };
        let a = ctx.unary(x, UnaryOp::Relu);
        let c1 = ctx.linear(&ctx.im2col3x3(&a, 1), &rcu.conv1_w, Some(&rcu.conv1_b));
        let a2 = ctx.unary(&c1, UnaryOp::Relu);
        let a2 = a2.reshaped(&[n, h, w, FEAT]);
        let c2 = ctx.linear(&ctx.im2col3x3(&a2, 1), &rcu.conv2_w, Some(&rcu.conv2_b));
        add(ctx, x, &c2).reshaped(&[n, h, w, FEAT])
    }
}

/// Plain elementwise add via the residual_add kernel (in-place on a copy).
fn add(ctx: &GpuContext, a: &GpuTensor, b: &GpuTensor) -> GpuTensor {
    assert_eq!(a.len(), b.len());
    let out = ctx.empty(&a.shape);
    ctx.copy(a, 0, &out, 0, a.byte_len());
    ctx.dispatch(
        "residual_add",
        bytemuck::bytes_of(&(a.len() as u32)),
        &[&out, b],
        crate::engine::grid_2d(a.len() as u64, 256),
    );
    out
}

/// MoGe-style UV positional embedding (doc/01 §4.3), pre-scaled by 0.1.
/// Returns (h·w·channels) NHWC.
fn uv_embed(h: usize, w: usize, channels: usize, aspect: f32) -> Vec<f32> {
    let d = (aspect * aspect + 1.0).sqrt();
    let (span_x, span_y) = (aspect / d, 1.0 / d);
    let quarter = channels / 4;
    let omegas: Vec<f32> =
        (0..quarter).map(|k| 1.0 / 100.0f32.powf(k as f32 / quarter as f32)).collect();

    let coord = |i: usize, len: usize, span: f32| -> f32 {
        let lo = -span * (len as f32 - 1.0) / len as f32;
        let hi = span * (len as f32 - 1.0) / len as f32;
        if len == 1 { 0.0 } else { lo + (hi - lo) * i as f32 / (len as f32 - 1.0) }
    };

    let mut out = vec![0.0f32; h * w * channels];
    for y in 0..h {
        for x in 0..w {
            let u = coord(x, w, span_x);
            let v = coord(y, h, span_y);
            let base = (y * w + x) * channels;
            for (k, &omega) in omegas.iter().enumerate() {
                out[base + k] = 0.1 * (u * omega).sin();
                out[base + quarter + k] = 0.1 * (u * omega).cos();
                out[base + 2 * quarter + k] = 0.1 * (v * omega).sin();
                out[base + 3 * quarter + k] = 0.1 * (v * omega).cos();
            }
        }
    }
    out
}
