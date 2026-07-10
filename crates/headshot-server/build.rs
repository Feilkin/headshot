//! Compiles WESL shaders in `shaders/` to WGSL artifacts in OUT_DIR.
//!
//! Every kernel in `DUAL_KERNELS` is compiled twice: with the `f16` feature
//! off (`<name>.wgsl`, f32 storage) and on (`<name>_f16.wgsl`, f16 storage /
//! f32 math). Kernels in `SINGLE_KERNELS` compile once as-is.
//! Access artifacts with `include_str!(concat!(env!("OUT_DIR"), "/<name>.wgsl"))`.

use std::collections::HashMap;

const DUAL_KERNELS: &[&str] = &[
    "linear",
    "layernorm",
    "gelu",
    "residual_ls",
    "split_heads",
    "attn_merge",
    "rope_apply",
    "attention",
    "im2col_patch",
    "concat_channels",
];

// linear_wmma.wgsl is NOT compiled through WESL: wgsl-parse has no
// cooperative-matrix syntax; it ships as plain WGSL, included directly.
const SINGLE_KERNELS: &[&str] = &[
    "residual_add",
    "cast_f32_to_f16",
    "cast_f16_to_f32",
    // dense head, f32 only
    "unary",
    "tiled_add",
    "im2col3x3",
    "shuffle_expand",
    "bilinear",
];

fn compile(kernel: &str, flags: &[(&str, bool)], artifact: &str) {
    let mut wesl = wesl::Wesl::new("shaders");
    let features = wesl::Features {
        flags: flags
            .iter()
            .map(|(name, on)| (name.to_string(), wesl::Feature::from(*on)))
            .collect::<HashMap<_, _>>(),
        ..Default::default()
    };
    wesl.set_options(wesl::CompileOptions { features, ..Default::default() });
    let root = format!("package::{kernel}").parse().expect("valid module path");
    wesl.build_artifact(&root, artifact);
}

fn main() {
    for kernel in DUAL_KERNELS {
        compile(kernel, &[("f16", false), ("d128", false)], kernel);
        compile(kernel, &[("f16", true), ("d128", false)], &format!("{kernel}_f16"));
    }
    // camera-head attention: head_dim 128, f32 only (heads run all-f32)
    compile("attention", &[("f16", false), ("d128", true)], "attention_d128");
    for kernel in SINGLE_KERNELS {
        compile(kernel, &[], kernel);
    }
    println!("cargo::rerun-if-changed=shaders/linear_wmma.wgsl");
}
