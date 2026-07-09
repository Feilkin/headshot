//! Compiles WESL shaders in `shaders/` to WGSL artifacts in OUT_DIR.
//! Access them with `include_str!(concat!(env!("OUT_DIR"), "/<name>.wgsl"))`.

fn main() {
    let wesl = wesl::Wesl::new("shaders");
    let kernels = [
        "residual_add",
        "linear",
        "layernorm",
        "gelu",
        "residual_ls",
        "qkv_split",
        "attn_merge",
        "rope_apply",
        "attention",
        "im2col_patch",
        "concat_channels",
    ];
    for kernel in kernels {
        let root = format!("package::{kernel}")
            .parse()
            .expect("valid module path");
        wesl.build_artifact(&root, kernel);
    }
}
