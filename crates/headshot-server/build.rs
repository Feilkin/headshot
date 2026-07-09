//! Compiles WESL shaders in `shaders/` to WGSL artifacts in OUT_DIR.
//! Access them with `include_str!(concat!(env!("OUT_DIR"), "/<name>.wgsl"))`.

fn main() {
    let wesl = wesl::Wesl::new("shaders");
    let kernels = ["residual_add"];
    for kernel in kernels {
        let root = format!("package::{kernel}")
            .parse()
            .expect("valid module path");
        wesl.build_artifact(&root, kernel);
    }
}
