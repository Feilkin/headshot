//! Measures the trunk's own noise amplification: run the engine twice, the
//! second time with ~1e-6 relative input perturbation, and compare layers.
//! Diagnostic (#[ignore]); explains what f32-vs-f32 parity drift is
//! attributable to reduction-order noise.

use headshot_server::engine::GpuContext;
use headshot_server::engine::tensor::Dtype;
use headshot_server::model::{dino::Dino, trunk::Trunk};
use headshot_server::parity::{Dump, compare, fixtures_dir};
use headshot_server::weights::Weights;
use std::collections::BTreeMap;

#[test]
#[ignore = "diagnostic; needs local fixtures + GPU"]
fn trunk_noise_amplification() {
    let ctx = GpuContext::new().expect("GPU");
    let weights = Weights::open(&fixtures_dir()).expect("weights");
    let scene = std::env::var("HEADSHOT_SENSITIVITY_SCENE").unwrap_or_else(|_| "small".into());
    let dump = Dump::open(&scene, "f32").expect("dump");
    let (shape, images) = dump.tensor("images").unwrap();
    let [n, _, height, width] = shape[..] else { panic!() };

    let dino = Dino::load(&ctx, &weights, Dtype::F32).unwrap();
    let trunk = Trunk::load(&ctx, &weights, Dtype::F32).unwrap();

    let run = |imgs: &[f32]| -> BTreeMap<String, Vec<f32>> {
        let mut outs = BTreeMap::new();
        let g = ctx.tensor_from_slice(&shape, imgs);
        let mut tap = |name: &str, t: &headshot_server::engine::tensor::GpuTensor| {
            outs.insert(name.to_string(), ctx.download(t));
        };
        let tokens = dino.forward(&ctx, &g, Some(&mut tap)).unwrap();
        let mut tap = |name: &str, t: &headshot_server::engine::tensor::GpuTensor| {
            outs.insert(name.to_string(), ctx.download(t));
        };
        trunk.forward(&ctx, &tokens, n, height / 16, width / 16, Some(&mut tap)).unwrap();
        outs
    };

    let base = run(&images);
    // ~1e-6 relative perturbation, deterministic
    let perturbed_imgs: Vec<f32> = images
        .iter()
        .enumerate()
        .map(|(i, v)| v + 1e-6 * ((i as f32 * 0.7).sin()))
        .collect();
    let perturbed = run(&perturbed_imgs);

    for (name, a) in &base {
        let m = compare(a, &perturbed[name]);
        println!(
            "{name:>16}: rel_max {:.3e}  cosine 1-{:.3e}",
            m.rel_max_err(),
            1.0 - m.cosine_sim
        );
    }
}
