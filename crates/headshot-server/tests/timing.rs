//! Rough wall-clock of the DINO+trunk forward per precision (no taps).
//! Diagnostic (#[ignore]); run with --no-capture to see the numbers.

use headshot_server::engine::GpuContext;
use headshot_server::engine::tensor::Dtype;
use headshot_server::model::{dino::Dino, trunk::Trunk};
use headshot_server::parity::{Dump, fixtures_dir};
use headshot_server::weights::Weights;

#[test]
#[ignore = "diagnostic; needs local fixtures + GPU"]
fn trunk_wallclock() {
    let ctx = GpuContext::new().unwrap();
    let weights = Weights::open(&fixtures_dir()).unwrap();
    let scene = std::env::var("HEADSHOT_TIMING_SCENE").unwrap_or_else(|_| "realistic".into());
    let dump = Dump::open(&scene, "f32").unwrap();
    let (shape, images) = dump.tensor("images").unwrap();
    let [n, _, height, width] = shape[..] else { panic!() };
    let images = ctx.tensor_from_slice(&shape, &images);

    for precision in [Dtype::F32, Dtype::F16] {
        let dino = Dino::load(&ctx, &weights, precision).unwrap();
        let trunk = Trunk::load(&ctx, &weights, precision).unwrap();
        // warm-up (pipeline compilation, allocator)
        let tokens = dino.forward(&ctx, &images, None).unwrap();
        let _ = trunk.forward(&ctx, &tokens, n, height / 16, width / 16, None).unwrap();
        ctx.sync();

        let start = std::time::Instant::now();
        let tokens = dino.forward(&ctx, &images, None).unwrap();
        let dino_done = std::time::Instant::now();
        let caches = trunk.forward(&ctx, &tokens, n, height / 16, width / 16, None).unwrap();
        ctx.sync();
        drop(caches);
        let end = std::time::Instant::now();
        println!(
            "{scene} {precision:?}: dino {:.2}s + trunk {:.2}s = {:.2}s",
            (dino_done - start).as_secs_f64(),
            (end - dino_done).as_secs_f64(),
            (end - start).as_secs_f64()
        );
    }
}
