//! Diagnostic: GTT usage per trunk block (is reclamation keeping up?).

use headshot_server::engine::GpuContext;
use headshot_server::engine::tensor::Dtype;
use headshot_server::model::{dino::Dino, trunk::Trunk};
use headshot_server::parity::fixtures_dir;
use headshot_server::weights::Weights;

fn gtt_gib() -> f64 {
    let mut total = 0u64;
    for card in std::fs::read_dir("/sys/class/drm").unwrap().flatten() {
        for name in ["mem_info_gtt_used", "mem_info_vram_used"] {
            let p = card.path().join("device").join(name);
            if let Ok(s) = std::fs::read_to_string(&p) {
                total += s.trim().parse::<u64>().unwrap_or(0);
            }
        }
    }
    total as f64 / (1u64 << 30) as f64
}

#[test]
#[ignore = "diagnostic; needs fixtures + GPU"]
fn trunk_memory_trace() {
    let ctx = GpuContext::new().unwrap();
    let weights = Weights::open(&fixtures_dir()).unwrap();
    // synthesize a 20-frame 512x512 input (content irrelevant)
    let n = 20;
    let (h, w) = (512, 512);
    let images: Vec<f32> = (0..n * 3 * h * w).map(|i| (i % 251) as f32 / 251.0).collect();
    let images = ctx.tensor_from_slice(&[n, 3, h, w], &images);

    println!("baseline: {:.1} GiB", gtt_gib());
    let dino = Dino::load(&ctx, &weights, Dtype::F16).unwrap();
    let trunk = Trunk::load(&ctx, &weights, Dtype::F16).unwrap();
    println!("weights loaded: {:.1} GiB", gtt_gib());

    let tokens = dino.forward(&ctx, &images, None);
    ctx.sync();
    println!("after dino sync: {:.1} GiB", gtt_gib());

    // no taps (taps download => sync => they mask reclamation behavior);
    // watch from a background sampler instead
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let peak = std::sync::Arc::new(std::sync::Mutex::new(Vec::<f64>::new()));
    let (stop2, peak2) = (stop.clone(), peak.clone());
    let sampler = std::thread::spawn(move || {
        while !stop2.load(std::sync::atomic::Ordering::Relaxed) {
            peak2.lock().unwrap().push(gtt_gib());
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    });
    let caches = trunk.forward(&ctx, &tokens, n, h / 16, w / 16, None);
    ctx.sync();
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    sampler.join().unwrap();
    drop(caches);
    let samples = peak.lock().unwrap();
    let peak_gib = samples.iter().cloned().fold(0.0f64, f64::max);
    println!("trunk (no taps): peak {peak_gib:.1} GiB over {} samples", samples.len());
    let line: Vec<String> = samples.iter().step_by(4).map(|v| format!("{v:.0}")).collect();
    println!("trace: {}", line.join(" "));
    println!("after trunk sync: {:.1} GiB", gtt_gib());
}


#[test]
#[ignore = "diagnostic; needs fixtures + GPU"]
fn full_pipeline_phase_peaks() {
    use headshot_server::model::{camera_head::CameraHead, dense_head::DenseHead};
    let ctx = GpuContext::new().unwrap();
    let weights = Weights::open(&fixtures_dir()).unwrap();
    let n = 100;
    let (h, w) = (512, 512);
    let images: Vec<f32> = (0..n * 3 * h * w).map(|i| (i % 251) as f32 / 251.0).collect();
    let images = ctx.tensor_from_slice(&[n, 3, h, w], &images);

    let peak = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let peak2 = peak.clone();
    std::thread::spawn(move || loop {
        let gib = (gtt_gib() * 1000.0) as u64;
        peak2.fetch_max(gib, std::sync::atomic::Ordering::Relaxed);
        std::thread::sleep(std::time::Duration::from_millis(50));
    });
    let phase = |name: &str| {
        let p = peak.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0;
        println!("{name}: peak {p:.1} GiB (now {:.1})", gtt_gib());
    };

    let dino = Dino::load(&ctx, &weights, Dtype::F16).unwrap();
    let trunk = Trunk::load(&ctx, &weights, Dtype::F16).unwrap();
    let camera = CameraHead::load(&ctx, &weights).unwrap();
    let dense = DenseHead::load(&ctx, &weights).unwrap();
    ctx.sync();
    std::thread::sleep(std::time::Duration::from_millis(200));
    phase("load");

    let tokens = dino.forward(&ctx, &images, None);
    ctx.sync();
    phase("dino");
    let caches = trunk.forward(&ctx, &tokens, n, h / 16, w / 16, None);
    ctx.sync();
    phase("trunk");
    let _pose = camera.forward(&ctx, &caches[3], n, None);
    phase("camera");
    let _out = dense.forward(&ctx, &caches, n, h / 16, w / 16, None);
    ctx.sync();
    phase("dense");
}


#[test]
#[ignore = "diagnostic; needs fixtures + GPU"]
fn dino_window_experiment() {
    let ctx = GpuContext::new().unwrap();
    let weights = Weights::open(&fixtures_dir()).unwrap();
    let n = 40; // big enough to show the pattern, fast enough to iterate
    let (h, w) = (512, 512);
    let images: Vec<f32> = (0..n * 3 * h * w).map(|i| (i % 251) as f32 / 251.0).collect();
    let images = ctx.tensor_from_slice(&[n, 3, h, w], &images);
    let dino = Dino::load(&ctx, &weights, Dtype::F16).unwrap();
    ctx.sync();

    let peak = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let peak2 = peak.clone();
    std::thread::spawn(move || loop {
        peak2.fetch_max((gtt_gib() * 1000.0) as u64, std::sync::atomic::Ordering::Relaxed);
        std::thread::sleep(std::time::Duration::from_millis(20));
    });

    let t = std::time::Instant::now();
    let tokens = dino.forward(&ctx, &images, None);
    ctx.sync();
    drop(tokens);
    println!(
        "max_in_flight={}: dino n={n} peak {:.1} GiB, {:.1}s",
        std::env::var("HEADSHOT_MAX_IN_FLIGHT").unwrap_or_else(|_| "default".into()),
        peak.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0,
        t.elapsed().as_secs_f64()
    );
}
