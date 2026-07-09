//! Parity suite (doc/02 §5) — needs locally generated fixtures:
//!
//! ```sh
//! tools/.venv/bin/python tools/convert_weights.py <ckpt.pt> -o fixtures
//! tools/.venv/bin/python tools/dump_reference.py <ckpt.pt> -o fixtures/dumps
//! just parity
//! ```
//!
//! All tests here are #[ignore]d so plain CI (no checkpoint) stays green.
//! `HEADSHOT_FIXTURES_DIR` overrides the fixture location.

use headshot_server::engine::GpuContext;
use headshot_server::engine::tensor::Dtype;
use headshot_server::model::{
    camera_head::CameraHead, dense_head::DenseHead, dino::Dino, trunk::Trunk,
};
use headshot_server::parity::{Dump, Gate, compare, fixtures_dir};
use headshot_server::weights::Weights;

/// M0 gate: the converted checkpoint loads, matches the expected tree and
/// the manifest, and its checksum round-trips.
#[test]
#[ignore = "needs locally converted weights (just parity)"]
fn parity_weights_load_and_validate() {
    let weights = Weights::open(&fixtures_dir()).expect("converted weights load + validate");
    assert_eq!(weights.manifest.tensors.len(), 1334);

    // Spot-check one tensor view end to end.
    let periods = weights.tensor("aggregator.rope_embed.periods").unwrap();
    assert_eq!(periods.shape(), &[16]);
    let values: &[f32] = bytemuck::cast_slice(periods.data());
    assert!(values.iter().all(|v| v.is_finite() && *v > 0.0));
}

/// The reference dumps exist, parse, and are self-consistent.
#[test]
#[ignore = "needs locally generated reference dumps (just parity)"]
fn parity_dumps_are_consistent() {
    for scene in ["small", "realistic"] {
        for variant in ["bf16", "f32"] {
            let dump = Dump::open(scene, variant)
                .unwrap_or_else(|e| panic!("dump {scene}/{variant}: {e}"));
            assert_eq!(dump.meta.scene, scene);
            assert_eq!(dump.meta.variant, variant);

            let (shape, images) = dump.tensor("images").unwrap();
            assert_eq!(
                shape,
                vec![
                    dump.meta.n_frames,
                    3,
                    dump.meta.height as usize,
                    dump.meta.width as usize
                ]
            );
            assert!(images.iter().all(|v| (0.0..=1.0).contains(v)));

            // Every advertised tensor is present with its advertised shape.
            for (name, shape) in &dump.meta.shapes {
                let (actual, _) = dump.tensor(name).unwrap();
                assert_eq!(&actual, shape, "{scene}/{variant}: {name}");
            }
        }
    }
}

/// M1 gate (doc/02 §5): the engine at `precision` vs the matching
/// reference dump ("f32"-forced for the all-f32 debug engine with the
/// measured-relative gates; "bf16" autocast for the f16 engine with the
/// cosine-only cliff gates).
fn trunk_parity(scene: &str, precision: Dtype) {
    let ctx = GpuContext::new().expect("GPU required for parity");
    let weights = Weights::open(&fixtures_dir()).expect("converted weights");
    let (ref_variant, shallow_gate, deep_gate) = match precision {
        Dtype::F32 => ("f32", Gate::F32_TIGHT, Gate::F32_DEEP),
        Dtype::F16 => ("bf16", Gate::F16_TRUNK, Gate::F16_TRUNK),
    };
    let dump = Dump::open(scene, ref_variant).expect("reference dump");
    // The f16 gate self-calibrates per layer: our divergence from the bf16
    // dump must stay within 2x the reference's own bf16-vs-f32 divergence
    // (floored at the static 1e-3) — measured, the two are the same size
    // at deep layers on the realistic scene (both ~1.7e-3 at layer 23),
    // because both are precision noise amplified by the trunk's gain.
    let other_dump =
        (precision == Dtype::F16).then(|| Dump::open(scene, "f32").expect("f32 dump"));

    let (shape, images) = dump.tensor("images").unwrap();
    let [n, _, height, width] = shape[..] else { panic!("images shape") };
    let (h_p, w_p) = (height / 16, width / 16);
    let images = ctx.tensor_from_slice(&shape, &images);

    let mut failures: Vec<String> = Vec::new();
    let mut check = |name: &str, ours: Vec<f32>, gate: Gate| {
        // some taps (per-DINO-block) have no reference tensor — NaN screen only
        if !dump.meta.shapes.contains_key(name) {
            let nan = ours.iter().filter(|v| !v.is_finite()).count();
            if nan > 0 {
                failures.push(format!("{name}: {nan}/{} non-finite values", ours.len()));
            }
            return;
        }
        let (ref_shape, reference) = dump.tensor(name).unwrap();
        assert_eq!(
            ours.len(),
            ref_shape.iter().product::<usize>(),
            "{name}: element count vs dump {ref_shape:?}"
        );
        let m = compare(&ours, &reference);
        let mut gate = gate;
        if let Some(other) = &other_dump
            && let Ok((_, f32_ref)) = other.tensor(name)
        {
            let self_drift = 1.0 - compare(&reference, &f32_ref).cosine_sim;
            let floor = gate.min_cosine.map_or(0.999, |c| 1.0 - c);
            gate.min_cosine = Some(1.0 - floor.max(2.0 * self_drift));
        }
        println!(
            "{name:>16}: rel_max {:.3e}  max_abs {:.3e}  ref_scale {:8.1}  cosine 1-{:.3e}  (gate 1-{:.1e})",
            m.rel_max_err(),
            m.max_abs_err,
            m.ref_scale,
            1.0 - m.cosine_sim,
            gate.min_cosine.map_or(f64::NAN, |c| 1.0 - c),
        );
        if let Err(e) = gate.check(name, m) {
            failures.push(e);
        }
    };

    let dino = Dino::load(&ctx, &weights, precision).expect("dino load");
    let mut tap = |name: &str, t: &headshot_server::engine::tensor::GpuTensor| {
        let gate = if name.starts_with("dino.patch_embed") { shallow_gate } else { deep_gate };
        check(name, ctx.download(t), gate);
    };
    let tokens = dino.forward(&ctx, &images, Some(&mut tap));

    let trunk = Trunk::load(&ctx, &weights, precision).expect("trunk load");
    let mut tap = |name: &str, t: &headshot_server::engine::tensor::GpuTensor| {
        check(name, ctx.download(t), deep_gate);
    };
    let caches = trunk.forward(&ctx, &tokens, n, h_p, w_p, Some(&mut tap));
    assert_eq!(caches.len(), 4);

    // Camera head (all f32; doc/01 §4.1). pose_enc gate: abs < 1e-3 per
    // component (doc/02 §5).
    let camera = CameraHead::load(&ctx, &weights).expect("camera head load");
    let mut tap = |name: &str, t: &headshot_server::engine::tensor::GpuTensor| {
        check(name, ctx.download(t), deep_gate);
    };
    let pose_enc = camera.forward(&ctx, &caches[3], n, Some(&mut tap));
    // End-to-end pose bound is drift-aware: the trunk legitimately diverges
    // (chaotic amplification, see sensitivity.rs), and that drift lands in
    // pose_enc. For the f16 engine the bound self-calibrates against the
    // reference's own bf16-vs-f32 pose sensitivity (worst 1.9e-2 on the
    // realistic scene — larger than our deviation). The tight doc/02 1e-3
    // gate applies to the isolated head test below, which feeds the
    // *reference* cache.
    let pose_tol = match &other_dump {
        Some(other) => {
            let (_, a) = dump.tensor("pose_enc").unwrap();
            let (_, b) = other.tensor("pose_enc").unwrap();
            let self_sens = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            (2.0 * self_sens).max(1e-2)
        }
        None => 1e-2,
    };
    check_pose(&dump, &pose_enc, pose_tol, &mut failures);

    // Dense head end to end (our caches). Depth carries the trunk drift;
    // gate on cosine like the trunk layers, self-calibrated for f16.
    let dense = DenseHead::load(&ctx, &weights).expect("dense head load");
    let out = dense.forward(&ctx, &caches, n, h_p, w_p, None);
    let (_, depth_ref) = dump.tensor("depth").unwrap();
    let m = compare(&out.depth, &depth_ref);
    let mut depth_gate = deep_gate;
    if let Some(other) = &other_dump {
        let (_, f32_depth) = other.tensor("depth").unwrap();
        let self_drift = 1.0 - compare(&depth_ref, &f32_depth).cosine_sim;
        let floor = depth_gate.min_cosine.map_or(0.999, |c| 1.0 - c);
        depth_gate.min_cosine = Some(1.0 - floor.max(2.0 * self_drift));
    }
    println!(
        "           depth: rel_max {:.3e}  cosine 1-{:.3e}  (gate 1-{:.1e})",
        m.rel_max_err(),
        1.0 - m.cosine_sim,
        depth_gate.min_cosine.map_or(f64::NAN, |c| 1.0 - c),
    );
    if let Err(e) = depth_gate.check("depth (end-to-end)", m) {
        failures.push(e);
    }

    assert!(failures.is_empty(), "parity gate failures:\n{}", failures.join("\n"));
}

#[test]
#[ignore = "needs local fixtures + GPU (just parity)"]
fn parity_trunk_f32_small() {
    trunk_parity("small", Dtype::F32);
}

#[test]
#[ignore = "needs local fixtures + GPU (just parity)"]
fn parity_trunk_f32_realistic() {
    trunk_parity("realistic", Dtype::F32);
}

#[test]
#[ignore = "needs local fixtures + GPU (just parity)"]
fn parity_trunk_f16_small() {
    trunk_parity("small", Dtype::F16);
}

#[test]
#[ignore = "needs local fixtures + GPU (just parity)"]
fn parity_trunk_f16_realistic() {
    trunk_parity("realistic", Dtype::F16);
}


fn check_pose(dump: &Dump, pose_enc: &[f32], tol: f32, failures: &mut Vec<String>) {
    let (_, ref_pose) = dump.tensor("pose_enc").unwrap();
    assert_eq!(pose_enc.len(), ref_pose.len());
    let mut worst = 0.0f32;
    for (i, (ours, want)) in pose_enc.iter().zip(&ref_pose).enumerate() {
        let (frame, comp) = (i / 9, i % 9);
        let err = (ours - want).abs();
        worst = worst.max(err);
        if err > tol {
            failures.push(format!(
                "pose_enc[{frame}][{comp}]: {ours} vs {want}, abs err {err:.3e} > {tol:.0e}"
            ));
        }
    }
    println!("        pose_enc: worst abs err {worst:.3e} (gate {tol:.0e})");
}

/// Head parity in isolation (doc/02 §5 gates): the *reference* cached
/// tensors from the dump drive our head, decoupling head correctness from
/// trunk drift.
fn heads_isolated(scene: &str) {
    let ctx = GpuContext::new().expect("GPU required for parity");
    let weights = Weights::open(&fixtures_dir()).expect("converted weights");
    let dump = Dump::open(scene, "f32").expect("f32 dump");
    let n = dump.meta.n_frames;

    let load_cache = |name: &str| {
        let (shape, data) = dump.tensor(name).unwrap();
        let rows: usize = shape[..shape.len() - 1].iter().product();
        ctx.tensor_from_slice(&[rows, 2048], &data)
    };
    let caches: Vec<_> =
        ["cache.04", "cache.11", "cache.17", "cache.23"].map(load_cache).into();

    let camera = CameraHead::load(&ctx, &weights).expect("camera head load");
    let mut failures = Vec::new();
    let pose_enc = camera.forward(&ctx, &caches[3], n, None);
    check_pose(&dump, &pose_enc, 1e-3, &mut failures);

    // Dense head on the reference caches (doc/02 §5: depth rel < 1e-3 at
    // f32). dense.fused in the dump is NCHW; ours is NHWC — permute.
    let (h_p, w_p) = (dump.meta.height as usize / 16, dump.meta.width as usize / 16);
    let dense = DenseHead::load(&ctx, &weights).expect("dense head load");
    let gate = Gate { rel_max: Some(1e-3), min_cosine: Some(1.0 - 1e-6) };
    let mut fused_ours: Vec<f32> = Vec::new();
    let mut tap = |name: &str, t: &headshot_server::engine::tensor::GpuTensor| {
        if name == "dense.fused" {
            fused_ours.extend(ctx.download(t));
        }
    };
    let out = dense.forward(&ctx, &caches, n, h_p, w_p, Some(&mut tap));

    let (fshape, fused_ref) = dump.tensor("dense.fused").unwrap();
    let [fn_, fc, fh, fw] = fshape[..] else { panic!("fused shape") };
    let mut fused_ref_nhwc = vec![0.0f32; fused_ref.len()];
    for ni in 0..fn_ {
        for c in 0..fc {
            for y in 0..fh {
                for x in 0..fw {
                    fused_ref_nhwc[((ni * fh + y) * fw + x) * fc + c] =
                        fused_ref[((ni * fc + c) * fh + y) * fw + x];
                }
            }
        }
    }
    for (name, ours, reference) in [
        ("dense.fused", &fused_ours, &fused_ref_nhwc),
        ("depth", &out.depth, &dump.tensor("depth").unwrap().1),
        ("depth_conf", &out.conf, &dump.tensor("depth_conf").unwrap().1),
    ] {
        let m = compare(ours, reference);
        println!(
            "{name:>12}: rel_max {:.3e}  ref_scale {:8.2}  cosine 1-{:.3e}",
            m.rel_max_err(),
            m.ref_scale,
            1.0 - m.cosine_sim
        );
        if let Err(e) = gate.check(name, m) {
            failures.push(e);
        }
    }

    assert!(failures.is_empty(), "isolated-head failures:\n{}", failures.join("\n"));
}

#[test]
#[ignore = "needs local fixtures + GPU (just parity)"]
fn parity_heads_isolated_small() {
    heads_isolated("small");
}

#[test]
#[ignore = "needs local fixtures + GPU (just parity)"]
fn parity_heads_isolated_realistic() {
    heads_isolated("realistic");
}
