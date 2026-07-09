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

use headshot_server::parity::{Dump, fixtures_dir};
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
