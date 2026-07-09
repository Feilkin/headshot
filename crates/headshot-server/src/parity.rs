//! Parity-harness scaffolding (doc/02 §5): comparison metrics, gates, and
//! reference-dump loading.
//!
//! Fixtures (converted weights, reference activation dumps) are generated
//! locally by each developer with tools/convert_weights.py and
//! tools/dump_reference.py — never committed. Tests that need them are
//! `#[ignore]`d and run via `just parity`; set `HEADSHOT_FIXTURES_DIR` to
//! point somewhere other than `<repo>/fixtures`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;

/// Comparison metrics between our activation and the reference's (doc/02 §5).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metrics {
    pub max_abs_err: f64,
    pub mean_abs_err: f64,
    pub cosine_sim: f64,
    /// max |reference| — the activation scale the errors live on. Trunk
    /// activations grow to O(100–1000), so gates use `rel_max_err`.
    pub ref_scale: f64,
}

impl Metrics {
    /// `max_abs_err` relative to the reference's own magnitude.
    pub fn rel_max_err(&self) -> f64 {
        self.max_abs_err / self.ref_scale.max(f64::MIN_POSITIVE)
    }
}

pub fn compare(ours: &[f32], reference: &[f32]) -> Metrics {
    assert_eq!(ours.len(), reference.len(), "length mismatch");
    assert!(!ours.is_empty());
    let mut max_abs: f64 = 0.0;
    let mut sum_abs: f64 = 0.0;
    let mut dot: f64 = 0.0;
    let mut norm_a: f64 = 0.0;
    let mut norm_b: f64 = 0.0;
    let mut ref_scale: f64 = 0.0;
    for (&a, &b) in ours.iter().zip(reference) {
        let (a, b) = (a as f64, b as f64);
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        sum_abs += d;
        dot += a * b;
        norm_a += a * a;
        norm_b += b * b;
        ref_scale = ref_scale.max(b.abs());
    }
    Metrics {
        max_abs_err: max_abs,
        mean_abs_err: sum_abs / ours.len() as f64,
        cosine_sim: dot / (norm_a.sqrt() * norm_b.sqrt()).max(f64::MIN_POSITIVE),
        ref_scale,
    }
}

/// A pass/fail gate over [`Metrics`]. Values from doc/02 §5, adjusted by
/// measurement: absolute thresholds are scale-relative because trunk
/// activations reach O(100–1000) while cosine stays ~1−2e-6 per layer in
/// the all-f32 engine — a kernel bug shows as a cliff, not smooth drift.
#[derive(Debug, Clone, Copy)]
pub struct Gate {
    pub rel_max: Option<f64>,
    pub min_cosine: Option<f64>,
}

impl Gate {
    /// f32 stages (heads) and the all-f32 debug engine at shallow depth.
    pub const F32_TIGHT: Gate = Gate { rel_max: Some(1e-4), min_cosine: Some(1.0 - 1e-6) };
    /// all-f32 debug mode through the trunk. The bounds are measured, not
    /// arbitrary (tests/sensitivity.rs): the trunk amplifies a 1e-6 input
    /// perturbation to ~4e-4 (small scene) / ~1.1e-2 (realistic, 8×624×416)
    /// by layer 23, so reduction-order noise vs the CPU reference reaches
    /// ~5e-3 / ~1e-1 rel-max in deep layers while cosine stays ≥ 1−3e-5.
    /// Cosine carries the gate; rel_max only catches scale/NaN-class bugs.
    pub const F32_DEEP: Gate = Gate { rel_max: Some(0.5), min_cosine: Some(1.0 - 1e-4) };
    /// f16-compute trunk, per layer — watch for cliffs, not the drift.
    pub const F16_TRUNK: Gate = Gate { rel_max: None, min_cosine: Some(0.999) };

    pub fn check(&self, name: &str, m: Metrics) -> Result<Metrics, String> {
        if let Some(gate) = self.rel_max
            && m.rel_max_err() > gate
        {
            return Err(format!(
                "{name}: rel_max_err {:.3e} > {gate:.3e} ({m:?})",
                m.rel_max_err()
            ));
        }
        if let Some(gate) = self.min_cosine
            && m.cosine_sim < gate
        {
            return Err(format!("{name}: cosine_sim {:.9} < {gate:.9} ({m:?})", m.cosine_sim));
        }
        Ok(m)
    }
}

/// Directory holding locally generated fixtures (weights + dumps).
pub fn fixtures_dir() -> PathBuf {
    std::env::var_os("HEADSHOT_FIXTURES_DIR").map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures"),
        PathBuf::from,
    )
}

/// Metadata embedded by tools/dump_reference.py.
#[derive(Debug, Deserialize)]
pub struct DumpMeta {
    pub scene: String,
    pub variant: String,
    pub device: String,
    pub n_frames: usize,
    pub width: u32,
    pub height: u32,
    pub seed: u64,
    pub shapes: BTreeMap<String, Vec<usize>>,
}

/// A reference activation dump (one scene × one precision variant).
pub struct Dump {
    pub meta: DumpMeta,
    mmap: Mmap,
}

impl Dump {
    /// Open `dump_{scene}_{variant}.safetensors` from the fixtures dir.
    pub fn open(scene: &str, variant: &str) -> anyhow::Result<Self> {
        Self::open_path(&fixtures_dir().join(format!("dumps/dump_{scene}_{variant}.safetensors")))
    }

    pub fn open_path(path: &Path) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)
            .map_err(|e| anyhow::anyhow!("{}: {e} (generate with tools/dump_reference.py)", path.display()))?;
        // SAFETY: read-only mapping, local fixture file.
        let mmap = unsafe { Mmap::map(&file)? };
        let (_, meta) = SafeTensors::read_metadata(&mmap)?;
        let meta_json = meta
            .metadata()
            .as_ref()
            .and_then(|m| m.get("headshot"))
            .ok_or_else(|| anyhow::anyhow!("{}: missing `headshot` metadata", path.display()))?;
        let meta: DumpMeta = serde_json::from_str(meta_json)?;
        Ok(Self { meta, mmap })
    }

    /// Tensor contents as f32 (dumps are saved all-f32).
    pub fn tensor(&self, name: &str) -> anyhow::Result<(Vec<usize>, Vec<f32>)> {
        let st = SafeTensors::deserialize(&self.mmap)?;
        let view = st.tensor(name)?;
        anyhow::ensure!(
            view.dtype() == Dtype::F32,
            "{name}: dump tensors must be f32, got {:?}",
            view.dtype()
        );
        let data = bytemuck::cast_slice::<u8, f32>(view.data()).to_vec();
        Ok((view.shape().to_vec(), data))
    }

    pub fn names(&self) -> anyhow::Result<Vec<String>> {
        Ok(SafeTensors::deserialize(&self.mmap)?
            .names()
            .into_iter()
            .map(String::from)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_on_identical_and_scaled() {
        let a = [1.0f32, -2.0, 3.0, 0.5];
        let m = compare(&a, &a);
        assert_eq!(m.max_abs_err, 0.0);
        assert_eq!(m.mean_abs_err, 0.0);
        assert!((m.cosine_sim - 1.0).abs() < 1e-12);

        // Same direction, different scale: cosine 1, nonzero abs error.
        let b: Vec<f32> = a.iter().map(|x| 2.0 * x).collect();
        let m = compare(&a, &b);
        assert!((m.cosine_sim - 1.0).abs() < 1e-12);
        assert_eq!(m.max_abs_err, 3.0);

        // Opposite direction.
        let c: Vec<f32> = a.iter().map(|x| -x).collect();
        assert!((compare(&a, &c).cosine_sim + 1.0).abs() < 1e-12);
    }

    #[test]
    fn gate_check() {
        let good =
            Metrics { max_abs_err: 1e-5, mean_abs_err: 1e-6, cosine_sim: 1.0, ref_scale: 1.0 };
        let drifted =
            Metrics { max_abs_err: 0.5, mean_abs_err: 0.1, cosine_sim: 0.9, ref_scale: 1.0 };
        assert!(Gate::F32_TIGHT.check("t", good).is_ok());
        assert!(Gate::F32_TIGHT.check("t", drifted).is_err());
        assert!(Gate::F16_TRUNK.check("t", drifted).is_err());
        // F16 gate has no abs bound: large-but-aligned drift passes.
        let aligned =
            Metrics { max_abs_err: 0.5, mean_abs_err: 0.1, cosine_sim: 0.9999, ref_scale: 1.0 };
        assert!(Gate::F16_TRUNK.check("t", aligned).is_ok());
        // same absolute error on large-scale activations passes F32_DEEP
        let scaled =
            Metrics { max_abs_err: 0.5, mean_abs_err: 0.1, cosine_sim: 1.0, ref_scale: 1000.0 };
        assert!(Gate::F32_DEEP.check("t", scaled).is_ok());
    }
}
