//! Converted-checkpoint loading and validation (doc/02 §3, §6).
//!
//! `tools/convert_weights.py` emits `model.safetensors` + `manifest.json`.
//! Before any GPU pipeline is created we verify, with a full diff on
//! mismatch: the manifest matches the expected model tree, the safetensors
//! contents match the manifest, always-f32 tensors really are f32, and the
//! file hash round-trips against the manifest.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

use memmap2::Mmap;
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use sha2::{Digest, Sha256};

/// Trunk/DINO embedding dim (doc/01).
const DIM: usize = 1024;
/// Camera/dense head input dim (2 × DIM).
const HEAD_IN: usize = 2048;
/// DPT pyramid channels per level (doc/01 §4.2).
const DENSE_OC: [usize; 4] = [256, 512, 1024, 1024];
/// Fused-map channel count.
const DENSE_FEATURES: usize = 256;

#[derive(Debug, thiserror::Error)]
pub enum WeightsError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("manifest parse: {0}")]
    ManifestParse(#[from] serde_json::Error),
    #[error("safetensors parse: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),
    #[error("model.safetensors hash mismatch: manifest {expected}, file {actual}")]
    HashMismatch { expected: String, actual: String },
    #[error("weights do not match the expected model tree:\n{diff}")]
    TreeMismatch { diff: String },
}

#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub config: serde_json::Value,
    pub checkpoint_sha256: String,
    pub model_sha256: String,
    pub conversion_tool_version: String,
    pub tensors: BTreeMap<String, TensorMeta>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
pub struct TensorMeta {
    pub shape: Vec<usize>,
    pub dtype: DtypeMeta,
}

#[derive(Debug, Deserialize, PartialEq, Eq, Clone, Copy)]
pub enum DtypeMeta {
    #[serde(rename = "float16")]
    F16,
    #[serde(rename = "float32")]
    F32,
}

/// The converted checkpoint, memory-mapped and fully validated.
pub struct Weights {
    pub manifest: Manifest,
    mmap: Mmap,
}

impl Weights {
    /// Open `manifest.json` + `model.safetensors` from `dir` and run every
    /// validation. Fails before returning if anything is off.
    pub fn open(dir: &Path) -> Result<Self, WeightsError> {
        let manifest: Manifest =
            serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json"))?)?;
        validate_manifest_tree(&manifest)?;

        let file = std::fs::File::open(dir.join("model.safetensors"))?;
        // SAFETY: read-only mapping of a file we just opened; concurrent
        // truncation would be an external fault we accept for local fixtures.
        let mmap = unsafe { Mmap::map(&file)? };

        let actual = hex(&Sha256::digest(&mmap[..]));
        if actual != manifest.model_sha256 {
            return Err(WeightsError::HashMismatch {
                expected: manifest.model_sha256.clone(),
                actual,
            });
        }

        let weights = Self { manifest, mmap };
        weights.validate_contents()?;
        Ok(weights)
    }

    /// Access the deserialized tensor views (borrow from the mmap).
    pub fn tensors(&self) -> Result<SafeTensors<'_>, WeightsError> {
        Ok(SafeTensors::deserialize(&self.mmap)?)
    }

    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>, WeightsError> {
        Ok(self.tensors()?.tensor(name)?)
    }

    /// File contents must match the manifest exactly: same names, shapes,
    /// dtypes.
    fn validate_contents(&self) -> Result<(), WeightsError> {
        let st = self.tensors()?;
        let mut actual = BTreeMap::new();
        for (name, view) in st.tensors() {
            let dtype = match view.dtype() {
                Dtype::F16 => DtypeMeta::F16,
                Dtype::F32 => DtypeMeta::F32,
                other => {
                    return Err(WeightsError::TreeMismatch {
                        diff: format!("{name}: unsupported dtype {other:?}"),
                    });
                }
            };
            actual.insert(
                name,
                TensorMeta {
                    shape: view.shape().to_vec(),
                    dtype,
                },
            );
        }
        diff_trees("manifest", &self.manifest.tensors, "file", &actual)
    }
}

/// Validate the manifest against the expected model tree (doc/02 §3):
/// exact name set, exact shapes, and f32 where f32 is mandatory (§4).
pub fn validate_manifest_tree(manifest: &Manifest) -> Result<(), WeightsError> {
    let expected: BTreeMap<String, TensorMeta> = expected_tree()
        .into_iter()
        .map(|(name, shape)| {
            // f16 storage is an option only for big GEMM weights; the
            // outlier scan may still have kept any of them f32, so expect
            // "either" there by mirroring the manifest's dtype when legal.
            let dtype = if must_be_f32(&name) {
                DtypeMeta::F32
            } else {
                manifest
                    .tensors
                    .get(&name)
                    .map(|t| t.dtype)
                    .unwrap_or(DtypeMeta::F16)
            };
            (name, TensorMeta { shape, dtype })
        })
        .collect();
    diff_trees("expected", &expected, "manifest", &manifest.tensors)
}

fn diff_trees(
    a_name: &str,
    a: &BTreeMap<String, TensorMeta>,
    b_name: &str,
    b: &BTreeMap<String, TensorMeta>,
) -> Result<(), WeightsError> {
    let mut diff = String::new();
    for (name, meta) in a {
        match b.get(name) {
            None => _ = writeln!(diff, "missing from {b_name}: {name} {meta:?}"),
            Some(other) if other != meta => {
                _ = writeln!(diff, "mismatch {name}: {a_name} {meta:?}, {b_name} {other:?}");
            }
            Some(_) => {}
        }
    }
    for (name, meta) in b {
        if !a.contains_key(name) {
            _ = writeln!(diff, "unexpected in {b_name}: {name} {meta:?}");
        }
    }
    if diff.is_empty() {
        Ok(())
    } else {
        Err(WeightsError::TreeMismatch { diff })
    }
}

/// Tensors that must be stored f32 (doc/02 §4). Mirrors
/// `is_always_f32` in tools/convert_weights.py.
fn must_be_f32(name: &str) -> bool {
    name.contains(".norm.weight")
        || name.contains(".norm.bias")
        || [".norm1.", ".norm2.", ".q_norm.", ".k_norm.", ".token_norm.", ".trunk_norm."]
            .iter()
            .any(|n| name.contains(n))
        || name.ends_with(".gamma")
        || name.ends_with(".periods")
        || matches!(
            name,
            "aggregator.camera_token"
                | "aggregator.register_token"
                | "aggregator.patch_embed.cls_token"
                | "aggregator.patch_embed.storage_tokens"
        )
        || name.starts_with("camera_head.camera_branch.2.")
        || name.starts_with("dense_head.proj.")
        || name.starts_with("dense_head.proj_conf.")
        || name.ends_with(".bias")
}

/// One pre-norm transformer block's tensors (doc/01 §2, §3.2), post-conversion
/// (bias_mask already folded away).
fn push_block(tree: &mut Vec<(String, Vec<usize>)>, prefix: &str, dim: usize, qk_norm: bool) {
    let hidden = 4 * dim;
    let mut push = |suffix: &str, shape: Vec<usize>| {
        tree.push((format!("{prefix}.{suffix}"), shape));
    };
    push("norm1.weight", vec![dim]);
    push("norm1.bias", vec![dim]);
    push("attn.qkv.weight", vec![3 * dim, dim]);
    push("attn.qkv.bias", vec![3 * dim]);
    push("attn.proj.weight", vec![dim, dim]);
    push("attn.proj.bias", vec![dim]);
    push("ls1.gamma", vec![dim]);
    push("norm2.weight", vec![dim]);
    push("norm2.bias", vec![dim]);
    push("mlp.fc1.weight", vec![hidden, dim]);
    push("mlp.fc1.bias", vec![hidden]);
    push("mlp.fc2.weight", vec![dim, hidden]);
    push("mlp.fc2.bias", vec![dim]);
    push("ls2.gamma", vec![dim]);
    if qk_norm {
        for norm in ["q_norm", "k_norm"] {
            push(&format!("attn.{norm}.weight"), vec![64]);
            push(&format!("attn.{norm}.bias"), vec![64]);
        }
    }
}

/// The full expected post-conversion tensor tree: name → shape (doc/02 §3).
/// Mirrors `expected_tree` in tools/convert_weights.py minus the consumed
/// `bias_mask` buffers and dropped `mask_token`.
pub fn expected_tree() -> Vec<(String, Vec<usize>)> {
    let mut tree: Vec<(String, Vec<usize>)> = Vec::with_capacity(1400);
    let mut push = |name: &str, shape: Vec<usize>| tree.push((name.to_string(), shape));

    // DINOv3 ViT-L/16 (doc/01 §2)
    let dino = "aggregator.patch_embed";
    push(&format!("{dino}.cls_token"), vec![1, 1, DIM]);
    push(&format!("{dino}.storage_tokens"), vec![1, 4, DIM]);
    push(&format!("{dino}.patch_embed.proj.weight"), vec![DIM, 3, 16, 16]);
    push(&format!("{dino}.patch_embed.proj.bias"), vec![DIM]);
    push(&format!("{dino}.rope_embed.periods"), vec![16]);
    push(&format!("{dino}.norm.weight"), vec![DIM]);
    push(&format!("{dino}.norm.bias"), vec![DIM]);
    for i in 0..24 {
        push_block(&mut tree, &format!("{dino}.blocks.{i}"), DIM, false);
    }

    // Aggregator trunk (doc/01 §3)
    tree.push(("aggregator.rope_embed.periods".into(), vec![16]));
    tree.push(("aggregator.camera_token".into(), vec![1, 2, 1, DIM]));
    tree.push(("aggregator.register_token".into(), vec![1, 2, 16, DIM]));
    for i in 0..24 {
        push_block(&mut tree, &format!("aggregator.frame_blocks.{i}"), DIM, true);
        push_block(&mut tree, &format!("aggregator.inter_frame_blocks.{i}"), DIM, true);
    }

    let mut push = |name: &str, shape: Vec<usize>| tree.push((name.to_string(), shape));

    // Camera head (doc/01 §4.1)
    push("camera_head.token_norm.weight", vec![HEAD_IN]);
    push("camera_head.token_norm.bias", vec![HEAD_IN]);
    push("camera_head.trunk_norm.weight", vec![HEAD_IN]);
    push("camera_head.trunk_norm.bias", vec![HEAD_IN]);
    push("camera_head.camera_branch.0.weight", vec![DIM, HEAD_IN]);
    push("camera_head.camera_branch.0.bias", vec![DIM]);
    push("camera_head.camera_branch.2.weight", vec![9, DIM]);
    push("camera_head.camera_branch.2.bias", vec![9]);
    for i in 0..4 {
        push_block(&mut tree, &format!("camera_head.trunk.{i}"), HEAD_IN, false);
    }

    let mut push = |name: &str, shape: Vec<usize>| tree.push((name.to_string(), shape));

    // Dense head (doc/01 §4.2)
    push("dense_head.norm.weight", vec![HEAD_IN]);
    push("dense_head.norm.bias", vec![HEAD_IN]);
    for (j, oc) in DENSE_OC.into_iter().enumerate() {
        push(&format!("dense_head.projects.{j}.weight"), vec![oc, HEAD_IN, 1, 1]);
        push(&format!("dense_head.projects.{j}.bias"), vec![oc]);
    }
    // resize layers keep per-level channels; idx 2 is identity (absent)
    push("dense_head.resize_layers.0.weight", vec![DENSE_OC[0], DENSE_OC[0], 4, 4]);
    push("dense_head.resize_layers.0.bias", vec![DENSE_OC[0]]);
    push("dense_head.resize_layers.1.weight", vec![DENSE_OC[1], DENSE_OC[1], 2, 2]);
    push("dense_head.resize_layers.1.bias", vec![DENSE_OC[1]]);
    push("dense_head.resize_layers.3.weight", vec![DENSE_OC[3], DENSE_OC[3], 3, 3]);
    push("dense_head.resize_layers.3.bias", vec![DENSE_OC[3]]);
    for k in 1..=4usize {
        push(
            &format!("dense_head.scratch.layer{k}_rn.weight"),
            vec![DENSE_FEATURES, DENSE_OC[k - 1], 3, 3],
        );
        let rn = format!("dense_head.scratch.refinenet{k}");
        let units: &[&str] = if k == 4 {
            &["resConfUnit2"]
        } else {
            &["resConfUnit1", "resConfUnit2"]
        };
        for unit in units {
            for conv in ["conv1", "conv2"] {
                push(
                    &format!("{rn}.{unit}.{conv}.weight"),
                    vec![DENSE_FEATURES, DENSE_FEATURES, 3, 3],
                );
                push(&format!("{rn}.{unit}.{conv}.bias"), vec![DENSE_FEATURES]);
            }
        }
        push(&format!("{rn}.out_conv.weight"), vec![DENSE_FEATURES, DENSE_FEATURES, 1, 1]);
        push(&format!("{rn}.out_conv.bias"), vec![DENSE_FEATURES]);
    }
    for proj in ["proj", "proj_conf"] {
        push(&format!("dense_head.{proj}.weight"), vec![16, DENSE_FEATURES, 1, 1]);
        push(&format!("dense_head.{proj}.bias"), vec![16]);
    }

    tree
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut s, b| {
        _ = write!(s, "{b:02x}");
        s
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_manifest(tensors: BTreeMap<String, TensorMeta>) -> Manifest {
        Manifest {
            config: serde_json::Value::Null,
            checkpoint_sha256: String::new(),
            model_sha256: String::new(),
            conversion_tool_version: "test".into(),
            tensors,
        }
    }

    fn full_fake_tensors() -> BTreeMap<String, TensorMeta> {
        expected_tree()
            .into_iter()
            .map(|(name, shape)| {
                let dtype = if must_be_f32(&name) || shape.len() == 1 {
                    DtypeMeta::F32
                } else {
                    DtypeMeta::F16
                };
                (name, TensorMeta { shape, dtype })
            })
            .collect()
    }

    #[test]
    fn expected_tree_size_and_uniqueness() {
        let tree = expected_tree();
        // 1411 checkpoint keys − 76 bias_masks − 1 mask_token = 1334
        assert_eq!(tree.len(), 1334);
        let names: std::collections::BTreeSet<_> = tree.iter().map(|(n, _)| n).collect();
        assert_eq!(names.len(), tree.len(), "duplicate tensor names");
    }

    #[test]
    fn manifest_matching_tree_validates() {
        validate_manifest_tree(&fake_manifest(full_fake_tensors())).unwrap();
    }

    #[test]
    fn manifest_diff_reports_all_mismatch_kinds() {
        let mut tensors = full_fake_tensors();
        tensors.remove("aggregator.camera_token");
        tensors.insert(
            "aggregator.bogus".into(),
            TensorMeta { shape: vec![1], dtype: DtypeMeta::F32 },
        );
        tensors.get_mut("camera_head.trunk_norm.weight").unwrap().shape = vec![7];
        // a mandatory-f32 tensor stored as f16 must also be caught
        tensors.get_mut("aggregator.frame_blocks.0.ls1.gamma").unwrap().dtype = DtypeMeta::F16;

        let err = validate_manifest_tree(&fake_manifest(tensors)).unwrap_err();
        let WeightsError::TreeMismatch { diff } = err else {
            panic!("expected TreeMismatch, got {err:?}");
        };
        assert!(diff.contains("missing from manifest: aggregator.camera_token"));
        assert!(diff.contains("unexpected in manifest: aggregator.bogus"));
        assert!(diff.contains("mismatch camera_head.trunk_norm.weight"));
        assert!(diff.contains("mismatch aggregator.frame_blocks.0.ls1.gamma"));
        assert_eq!(diff.lines().count(), 4, "no spurious diff lines:\n{diff}");
    }

    #[test]
    fn open_validates_hash_and_contents() {
        use safetensors::tensor::TensorView;

        let dir = tempfile::tempdir().unwrap();
        // A minimal well-formed pair: one tensor, correct hash.
        let data: Vec<u8> = bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0]).to_vec();
        let view = TensorView::new(Dtype::F32, vec![2, 2], &data).unwrap();
        let st = safetensors::serialize([("t".to_string(), view)], None).unwrap();
        std::fs::write(dir.path().join("model.safetensors"), &st).unwrap();

        let manifest = serde_json::json!({
            "config": null,
            "checkpoint_sha256": "",
            "model_sha256": hex(&Sha256::digest(&st)),
            "conversion_tool_version": "test",
            "tensors": {"t": {"shape": [2, 2], "dtype": "float32"}},
        });
        std::fs::write(dir.path().join("manifest.json"), manifest.to_string()).unwrap();

        // Full-tree validation would fail (it's not a real model), so test the
        // pieces `open` runs after it: hash + contents.
        let manifest: Manifest = serde_json::from_str(
            &std::fs::read_to_string(dir.path().join("manifest.json")).unwrap(),
        )
        .unwrap();
        let file = std::fs::File::open(dir.path().join("model.safetensors")).unwrap();
        let mmap = unsafe { Mmap::map(&file).unwrap() };
        assert_eq!(hex(&Sha256::digest(&mmap[..])), manifest.model_sha256);
        let weights = Weights { manifest, mmap };
        weights.validate_contents().unwrap();
        let view = weights.tensor("t").unwrap();
        assert_eq!(view.shape(), &[2, 2]);

        // Corrupt the manifest's shape → contents validation must fail.
        let mut bad = weights;
        bad.manifest.tensors.get_mut("t").unwrap().shape = vec![4];
        assert!(matches!(
            bad.validate_contents(),
            Err(WeightsError::TreeMismatch { .. })
        ));
    }
}
