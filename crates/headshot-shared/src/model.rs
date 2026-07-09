//! VGGT-Omega 1B architecture constants, from doc/01-model-spec.md.
//!
//! Shapes use `N` = frames, `H×W` = input pixels, `H' = H/16`, `W' = W/16`,
//! `P = H'·W'` patch tokens per frame.

/// ViT patch size; input dims must be divisible by this.
pub const PATCH_SIZE: u32 = 16;

/// Token embedding dim of DINO and the aggregator trunk (`C`).
pub const EMBED_DIM: usize = 1024;

/// Attention head dim in DINO and the trunk (16 heads × 64).
pub const HEAD_DIM: usize = 64;

/// DINO blocks, aggregator frame blocks, and aggregator inter-frame blocks each.
pub const NUM_BLOCKS: usize = 24;

/// Per-frame trunk prefix: 1 camera token + 16 register tokens.
pub const PREFIX_TOKENS: usize = 17;

/// DINO prefix: 1 cls token + 4 storage tokens (dropped after DINO).
pub const DINO_PREFIX_TOKENS: usize = 5;

/// Inter-frame blocks that run register attention instead of global attention.
pub const REGISTER_ATTENTION_LAYERS: [usize; 5] = [2, 6, 9, 14, 20];

/// Inter-frame block outputs cached for the heads (doc/01 §3.3).
pub const CACHED_LAYERS: [usize; 4] = [4, 11, 17, 23];

/// Length of a per-frame pose encoding: t(3), quat xyzw(4), fov_h, fov_w.
pub const POSE_ENC_LEN: usize = 9;

/// Dense head processes frames in independent chunks of this size (doc/01 §4.2);
/// this is also the streaming unit of `DepthChunkMsg` (doc/04).
pub const DEPTH_CHUNK_FRAMES: usize = 8;

/// ImageNet normalization applied server-side as the first model op (doc/01 §1).
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Patch tokens per frame (`P`) for a valid input size.
///
/// Panics if `width`/`height` are not multiples of [`PATCH_SIZE`] — the
/// server rejects such frames before this is ever reached.
pub fn patch_tokens(width: u32, height: u32) -> usize {
    assert!(
        width.is_multiple_of(PATCH_SIZE) && height.is_multiple_of(PATCH_SIZE),
        "frame dims must be divisible by {PATCH_SIZE}, got {width}x{height}"
    );
    (width / PATCH_SIZE) as usize * (height / PATCH_SIZE) as usize
}

/// Per-frame trunk sequence length (`T = P + 17`).
pub fn frame_tokens(width: u32, height: u32) -> usize {
    patch_tokens(width, height) + PREFIX_TOKENS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_counts_match_spec_example() {
        // doc/03 §3: P = 1014, T = 1031 at 624×416.
        assert_eq!(patch_tokens(624, 416), 1014);
        assert_eq!(frame_tokens(624, 416), 1031);
        // small parity fixture (doc/02 §5): 128×96
        assert_eq!(patch_tokens(128, 96), 48);
    }

    #[test]
    #[should_panic(expected = "divisible by 16")]
    fn rejects_unaligned_dims() {
        patch_tokens(625, 416);
    }
}
