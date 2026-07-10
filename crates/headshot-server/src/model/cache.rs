//! Streaming head-input cache (doc/01 §3.3).
//!
//! The trunk emits four cached layer outputs — each `(N·T, 2·1024)` f32,
//! consumed per-frame by the camera and dense heads (§4.1/§4.2). Held as one
//! buffer it overflows wgpu's 2 GiB cap past ~250 frames, so each is stored as
//! a list of frame-group chunks with per-frame row access. Every frame
//! occupies `t` contiguous rows (17 prefix + P patch tokens).

use crate::engine::GpuContext;
use crate::engine::tensor::{Dtype, GpuTensor};

pub struct Cache {
    /// f32 `(frames·t, 2·C)` chunks; chunk `c` holds frames
    /// `[c·frames_per_chunk, …)`, the last group possibly short.
    chunks: Vec<GpuTensor>,
    t: usize,
    frames_per_chunk: usize,
}

impl Cache {
    /// Concatenate the frame- and inter-block outputs `(N·T, C)` channel-wise
    /// and cast to f32, streamed in frame groups small enough that no chunk
    /// exceeds the buffer cap. Concat and cast are row-independent, so this is
    /// identical to a single monolithic concat+cast.
    pub fn build(
        ctx: &GpuContext,
        frame_out: &GpuTensor,
        inter_out: &GpuTensor,
        n: usize,
        t: usize,
    ) -> Self {
        let dim = *frame_out.shape.last().unwrap();
        // Largest frame group whose (frames·t, 2·dim) f32 chunk stays under
        // the device buffer cap.
        let frames_per_chunk = (ctx.max_rows(2 * dim, Dtype::F32) / t).clamp(1, n.max(1));
        let mut chunks = Vec::new();
        let mut f0 = 0;
        while f0 < n {
            let f1 = (f0 + frames_per_chunk).min(n);
            let rows = (f1 - f0) * t;
            let fo = ctx.empty_typed(&[rows, dim], frame_out.dtype);
            ctx.copy_rows(frame_out, f0 * t, &fo, 0, rows);
            let io = ctx.empty_typed(&[rows, dim], inter_out.dtype);
            ctx.copy_rows(inter_out, f0 * t, &io, 0, rows);
            let cat = ctx.concat_channels(&fo, &io);
            chunks.push(if cat.dtype == Dtype::F16 { ctx.cast_to_f32(&cat) } else { cat });
            // Bound peak to one group's transients under serialized backpressure.
            ctx.flush();
            f0 = f1;
        }
        Self { chunks, t, frames_per_chunk }
    }

    /// Wrap a monolithic `(N·T, 2·C)` f32 cache as a single-chunk `Cache`
    /// (parity tests feed reference caches this way).
    pub fn from_tensor(tensor: GpuTensor, n: usize) -> Self {
        assert_eq!(tensor.dtype, Dtype::F32, "head caches are f32");
        let dim = *tensor.shape.last().unwrap();
        let t = tensor.len() / dim / n.max(1);
        Self { chunks: vec![tensor], t, frames_per_chunk: n.max(1) }
    }

    /// Tokens per frame (`t = P + prefix`).
    pub fn tokens_per_frame(&self) -> usize {
        self.t
    }

    /// Copy `n_tokens` rows starting at token `tok0` of global `frame` into
    /// `dst` at row `dst_row`.
    pub fn copy_frame(
        &self,
        ctx: &GpuContext,
        frame: usize,
        tok0: usize,
        n_tokens: usize,
        dst: &GpuTensor,
        dst_row: usize,
    ) {
        let c = frame / self.frames_per_chunk;
        let local = frame % self.frames_per_chunk;
        ctx.copy_rows(&self.chunks[c], local * self.t + tok0, dst, dst_row, n_tokens);
    }

    /// The whole cache as one tensor when it wasn't split — parity taps at
    /// small frame counts, where a single chunk holds every frame.
    pub fn as_full(&self) -> Option<&GpuTensor> {
        (self.chunks.len() == 1).then(|| &self.chunks[0])
    }
}
