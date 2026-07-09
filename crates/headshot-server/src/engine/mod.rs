//! WGSL inference engine (doc/03).
//!
//! M1 state: all-f32 compute — this is the "debug mode" of doc/02 §5 and the
//! correctness baseline the parity gates validate; the f16-storage /
//! f32-accumulate fast path (WMMA) layers on top later.
//!
//! Execution model: ops encode compute dispatches into one open command
//! encoder on [`GpuContext`]; nothing runs until [`GpuContext::flush`] (or a
//! download, which flushes first). This keeps per-op overhead low without a
//! scheduler.

pub mod ops;
pub mod rope;
pub mod tensor;

use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::{Context as _, Result};
use tensor::GpuTensor;
use wgpu::util::DeviceExt;

/// Compiled-in WGSL sources (WESL artifacts from build.rs). Each dual
/// kernel has an f32 variant and an `_f16` variant (f16 storage, f32 math).
macro_rules! kernel_sources {
    ($($name:literal),* $(,)?) => {
        &[
            $(
                ($name, include_str!(concat!(env!("OUT_DIR"), "/", $name, ".wgsl"))),
                (
                    concat!($name, "_f16"),
                    include_str!(concat!(env!("OUT_DIR"), "/", $name, "_f16.wgsl")),
                ),
            )*
        ]
    };
}

const DUAL_KERNELS: &[(&str, &str)] = kernel_sources![
    "linear",
    "layernorm",
    "gelu",
    "residual_ls",
    "qkv_split",
    "attn_merge",
    "rope_apply",
    "attention",
    "im2col_patch",
    "concat_channels",
];

const RESIDUAL_ADD: &str = include_str!(concat!(env!("OUT_DIR"), "/residual_add.wgsl"));
/// Dense-head kernels (f32 only; doc/01 §4.2).
const DENSE_KERNELS: &[(&str, &str)] = &[
    ("unary", include_str!(concat!(env!("OUT_DIR"), "/unary.wgsl"))),
    ("tiled_add", include_str!(concat!(env!("OUT_DIR"), "/tiled_add.wgsl"))),
    ("im2col3x3", include_str!(concat!(env!("OUT_DIR"), "/im2col3x3.wgsl"))),
    ("shuffle_expand", include_str!(concat!(env!("OUT_DIR"), "/shuffle_expand.wgsl"))),
    ("bilinear", include_str!(concat!(env!("OUT_DIR"), "/bilinear.wgsl"))),
];
const CAST_TO_F16: &str = include_str!(concat!(env!("OUT_DIR"), "/cast_f32_to_f16.wgsl"));
const CAST_TO_F32: &str = include_str!(concat!(env!("OUT_DIR"), "/cast_f16_to_f32.wgsl"));
/// Camera-head attention: head_dim 128, f32 only (doc/01 §4.1).
const ATTENTION_D128: &str = include_str!(concat!(env!("OUT_DIR"), "/attention_d128.wgsl"));
/// Plain WGSL (naga cooperative-matrix extension; wgsl-parse can't parse it,
/// so it bypasses WESL).
const LINEAR_WMMA: &str = include_str!("../../shaders/linear_wmma.wgsl");

/// Maximum submissions in flight before [`GpuContext::flush`] blocks on the
/// oldest. This is the engine's memory backpressure: dropped buffers are
/// only reclaimed once the submission using them completes on the GPU, and
/// the CPU encodes the whole network in milliseconds while the GPU takes
/// minutes — unbounded, every layer's intermediates coexist (~200 GB at
/// 100 frames, absorbed only by GTT overcommit and paging).
///
/// Measured on a 100-frame drone scene (512², f16 trunk): depth 0 →
/// 37 GiB peak / 248 s; depth 1 → 75 GiB / 230 s; depth 2 → 118 GiB /
/// 227 s. Each unit of depth retains ~38 GiB — far more than one
/// submission's ~5–9 GB of transients, i.e. the allocator grows heavily
/// under interleaved alloc/free generations. Until the engine gets a
/// buffer-reuse pool, full serialization is the right default: ~9 % wall
/// clock for a budget compatible with 500-frame sessions. Override with
/// HEADSHOT_MAX_IN_FLIGHT for experiments.
const MAX_IN_FLIGHT: usize = 0;

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pipelines: HashMap<&'static str, wgpu::ComputePipeline>,
    encoder: Mutex<Option<wgpu::CommandEncoder>>,
    in_flight: Mutex<std::collections::VecDeque<wgpu::SubmissionIndex>>,
    /// Cooperative cancellation (doc/03 §7): model forwards check this
    /// between blocks and bail with [`Cancelled`]. Set from any thread via
    /// [`GpuContext::request_cancel`]; reset at session start.
    cancel: std::sync::atomic::AtomicBool,
    /// shader-f16 available: the f16 kernel variants exist.
    pub f16_supported: bool,
    /// 16x16x16 f16→f32 cooperative matrix available: `linear_wmma` exists.
    pub wmma_supported: bool,
}

/// Marker error for cooperative cancellation; detect with
/// `err.is::<Cancelled>()` on the anyhow chain.
#[derive(Debug, thiserror::Error)]
#[error("cancelled")]
pub struct Cancelled;

impl GpuContext {
    /// Create a context on the first available adapter. `Err` when no
    /// adapter exists (CI without a GPU) — callers in tests skip then.
    pub fn new() -> Result<Self> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster_block(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .context("no GPU adapter")?;

        // Activations at high frame counts exceed the default 128 MiB
        // binding limit; take what the adapter offers.
        let adapter_limits = adapter.limits();
        let mut limits = wgpu::Limits::defaults();
        limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;
        limits.max_buffer_size = adapter_limits.max_buffer_size;
        limits.max_storage_buffers_per_shader_stage =
            adapter_limits.max_storage_buffers_per_shader_stage;
        // flash attention stages 64x64 K and V tiles in workgroup memory
        // (32 KiB > the 16 KiB WebGPU default)
        limits.max_compute_workgroup_storage_size =
            adapter_limits.max_compute_workgroup_storage_size;

        let mut features = wgpu::Features::empty();
        for wanted in [
            wgpu::Features::SHADER_F16,
            wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX,
            wgpu::Features::TIMESTAMP_QUERY,
        ] {
            if adapter.features().contains(wanted) {
                features |= wanted;
            }
        }

        let (device, queue) = pollster_block(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("headshot-engine"),
            required_features: features,
            required_limits: limits,
            // Cooperative matrix (the WMMA GEMM path, doc/03) is behind
            // wgpu's experimental opt-in.
            // SAFETY: acknowledges possible bugs in experimental APIs.
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            ..Default::default()
        }))?;

        let f16_supported = features.contains(wgpu::Features::SHADER_F16);
        let wmma_supported = f16_supported
            && features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
            && adapter.cooperative_matrix_properties().iter().any(|c| {
                (c.m_size, c.n_size, c.k_size) == (16, 16, 16)
                    && c.ab_type == wgpu::CooperativeScalarType::F16
                    && c.cr_type == wgpu::CooperativeScalarType::F32
            });

        let make_pipeline = |device: &wgpu::Device, name: &str, source: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let mut pipelines = HashMap::new();
        for (name, source) in DUAL_KERNELS {
            if name.ends_with("_f16") && !f16_supported {
                continue;
            }
            pipelines.insert(*name, make_pipeline(&device, name, source));
        }
        pipelines.insert("residual_add", make_pipeline(&device, "residual_add", RESIDUAL_ADD));
        for (name, source) in DENSE_KERNELS {
            pipelines.insert(*name, make_pipeline(&device, name, source));
        }
        pipelines.insert("attention_d128", make_pipeline(&device, "attention_d128", ATTENTION_D128));
        if f16_supported {
            pipelines
                .insert("cast_f32_to_f16", make_pipeline(&device, "cast_f32_to_f16", CAST_TO_F16));
            pipelines
                .insert("cast_f16_to_f32", make_pipeline(&device, "cast_f16_to_f32", CAST_TO_F32));
        }
        if wmma_supported {
            pipelines.insert("linear_wmma", make_pipeline(&device, "linear_wmma", LINEAR_WMMA));
        }

        Ok(Self {
            device,
            queue,
            pipelines,
            encoder: Mutex::new(None),
            in_flight: Mutex::new(std::collections::VecDeque::new()),
            cancel: std::sync::atomic::AtomicBool::new(false),
            f16_supported,
            wmma_supported,
        })
    }

    /// Encode one compute dispatch. Convention shared by every kernel:
    /// binding 0 is a uniform param struct, bindings 1.. are storage buffers
    /// in argument order.
    pub fn dispatch(
        &self,
        kernel: &str,
        params: &[u8],
        buffers: &[&GpuTensor],
        workgroups: [u32; 3],
    ) {
        let pipeline = self
            .pipelines
            .get(kernel)
            .unwrap_or_else(|| panic!("unknown kernel {kernel}"));
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(kernel),
            contents: params,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: params_buf.as_entire_binding(),
        }];
        for (i, tensor) in buffers.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: (i + 1) as u32,
                resource: tensor.buffer.as_entire_binding(),
            });
        }
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(kernel),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });

        let mut guard = self.encoder.lock().unwrap();
        let encoder = guard.get_or_insert_with(|| {
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("engine"),
            })
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
    }

    /// Submit all encoded work. Cheap no-op when nothing is pending.
    pub fn flush(&self) {
        if let Some(encoder) = self.encoder.lock().unwrap().take() {
            let index = self.queue.submit([encoder.finish()]);
            let mut in_flight = self.in_flight.lock().unwrap();
            in_flight.push_back(index);
            // Backpressure (see MAX_IN_FLIGHT): block on the oldest
            // submission so completed blocks' buffers get reclaimed before
            // the next block allocates its own.
            let max_in_flight = std::env::var("HEADSHOT_MAX_IN_FLIGHT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(MAX_IN_FLIGHT);
            while in_flight.len() > max_in_flight {
                let oldest = in_flight.pop_front().unwrap();
                let t = std::time::Instant::now();
                self.device
                    .poll(wgpu::PollType::Wait { submission_index: Some(oldest), timeout: None })
                    .expect("device poll");
                if std::env::var_os("HEADSHOT_TRACE_FLUSH").is_some() {
                    eprintln!("flush wait: {:?}", t.elapsed());
                }
            }
        }
    }

    /// Request cooperative cancellation of the in-progress forward passes
    /// (safe from any thread).
    pub fn request_cancel(&self) {
        self.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Clear the cancel flag (call at session start).
    pub fn reset_cancel(&self) {
        self.cancel.store(false, std::sync::atomic::Ordering::Relaxed);
    }

    /// `Err(Cancelled)` when cancellation was requested; model forwards
    /// call this between blocks.
    pub fn check_cancelled(&self) -> Result<()> {
        if self.cancel.load(std::sync::atomic::Ordering::Relaxed) {
            anyhow::bail!(Cancelled);
        }
        Ok(())
    }

    /// Flush and block until the GPU is idle.
    pub fn sync(&self) {
        self.flush();
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
        self.in_flight.lock().unwrap().clear();
    }

    /// Record a buffer-to-buffer copy in the open encoder (used for token
    /// gather/scatter where the region is contiguous).
    pub fn copy(
        &self,
        src: &GpuTensor,
        src_offset: u64,
        dst: &GpuTensor,
        dst_offset: u64,
        bytes: u64,
    ) {
        let mut guard = self.encoder.lock().unwrap();
        let encoder = guard.get_or_insert_with(|| {
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("engine") })
        });
        encoder.copy_buffer_to_buffer(&src.buffer, src_offset, &dst.buffer, dst_offset, bytes);
    }
}

/// 2D grid for `total` threads at `wg` threads/workgroup, avoiding the
/// 65535 per-dimension dispatch limit. Kernels using this compute
/// `index = (wg_id.y * num_wg_x + wg_id.x) * wg_size + local_index`.
pub fn grid_2d(total: u64, wg: u64) -> [u32; 3] {
    let wgs = total.div_ceil(wg);
    if wgs <= 65535 {
        [wgs as u32, 1, 1]
    } else {
        let x = 256u64;
        [x as u32, wgs.div_ceil(x) as u32, 1]
    }
}

/// Minimal blocking executor for wgpu's native (immediately-ready) futures.
pub fn pollster_block<F: Future>(future: F) -> F::Output {
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    fn noop_raw_waker() -> RawWaker {
        fn clone(_: *const ()) -> RawWaker {
            noop_raw_waker()
        }
        fn noop(_: *const ()) {}
        RawWaker::new(std::ptr::null(), &RawWakerVTable::new(clone, noop, noop, noop))
    }

    let waker = unsafe { Waker::from_raw(noop_raw_waker()) };
    let mut cx = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(out) => return out,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}
