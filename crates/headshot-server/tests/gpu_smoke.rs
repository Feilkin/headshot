//! End-to-end check of the wgpu + WESL pipeline: compile the residual_add
//! kernel, run it on the first available adapter, read back the result.
//!
//! Skips (passes with a note) when no GPU adapter is present so plain CI
//! runners stay green; kernel-correctness tests proper start in M1.

use wgpu::util::DeviceExt;

const RESIDUAL_ADD_WGSL: &str = include_str!(concat!(env!("OUT_DIR"), "/residual_add.wgsl"));

#[test]
fn residual_add_kernel() {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let Ok(adapter) = pollster_block(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    })) else {
        eprintln!("no GPU adapter available — skipping gpu_smoke");
        return;
    };

    let (device, queue) = pollster_block(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("gpu_smoke"),
        ..Default::default()
    }))
    .expect("device");

    let n = 1024usize;
    let acc_init: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let addend: Vec<f32> = (0..n).map(|i| 2.0 * i as f32).collect();

    let acc_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("acc"),
        contents: bytemuck::cast_slice(&acc_init),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let addend_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("addend"),
        contents: bytemuck::cast_slice(&addend),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (n * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("residual_add"),
        source: wgpu::ShaderSource::Wgsl(RESIDUAL_ADD_WGSL.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("residual_add"),
        layout: None,
        module: &module,
        entry_point: Some("residual_add"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: acc_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: addend_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(n.div_ceil(256) as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&acc_buf, 0, &readback, 0, (n * 4) as u64);
    queue.submit([encoder.finish()]);

    readback.map_async(wgpu::MapMode::Read, .., |r| r.expect("map"));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");

    let data = readback.get_mapped_range(..);
    let result: &[f32] = bytemuck::cast_slice(&data);
    for (i, &value) in result.iter().enumerate() {
        assert_eq!(value, 3.0 * i as f32, "mismatch at {i}");
    }
}

/// Minimal executor: wgpu native futures resolve without a reactor.
fn pollster_block<F: Future>(future: F) -> F::Output {
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    fn noop_raw_waker() -> RawWaker {
        fn clone(_: *const ()) -> RawWaker {
            noop_raw_waker()
        }
        fn noop(_: *const ()) {}
        RawWaker::new(
            std::ptr::null(),
            &RawWakerVTable::new(clone, noop, noop, noop),
        )
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
