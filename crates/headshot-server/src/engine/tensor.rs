//! GPU tensors. M1: f32 only (the all-f32 debug engine); dtype plumbing for
//! f16 arrives with the fast path.

use super::GpuContext;
use wgpu::util::DeviceExt;

pub struct GpuTensor {
    pub buffer: wgpu::Buffer,
    pub shape: Vec<usize>,
}

impl GpuTensor {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn byte_len(&self) -> u64 {
        (self.len() * 4) as u64
    }

    /// Reinterpret the shape (same element count).
    pub fn reshaped(self, shape: &[usize]) -> Self {
        assert_eq!(self.len(), shape.iter().product::<usize>(), "reshape element count");
        Self { buffer: self.buffer, shape: shape.to_vec() }
    }
}

impl GpuContext {
    pub fn tensor_from_slice(&self, shape: &[usize], data: &[f32]) -> GpuTensor {
        assert_eq!(data.len(), shape.iter().product::<usize>(), "shape/data mismatch");
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        GpuTensor { buffer, shape: shape.to_vec() }
    }

    /// Uninitialized tensor (kernel output).
    pub fn empty(&self, shape: &[usize]) -> GpuTensor {
        let len: usize = shape.iter().product();
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        GpuTensor { buffer, shape: shape.to_vec() }
    }

    /// Flushes pending work, reads the tensor back.
    pub fn download(&self, tensor: &GpuTensor) -> Vec<f32> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("download"),
            size: tensor.byte_len(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        {
            let mut guard = self.encoder.lock().unwrap();
            let encoder = guard.get_or_insert_with(|| {
                self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("engine"),
                })
            });
            encoder.copy_buffer_to_buffer(&tensor.buffer, 0, &staging, 0, tensor.byte_len());
        }
        self.sync();
        staging.map_async(wgpu::MapMode::Read, .., |r| r.expect("map"));
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
        let data = staging.get_mapped_range(..);
        bytemuck::cast_slice(&data).to_vec()
    }
}
