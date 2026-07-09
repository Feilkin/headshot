//! GPU tensors: f32 or f16 storage (math is always f32 in the kernels).
//!
//! Allocations carry 16 rows of zero-initialized slack past the logical
//! size: the WMMA GEMM loads full 16-row A-tiles, so a partial final tile
//! reads zeros instead of tripping bounds checks (stores are scalar and
//! bounds-checked, so nothing ever writes the slack).

use half::f16;

use super::GpuContext;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
}

impl Dtype {
    pub fn size(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 => 2,
        }
    }
}

pub struct GpuTensor {
    pub buffer: wgpu::Buffer,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

impl GpuTensor {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Logical payload bytes (allocation may be larger; see module docs).
    pub fn byte_len(&self) -> u64 {
        (self.len() * self.dtype.size()) as u64
    }

    /// Bytes per row of the last dimension.
    pub fn row_bytes(&self) -> u64 {
        (self.shape.last().unwrap() * self.dtype.size()) as u64
    }

    /// Reinterpret the shape (same element count).
    pub fn reshaped(self, shape: &[usize]) -> Self {
        assert_eq!(self.len(), shape.iter().product::<usize>(), "reshape element count");
        Self { buffer: self.buffer, shape: shape.to_vec(), dtype: self.dtype }
    }
}

fn padded_bytes(shape: &[usize], dtype: Dtype) -> u64 {
    let len: usize = shape.iter().product();
    let slack = 16 * shape.last().copied().unwrap_or(0);
    (((len + slack) * dtype.size()) as u64).next_multiple_of(4)
}

const USAGES: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);

impl GpuContext {
    /// Upload f32 data as `dtype` storage (converting to f16 if asked).
    pub fn tensor_from_f32(&self, shape: &[usize], data: &[f32], dtype: Dtype) -> GpuTensor {
        assert_eq!(data.len(), shape.iter().product::<usize>(), "shape/data mismatch");
        let bytes: Vec<u8> = match dtype {
            Dtype::F32 => bytemuck::cast_slice(data).to_vec(),
            Dtype::F16 => {
                let halves: Vec<f16> = data.iter().map(|&v| f16::from_f32(v)).collect();
                bytemuck::cast_slice(&halves).to_vec()
            }
        };
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: padded_bytes(shape, dtype),
            usage: USAGES,
            mapped_at_creation: true,
        });
        let mut view = buffer.slice(..).get_mapped_range_mut();
        view.slice(..bytes.len()).copy_from_slice(&bytes);
        drop(view);
        buffer.unmap();
        GpuTensor { buffer, shape: shape.to_vec(), dtype }
    }

    pub fn tensor_from_slice(&self, shape: &[usize], data: &[f32]) -> GpuTensor {
        self.tensor_from_f32(shape, data, Dtype::F32)
    }

    /// Upload raw f16 bits as an f16 tensor (weight loading).
    pub fn tensor_from_f16_bits(&self, shape: &[usize], data: &[f16]) -> GpuTensor {
        assert_eq!(data.len(), shape.iter().product::<usize>(), "shape/data mismatch");
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: padded_bytes(shape, Dtype::F16),
            usage: USAGES,
            mapped_at_creation: true,
        });
        let mut view = buffer.slice(..).get_mapped_range_mut();
        view.slice(..bytes.len()).copy_from_slice(bytes);
        drop(view);
        buffer.unmap();
        GpuTensor { buffer, shape: shape.to_vec(), dtype: Dtype::F16 }
    }

    /// Zero-initialized tensor (kernel output).
    pub fn empty_typed(&self, shape: &[usize], dtype: Dtype) -> GpuTensor {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: padded_bytes(shape, dtype),
            usage: USAGES,
            mapped_at_creation: false,
        });
        GpuTensor { buffer, shape: shape.to_vec(), dtype }
    }

    pub fn empty(&self, shape: &[usize]) -> GpuTensor {
        self.empty_typed(shape, Dtype::F32)
    }

    /// Copy `n_rows` rows (of the last dimension) between tensors with the
    /// same row width and dtype.
    pub fn copy_rows(
        &self,
        src: &GpuTensor,
        src_row: usize,
        dst: &GpuTensor,
        dst_row: usize,
        n_rows: usize,
    ) {
        assert_eq!(src.dtype, dst.dtype, "copy_rows dtype");
        assert_eq!(src.row_bytes(), dst.row_bytes(), "copy_rows row width");
        let row = src.row_bytes();
        self.copy(src, src_row as u64 * row, dst, dst_row as u64 * row, n_rows as u64 * row);
    }

    /// Flushes pending work, reads the tensor back as f32.
    pub fn download(&self, tensor: &GpuTensor) -> Vec<f32> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("download"),
            size: tensor.byte_len().next_multiple_of(4),
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
            encoder.copy_buffer_to_buffer(
                &tensor.buffer,
                0,
                &staging,
                0,
                tensor.byte_len().next_multiple_of(4),
            );
        }
        self.sync();
        staging.map_async(wgpu::MapMode::Read, .., |r| r.expect("map"));
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
        let data = staging.get_mapped_range(..);
        match tensor.dtype {
            Dtype::F32 => bytemuck::cast_slice(&data).to_vec(),
            Dtype::F16 => bytemuck::cast_slice::<u8, f16>(&data[..tensor.byte_len() as usize])
                .iter()
                .map(|v| v.to_f32())
                .collect(),
        }
    }
}
