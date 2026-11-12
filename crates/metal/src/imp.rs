//! macOS implementation (RED stub: compiles, returns wrong-but-valid results).

use linalg::abi::{alloc_shared, MetalContext};
use linalg_core::{row_major_strides, Backend, DType, DeviceKind, Error, Identified, Tensor};

#[derive(Clone)]
pub struct MetalStore {
    pub(crate) buffer: metal::Buffer,
}

impl Identified for MetalStore {
    fn storage_id(&self) -> u64 {
        self.buffer.contents() as u64
    }
}

pub struct MetalBackend {
    ctx: MetalContext,
}

impl MetalBackend {
    pub fn new() -> Result<MetalBackend, Error> {
        let ctx = MetalContext::new().map_err(|_| Error::UnsupportedDevice)?;
        Ok(MetalBackend { ctx })
    }
}

fn numel(shape: &[usize]) -> Result<usize, Error> {
    shape.iter().try_fold(1usize, |acc, &e| acc.checked_mul(e)).ok_or(Error::Shape)
}

impl Backend for MetalBackend {
    type Store = MetalStore;

    fn kind(&self) -> DeviceKind {
        DeviceKind::Metal
    }

    fn alloc(&self, dtype: DType, shape: &[usize]) -> Result<Tensor<MetalStore>, Error> {
        let bytes = numel(shape)?.checked_mul(dtype.size_of()).ok_or(Error::Shape)?;
        let buffer = alloc_shared::<u8>(&self.ctx, bytes);
        Ok(Tensor {
            storage: MetalStore { buffer },
            dtype,
            shape: shape.to_vec(),
            strides: row_major_strides(shape),
            offset: 0,
        })
    }

    fn upload(
        &self,
        dtype: DType,
        shape: &[usize],
        row_major: &[u8],
    ) -> Result<Tensor<MetalStore>, Error> {
        let tensor = self.alloc(dtype, shape)?;
        if row_major.len() != numel(shape)? * dtype.size_of() {
            return Err(Error::Dtype);
        }
        // SAFETY: the fresh buffer holds at least `row_major.len()` bytes and no GPU work uses it.
        unsafe {
            std::ptr::copy_nonoverlapping(
                row_major.as_ptr(),
                tensor.storage.buffer.contents().cast::<u8>(),
                row_major.len(),
            );
        }
        Ok(tensor)
    }

    fn download(&self, tensor: &Tensor<MetalStore>) -> Result<Vec<u8>, Error> {
        let len = numel(&tensor.shape)? * tensor.dtype.size_of();
        // SAFETY: the buffer is at least `len` bytes and no GPU work is in flight.
        Ok(unsafe {
            std::slice::from_raw_parts(tensor.storage.buffer.contents().cast::<u8>(), len).to_vec()
        })
    }

    fn gemm(
        &self,
        _a: &Tensor<MetalStore>,
        _b: &Tensor<MetalStore>,
        _c: &mut Tensor<MetalStore>,
        _alpha: f32,
        _beta: f32,
    ) -> Result<(), Error> {
        Ok(())
    }

    fn gelu(&self, _x: &Tensor<MetalStore>, _y: &mut Tensor<MetalStore>) -> Result<(), Error> {
        Ok(())
    }

    fn gelu_backward(
        &self,
        _pre: &Tensor<MetalStore>,
        _dy: &Tensor<MetalStore>,
        _dx: &mut Tensor<MetalStore>,
    ) -> Result<(), Error> {
        Ok(())
    }

    fn debug_allocs(&self) -> u64 {
        0
    }

    fn debug_launches(&self) -> Vec<&'static str> {
        Vec::new()
    }

    fn debug_downloads(&self) -> u64 {
        0
    }

    fn debug_uploads(&self) -> u64 {
        0
    }
}
