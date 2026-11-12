//! CUDA backend for the device-independent linalg contract.
//!
//! Enabled with the `cuda` feature. Without it, or when no device is present,
//! [`CudaBackend::new`] returns [`Error::UnsupportedDevice`].

#[cfg(feature = "cuda")]
mod device;

#[cfg(feature = "cuda")]
pub use device::{CudaBackend, CudaStore};

#[cfg(not(feature = "cuda"))]
mod host_stub {
    use linalg_core::{Backend, DType, DeviceKind, Error, Identified, Tensor};

    /// Placeholder store when the `cuda` feature is off.
    #[derive(Clone)]
    pub struct CudaStore {
        id: u64,
    }

    impl Identified for CudaStore {
        fn storage_id(&self) -> u64 {
            self.id
        }
    }

    /// Backend type present in all builds. [`new`](CudaBackend::new) always fails
    /// without the `cuda` feature.
    pub struct CudaBackend;

    impl CudaBackend {
        pub fn new() -> Result<Self, Error> {
            Err(Error::UnsupportedDevice)
        }
    }

    impl Backend for CudaBackend {
        type Store = CudaStore;

        fn kind(&self) -> DeviceKind {
            DeviceKind::Cuda
        }

        fn alloc(&self, _dtype: DType, _shape: &[usize]) -> Result<Tensor<Self::Store>, Error> {
            Err(Error::UnsupportedDevice)
        }

        fn upload(
            &self,
            _dtype: DType,
            _shape: &[usize],
            _row_major: &[u8],
        ) -> Result<Tensor<Self::Store>, Error> {
            Err(Error::UnsupportedDevice)
        }

        fn download(&self, _tensor: &Tensor<Self::Store>) -> Result<Vec<u8>, Error> {
            Err(Error::UnsupportedDevice)
        }

        fn gemm(
            &self,
            _a: &Tensor<Self::Store>,
            _b: &Tensor<Self::Store>,
            _c: &mut Tensor<Self::Store>,
            _alpha: f32,
            _beta: f32,
        ) -> Result<(), Error> {
            Err(Error::UnsupportedDevice)
        }

        fn gelu(
            &self,
            _x: &Tensor<Self::Store>,
            _y: &mut Tensor<Self::Store>,
        ) -> Result<(), Error> {
            Err(Error::UnsupportedDevice)
        }

        fn gelu_backward(
            &self,
            _pre: &Tensor<Self::Store>,
            _dy: &Tensor<Self::Store>,
            _dx: &mut Tensor<Self::Store>,
        ) -> Result<(), Error> {
            Err(Error::UnsupportedDevice)
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
}

#[cfg(not(feature = "cuda"))]
pub use host_stub::{CudaBackend, CudaStore};

#[cfg(test)]
mod invariants;
