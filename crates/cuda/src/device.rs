//! CUDA device backend (feature `cuda`).
//!
//! RED stub: alloc/upload/download talk to the device; gemm/gelu are intentional
//! no-ops that return `Ok` so tests fail on value and counter assertions.

use std::sync::Arc;

#[cfg(any(test, feature = "debug-counters"))]
use std::sync::Mutex;

use cudarc::driver::{CudaContext, CudaSlice, DevicePtr};
use linalg_core::{row_major_strides, Backend, DType, DeviceKind, Error, Identified, Tensor};

#[cfg(any(test, feature = "debug-counters"))]
struct Stats {
    allocs: u64,
    downloads: u64,
    uploads: u64,
    launches: Vec<&'static str>,
    /// Deliberately unstable write pattern for RED `repeated_gemm_is_bitwise_stable`.
    gemm_calls: u64,
}

struct DeviceBuf {
    /// Owned device allocation. Kept for drop; address is mirrored in `ptr`.
    slice: CudaSlice<u8>,
    /// Device pointer value used as [`Identified::storage_id`].
    ptr: u64,
    /// Logical byte length of tensor payload (may be 0 while `slice` holds 1 byte).
    nbytes: usize,
}

// SAFETY: `DeviceBuf` uniquely owns one `CudaSlice`. Moving it moves that owner.
// No method on `&DeviceBuf` forms a Rust reference into device memory.
unsafe impl Send for DeviceBuf {}

// SAFETY: Sharing is sound because core's Backend obligation says no thread may
// read or write a storage_id while another call writes that same storage_id.
unsafe impl Sync for DeviceBuf {}

/// Device storage. [`Clone`] only bumps the `Arc` refcount (no device copy).
#[derive(Clone)]
pub struct CudaStore {
    buf: Arc<DeviceBuf>,
}

impl Identified for CudaStore {
    fn storage_id(&self) -> u64 {
        self.buf.ptr
    }
}

/// CUDA backend. Construct with [`CudaBackend::new`].
pub struct CudaBackend {
    stream: Arc<cudarc::driver::CudaStream>,
    #[cfg(any(test, feature = "debug-counters"))]
    stats: Mutex<Stats>,
}

impl CudaBackend {
    pub fn new() -> Result<Self, Error> {
        let ctx = CudaContext::new(0).map_err(|_| Error::UnsupportedDevice)?;
        let stream = ctx.default_stream();
        Ok(Self {
            stream,
            #[cfg(any(test, feature = "debug-counters"))]
            stats: Mutex::new(Stats {
                allocs: 0,
                downloads: 0,
                uploads: 0,
                launches: Vec::new(),
                gemm_calls: 0,
            }),
        })
    }

    fn bump_allocs(&self) {
        #[cfg(any(test, feature = "debug-counters"))]
        {
            self.stats.lock().expect("cuda stats").allocs += 1;
        }
    }

    fn bump_uploads(&self) {
        #[cfg(any(test, feature = "debug-counters"))]
        {
            self.stats.lock().expect("cuda stats").uploads += 1;
        }
    }

    fn bump_downloads(&self) {
        #[cfg(any(test, feature = "debug-counters"))]
        {
            self.stats.lock().expect("cuda stats").downloads += 1;
        }
    }

    fn record(&self, name: &'static str) {
        #[cfg(any(test, feature = "debug-counters"))]
        {
            self.stats.lock().expect("cuda stats").launches.push(name);
        }
        #[cfg(not(any(test, feature = "debug-counters")))]
        {
            let _ = name;
        }
    }

    fn make_store(&self, slice: CudaSlice<u8>, nbytes: usize) -> Result<CudaStore, Error> {
        let ptr = {
            let (ptr, sync) = slice.device_ptr(&self.stream);
            drop(sync);
            ptr
        };
        if ptr == 0 {
            return Err(Error::Schedule);
        }
        Ok(CudaStore { buf: Arc::new(DeviceBuf { slice, ptr: ptr as u64, nbytes }) })
    }
}

impl Backend for CudaBackend {
    type Store = CudaStore;

    fn kind(&self) -> DeviceKind {
        DeviceKind::Cuda
    }

    fn alloc(&self, dtype: DType, shape: &[usize]) -> Result<Tensor<Self::Store>, Error> {
        let nbytes = checked_nbytes(dtype, shape)?;
        let slice = self.stream.alloc_zeros::<u8>(nbytes.max(1)).map_err(|_| Error::Schedule)?;
        let storage = self.make_store(slice, nbytes)?;
        self.bump_allocs();
        Ok(Tensor {
            storage,
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
    ) -> Result<Tensor<Self::Store>, Error> {
        let nbytes = checked_nbytes(dtype, shape)?;
        if row_major.len() != nbytes {
            return Err(Error::Dtype);
        }
        let alloc_len = nbytes.max(1);
        let mut slice =
            unsafe { self.stream.alloc::<u8>(alloc_len) }.map_err(|_| Error::Schedule)?;
        if nbytes > 0 {
            self.stream.memcpy_htod(row_major, &mut slice).map_err(|_| Error::Schedule)?;
        } else {
            self.stream.memset_zeros(&mut slice).map_err(|_| Error::Schedule)?;
        }
        let storage = self.make_store(slice, nbytes)?;
        self.bump_uploads();
        Ok(Tensor {
            storage,
            dtype,
            shape: shape.to_vec(),
            strides: row_major_strides(shape),
            offset: 0,
        })
    }

    fn download(&self, tensor: &Tensor<Self::Store>) -> Result<Vec<u8>, Error> {
        if tensor.strides != row_major_strides(&tensor.shape) {
            return Err(Error::UnsupportedLayout);
        }
        let nbytes = checked_nbytes(tensor.dtype, &tensor.shape)?;
        if nbytes != tensor.storage.buf.nbytes {
            return Err(Error::Shape);
        }
        if nbytes == 0 {
            self.bump_downloads();
            return Ok(Vec::new());
        }
        let mut out = vec![0u8; nbytes];
        let view = tensor.storage.buf.slice.slice(0..nbytes);
        self.stream.memcpy_dtoh(&view, &mut out[..]).map_err(|_| Error::Schedule)?;
        self.stream.synchronize().map_err(|_| Error::Schedule)?;
        self.bump_downloads();
        Ok(out)
    }

    fn gemm(
        &self,
        a: &Tensor<Self::Store>,
        b: &Tensor<Self::Store>,
        c: &mut Tensor<Self::Store>,
        _alpha: f32,
        _beta: f32,
    ) -> Result<(), Error> {
        let _ = (a, b);
        // RED: wrong launch name, extra alloc bump, byte-fill that changes per call.
        self.bump_allocs();
        self.record("gemm-stub");
        #[cfg(any(test, feature = "debug-counters"))]
        let pattern = {
            let mut stats = self.stats.lock().expect("cuda stats");
            stats.gemm_calls += 1;
            (stats.gemm_calls & 0xff) as u8
        };
        #[cfg(not(any(test, feature = "debug-counters")))]
        let pattern = 0xabu8;
        let nbytes = c.storage.buf.nbytes;
        if nbytes > 0 {
            // SAFETY: writes `c`'s device bytes through its stored pointer. Valid
            // because Backend callers never race another write on this storage_id,
            // and `ptr`/`nbytes` describe the live allocation in `c.storage`.
            unsafe {
                cudarc::driver::result::memset_d8_async(
                    c.storage.buf.ptr,
                    pattern,
                    nbytes,
                    self.stream.cu_stream(),
                )
            }
            .map_err(|_| Error::Schedule)?;
            self.stream.synchronize().map_err(|_| Error::Schedule)?;
        }
        Ok(())
    }

    fn gelu(&self, x: &Tensor<Self::Store>, y: &mut Tensor<Self::Store>) -> Result<(), Error> {
        let _ = (x, y);
        // RED: pretend to download and do not write y.
        self.bump_downloads();
        self.record("gelu-stub");
        Ok(())
    }

    fn gelu_backward(
        &self,
        pre: &Tensor<Self::Store>,
        dy: &Tensor<Self::Store>,
        dx: &mut Tensor<Self::Store>,
    ) -> Result<(), Error> {
        let _ = (pre, dy, dx);
        self.record("gelu-backward-stub");
        Ok(())
    }

    fn debug_allocs(&self) -> u64 {
        #[cfg(any(test, feature = "debug-counters"))]
        {
            self.stats.lock().expect("cuda stats").allocs
        }
        #[cfg(not(any(test, feature = "debug-counters")))]
        {
            let _ = self;
            0
        }
    }

    fn debug_launches(&self) -> Vec<&'static str> {
        #[cfg(any(test, feature = "debug-counters"))]
        {
            self.stats.lock().expect("cuda stats").launches.clone()
        }
        #[cfg(not(any(test, feature = "debug-counters")))]
        {
            let _ = self;
            Vec::new()
        }
    }

    fn debug_downloads(&self) -> u64 {
        #[cfg(any(test, feature = "debug-counters"))]
        {
            self.stats.lock().expect("cuda stats").downloads
        }
        #[cfg(not(any(test, feature = "debug-counters")))]
        {
            let _ = self;
            0
        }
    }

    fn debug_uploads(&self) -> u64 {
        #[cfg(any(test, feature = "debug-counters"))]
        {
            self.stats.lock().expect("cuda stats").uploads
        }
        #[cfg(not(any(test, feature = "debug-counters")))]
        {
            let _ = self;
            0
        }
    }
}

fn checked_nbytes(dtype: DType, shape: &[usize]) -> Result<usize, Error> {
    let mut numel = 1usize;
    for &extent in shape {
        numel = numel.checked_mul(extent).ok_or(Error::Shape)?;
    }
    numel.checked_mul(dtype.size_of()).ok_or(Error::Shape)
}
