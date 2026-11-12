//! CUDA device backend (feature `cuda`).
//!
//! f32 gemm is one cuBLAS `sgemm` named `gemm-f32-cublas`. GELU and GELU' are
//! NVRTC kernels compiled once for `compute_61` in [`CudaBackend::new`].

use std::sync::Arc;

#[cfg(any(test, feature = "debug-counters"))]
use std::sync::Mutex;

use cudarc::cublas::{result as cublas_result, sys as cublas_sys, CudaBlas};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use linalg_core::{row_major_strides, Backend, DType, DeviceKind, Error, Identified, Tensor};

const GELU_SRC: &str = include_str!("../gelu.cu");

#[cfg(any(test, feature = "debug-counters"))]
struct Stats {
    allocs: u64,
    downloads: u64,
    uploads: u64,
    launches: Vec<&'static str>,
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
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    gelu: CudaFunction,
    gelu_backward: CudaFunction,
    /// Keeps the NVRTC module alive for the function handles.
    _module: Arc<CudaModule>,
    #[cfg(any(test, feature = "debug-counters"))]
    stats: Mutex<Stats>,
}

impl CudaBackend {
    pub fn new() -> Result<Self, Error> {
        let ctx = CudaContext::new(0).map_err(|_| Error::UnsupportedDevice)?;
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).map_err(|_| Error::Schedule)?;

        let ptx = compile_ptx_with_opts(
            GELU_SRC,
            CompileOptions { arch: Some("compute_61"), ..Default::default() },
        )
        .map_err(|_| Error::Schedule)?;
        let module = ctx.load_module(ptx).map_err(|_| Error::Schedule)?;
        let gelu = module.load_function("gelu_kernel").map_err(|_| Error::Schedule)?;
        let gelu_backward =
            module.load_function("gelu_backward_kernel").map_err(|_| Error::Schedule)?;

        Ok(Self {
            stream,
            blas,
            gelu,
            gelu_backward,
            _module: module,
            #[cfg(any(test, feature = "debug-counters"))]
            stats: Mutex::new(Stats { allocs: 0, downloads: 0, uploads: 0, launches: Vec::new() }),
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
        alpha: f32,
        beta: f32,
    ) -> Result<(), Error> {
        let (m, k) = matrix_dims(a)?;
        let (k_b, n) = matrix_dims(b)?;
        let (m_c, n_c) = matrix_dims(c)?;
        if k != k_b || m != m_c || n != n_c {
            return Err(Error::Shape);
        }
        if a.dtype != b.dtype || a.dtype != c.dtype {
            return Err(Error::Dtype);
        }

        let a_id = a.storage.storage_id();
        let b_id = b.storage.storage_id();
        let c_id = c.storage.storage_id();
        if a_id == b_id || a_id == c_id || b_id == c_id {
            return Err(Error::Alias);
        }

        let a_layout = matrix_layout(a)?;
        let b_layout = matrix_layout(b)?;
        let c_layout = matrix_layout(c)?;
        let supported = matches!(
            (a_layout, b_layout, c_layout),
            (MatrixLayout::RowMajor, MatrixLayout::RowMajor, MatrixLayout::RowMajor)
                | (MatrixLayout::Transposed, MatrixLayout::RowMajor, MatrixLayout::RowMajor)
                | (MatrixLayout::RowMajor, MatrixLayout::Transposed, MatrixLayout::RowMajor)
        );
        if !supported {
            return Err(Error::UnsupportedLayout);
        }

        match a.dtype {
            DType::F16 | DType::BF16 => {
                // Spec: f16 and bf16 launch nothing.
                return Ok(());
            }
            DType::F32 => {}
        }

        let c_bytes = m.checked_mul(n).and_then(|x| x.checked_mul(4)).ok_or(Error::Shape)?;
        if c_bytes > c.storage.buf.nbytes {
            return Err(Error::Shape);
        }
        if alpha != 0.0 && k > 0 {
            let a_bytes = m.checked_mul(k).and_then(|x| x.checked_mul(4)).ok_or(Error::Shape)?;
            let b_bytes = k.checked_mul(n).and_then(|x| x.checked_mul(4)).ok_or(Error::Shape)?;
            if a_bytes > a.storage.buf.nbytes || b_bytes > b.storage.buf.nbytes {
                return Err(Error::Shape);
            }
        }

        // K==0 with beta==0: cuBLAS may skip writing C. One device memset zeros C.
        if k == 0 && beta == 0.0 {
            if c_bytes > 0 {
                // SAFETY: exclusive write to `c` under Backend's caller obligation;
                // `ptr`/`c_bytes` describe the live allocation.
                unsafe {
                    cudarc::driver::result::memset_d8_async(
                        c.storage.buf.ptr,
                        0,
                        c_bytes,
                        self.stream.cu_stream(),
                    )
                }
                .map_err(|_| Error::Schedule)?;
            }
            self.record("gemm-f32-cublas");
            return Ok(());
        }

        // Row-major C = A B via column-major identity C^T = B^T A^T: one sgemm
        // with operands swapped. A transpose_matrix view is an OP_T flag.
        let (trans_b, ldb) = match b_layout {
            MatrixLayout::RowMajor => (cublas_sys::cublasOperation_t::CUBLAS_OP_N, n as i32),
            MatrixLayout::Transposed => (cublas_sys::cublasOperation_t::CUBLAS_OP_T, k as i32),
        };
        let (trans_a, lda) = match a_layout {
            MatrixLayout::RowMajor => (cublas_sys::cublasOperation_t::CUBLAS_OP_N, k as i32),
            MatrixLayout::Transposed => (cublas_sys::cublasOperation_t::CUBLAS_OP_T, m as i32),
        };

        // SAFETY: pointers are live device allocations for a/b/c; dims/ld match
        // the layouts above; Backend callers do not race writes on these ids.
        // cuBLAS with beta==0 does not read C (NaN-safe).
        unsafe {
            cublas_result::sgemm(
                *self.blas.handle(),
                trans_b,
                trans_a,
                n as i32,
                m as i32,
                k as i32,
                &alpha,
                b.storage.buf.ptr as *const f32,
                ldb,
                a.storage.buf.ptr as *const f32,
                lda,
                &beta,
                c.storage.buf.ptr as *mut f32,
                n as i32,
            )
        }
        .map_err(|_| Error::Schedule)?;

        self.record("gemm-f32-cublas");
        Ok(())
    }

    fn gelu(&self, x: &Tensor<Self::Store>, y: &mut Tensor<Self::Store>) -> Result<(), Error> {
        let n = prepare_f32_row_major(&[x, y])?;
        if x.storage.storage_id() == y.storage.storage_id() {
            return Err(Error::Alias);
        }
        let n_i32 = i32::try_from(n).map_err(|_| Error::Shape)?;
        let mut builder = self.stream.launch_builder(&self.gelu);
        builder.arg(&y.storage.buf.slice);
        builder.arg(&x.storage.buf.slice);
        builder.arg(&n_i32);
        let cfg = LaunchConfig::for_num_elems(n.max(1) as u32);
        // SAFETY: kernel writes `y` and reads `x`; lengths are `n` f32 elements;
        // Backend callers do not race these storage_ids.
        unsafe { builder.launch(cfg) }.map_err(|_| Error::Schedule)?;
        self.record("gelu-cuda");
        Ok(())
    }

    fn gelu_backward(
        &self,
        pre: &Tensor<Self::Store>,
        dy: &Tensor<Self::Store>,
        dx: &mut Tensor<Self::Store>,
    ) -> Result<(), Error> {
        let n = prepare_f32_row_major(&[pre, dy, dx])?;
        let pre_id = pre.storage.storage_id();
        let dy_id = dy.storage.storage_id();
        let dx_id = dx.storage.storage_id();
        if pre_id == dy_id || pre_id == dx_id || dy_id == dx_id {
            return Err(Error::Alias);
        }
        let n_i32 = i32::try_from(n).map_err(|_| Error::Shape)?;
        let mut builder = self.stream.launch_builder(&self.gelu_backward);
        builder.arg(&dx.storage.buf.slice);
        builder.arg(&pre.storage.buf.slice);
        builder.arg(&dy.storage.buf.slice);
        builder.arg(&n_i32);
        let cfg = LaunchConfig::for_num_elems(n.max(1) as u32);
        // SAFETY: kernel writes `dx` and reads `pre`/`dy`; lengths are `n` f32
        // elements; Backend callers do not race these storage_ids.
        unsafe { builder.launch(cfg) }.map_err(|_| Error::Schedule)?;
        self.record("gelu-backward-cuda");
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

fn matrix_dims(tensor: &Tensor<CudaStore>) -> Result<(usize, usize), Error> {
    if tensor.shape.len() != 2 || tensor.strides.len() != 2 {
        return Err(Error::Shape);
    }
    Ok((tensor.shape[0], tensor.shape[1]))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MatrixLayout {
    RowMajor,
    Transposed,
}

fn matrix_layout(tensor: &Tensor<CudaStore>) -> Result<MatrixLayout, Error> {
    if tensor.offset != 0 {
        return Err(Error::UnsupportedLayout);
    }
    if tensor.strides == row_major_strides(&tensor.shape) {
        return Ok(MatrixLayout::RowMajor);
    }
    // `transpose_matrix` of row-major `[cols, rows]` yields strides `[1, rows]`.
    if tensor.strides[0] == 1 && tensor.strides[1] == tensor.shape[0] {
        return Ok(MatrixLayout::Transposed);
    }
    Err(Error::UnsupportedLayout)
}

fn prepare_f32_row_major(tensors: &[&Tensor<CudaStore>]) -> Result<usize, Error> {
    for tensor in tensors {
        if tensor.dtype != DType::F32 {
            return Err(Error::Dtype);
        }
    }
    let shape = &tensors[0].shape;
    for tensor in tensors.iter().skip(1) {
        if tensor.shape != *shape {
            return Err(Error::Shape);
        }
    }
    let packed = row_major_strides(shape);
    for tensor in tensors {
        if tensor.offset != 0 || tensor.strides != packed {
            return Err(Error::UnsupportedLayout);
        }
    }
    let count = {
        let mut n = 1usize;
        for &extent in shape.iter() {
            n = n.checked_mul(extent).ok_or(Error::Shape)?;
        }
        n
    };
    let bytes = count.checked_mul(DType::F32.size_of()).ok_or(Error::Shape)?;
    for tensor in tensors {
        if tensor.storage.buf.nbytes < bytes {
            return Err(Error::Shape);
        }
    }
    Ok(count)
}
