//! CUDA device backend (feature `cuda`).
//!
//! f32 gemm is one cuBLAS `sgemm` named `gemm-f32-cublas`. GELU and GELU' are
//! NVRTC kernels compiled once for `compute_61` in [`CudaBackend::new`].
//!
//! gemm, gelu and gelu_backward take no lock and perform no atomic. cudarc's safe layer
//! does both on these paths (`CudaContext::bind_to_thread` swaps the context's error word,
//! `PushKernelArg` loads its stream count, and every `sys::*` wrapper reads a `OnceLock`), so
//! the paths call the driver and cuBLAS entry points resolved once in [`CudaBackend::new`].
//!
//! The training ops (`add_bias`, `bias_grad`, `mse_grad`, `sgd_step`, `convert`) are NVRTC
//! kernels from `train.cu`, compiled with the GELU kernels. Every op is stream-ordered on the
//! backend's one stream; only `sync`, `download`, `download_into`, and `mse_grad` synchronize.
//!
//! Device buffers come from a size-keyed free list ([`FreeList`]): a `Store` drop returns its
//! buffer, and `alloc` and `upload` reuse one of the same size before they call the allocator.
//! `download_into` and `upload_into` copy through pinned host staging that grows to the largest
//! transfer and is otherwise allocated once.
//!
//! cuBLAS runs in `CUBLAS_DEFAULT_MATH`: f32 sgemm stays f32, with no TF32 tensor-op math,
//! matching PyTorch's f32 path with TF32 disabled.
use std::ffi::{c_void, CString};
use std::mem::ManuallyDrop;
use std::sync::Arc;

#[cfg(any(test, feature = "debug-counters"))]
use std::sync::Mutex;

use cudarc::cublas::{sys as cublas_sys, CudaBlas};
use cudarc::driver::{
    result as driver_result, sys as driver_sys, CudaContext, CudaSlice, CudaStream, DevicePtr,
    LaunchConfig,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use linalg_core::{row_major_strides, Backend, DType, DeviceKind, Error, Identified, Tensor};

use crate::shared_state::{Exclusive, FreeList};

const GELU_SRC: &str = include_str!("../gelu.cu");
const TRAIN_SRC: &str = include_str!("../train.cu");

/// Threads in `mse_grad_kernel`'s one block; `MSE_THREADS` in `train.cu` must match.
const MSE_THREADS: u32 = 1024;

/// Threads per block of the elementwise and per-column kernels.
const BLOCK: u32 = 256;

#[cfg(any(test, feature = "debug-counters"))]
struct Stats {
    allocs: u64,
    downloads: u64,
    uploads: u64,
    launches: Vec<&'static str>,
    /// Device allocator calls (`cuMemAlloc`/`cuMemAllocAsync`): free-list misses.
    mem_allocs: u64,
    /// Copies between device memory and host memory made by `download_into`/`upload_into`.
    device_host_copies: u64,
    /// Pinned staging allocations.
    pinned_allocs: u64,
}

/// The counters beyond the four the contract reports.
#[derive(Clone, Copy)]
enum Counter {
    MemAllocs,
    DeviceHostCopies,
    PinnedAllocs,
}

type Pool = FreeList<CudaSlice<u8>>;

struct DeviceBuf {
    /// Owned device allocation, returned to `pool` on drop; its address is mirrored in `ptr`.
    slice: ManuallyDrop<CudaSlice<u8>>,
    /// Device pointer value used as [`Identified::storage_id`].
    ptr: u64,
    /// Logical byte length of tensor payload (may be 0 while `slice` holds 1 byte).
    nbytes: usize,
    /// The free list this buffer returns to.
    pool: Arc<Pool>,
}

impl Drop for DeviceBuf {
    fn drop(&mut self) {
        // SAFETY: `slice` is taken once, here, and never used again. Work already queued on the
        // backend's one stream that reads or writes it runs before any later reuse.
        let slice = unsafe { ManuallyDrop::take(&mut self.slice) };
        self.pool.put(slice.len(), slice);
    }
}

// SAFETY: `DeviceBuf` uniquely owns one `CudaSlice`. Moving it moves that owner.
// No method on `&DeviceBuf` forms a Rust reference into device memory. Core's
// caller obligation: no thread reads or writes a storage_id while another call
// writes it.
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

/// The GELU module, loaded once, and the raw handles the compute paths launch.
struct Kernels {
    driver: &'static driver_sys::Lib,
    ctx: driver_sys::CUcontext,
    module: driver_sys::CUmodule,
    gelu: driver_sys::CUfunction,
    gelu_backward: driver_sys::CUfunction,
    add_bias: driver_sys::CUfunction,
    bias_grad: driver_sys::CUfunction,
    mse_grad: driver_sys::CUfunction,
    sgd_step: driver_sys::CUfunction,
    to_bf16: driver_sys::CUfunction,
    to_f32: driver_sys::CUfunction,
}

// SAFETY: the handles are written once in `CudaBackend::new` and only read afterwards. The
// driver API accepts a context, module and function from any thread. Kernels launched through
// them touch only the tensors passed in, and core's caller obligation says no thread reads or
// writes a storage_id while another call writes it.
unsafe impl Send for Kernels {}

// SAFETY: as for `Send`: `&Kernels` exposes only immutable handle values, and concurrent
// launches are governed by core's caller obligation on storage_ids.
unsafe impl Sync for Kernels {}

impl Drop for Kernels {
    fn drop(&mut self) {
        // SAFETY: `module` was loaded into `ctx`, which `CudaBackend::stream` keeps alive until
        // after this field is dropped; nothing launches from it once the backend is dropped.
        unsafe {
            let _ = (self.driver.cuCtxSetCurrent)(self.ctx);
            let _ = (self.driver.cuModuleUnload)(self.module);
        }
    }
}

/// Page-locked host memory from `cuMemHostAlloc`.
struct Pinned {
    ptr: *mut u8,
    cap: usize,
}

/// Pinned staging for transfers and the loss, plus the event that marks when the last
/// `upload_into` has finished reading `up`.
struct Staging {
    driver: &'static driver_sys::Lib,
    ctx: driver_sys::CUcontext,
    down: Pinned,
    up: Pinned,
    up_done: driver_sys::CUevent,
    /// Owns the device `f32` at `loss_ptr` that `mse_grad_kernel` writes the loss into.
    _loss: CudaSlice<u8>,
    loss_ptr: u64,
}

// SAFETY: the pinned pointers and the event are owned by `Staging` and used only under the
// `Exclusive` lock; the driver accepts them from any thread with the context current.
unsafe impl Send for Staging {}

impl Staging {
    /// `which`, grown to at least `len` bytes. Growing frees the old block, after `up_done`
    /// says no copy still reads it. Returns whether it allocated.
    fn ensure(&mut self, up: bool, len: usize) -> Result<bool, Error> {
        let driver = self.driver;
        let done = self.up_done;
        let block = if up { &mut self.up } else { &mut self.down };
        if block.cap >= len {
            return Ok(false);
        }
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: the context is current (callers bind first); `ptr` is a valid out-pointer.
        // The old block is idle: the up block once `up_done` has fired, the down block because
        // every download synchronizes before it returns.
        unsafe {
            if up && (driver.cuEventSynchronize)(done) != driver_sys::CUresult::CUDA_SUCCESS {
                return Err(Error::Schedule);
            }
            if (driver.cuMemHostAlloc)(&mut ptr, len, 0) != driver_sys::CUresult::CUDA_SUCCESS {
                return Err(Error::Schedule);
            }
            if !block.ptr.is_null() {
                let _ = (driver.cuMemFreeHost)(block.ptr.cast());
            }
        }
        *block = Pinned { ptr: ptr.cast(), cap: len };
        Ok(true)
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        // SAFETY: the context outlives this (the backend's stream holds it); nothing else uses
        // these blocks or the event once the backend is gone.
        unsafe {
            let _ = (self.driver.cuCtxSetCurrent)(self.ctx);
            let _ = (self.driver.cuEventSynchronize)(self.up_done);
            for block in [&self.down, &self.up] {
                if !block.ptr.is_null() {
                    let _ = (self.driver.cuMemFreeHost)(block.ptr.cast());
                }
            }
            let _ = (self.driver.cuEventDestroy_v2)(self.up_done);
        }
    }
}

/// CUDA backend. Construct with [`CudaBackend::new`].
pub struct CudaBackend {
    /// Declared first, so the module unloads while `stream` still holds the context.
    kernels: Kernels,
    /// Pinned staging and the loss buffer; dropped while `stream` still holds the context.
    staging: Exclusive<Staging>,
    /// Device buffers returned by dropped stores.
    pool: Arc<Pool>,
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    /// cuBLAS entry points, resolved once.
    cublas: &'static cublas_sys::Lib,
    #[cfg(any(test, feature = "debug-counters"))]
    stats: Mutex<Stats>,
}

/// True when the CUDA driver, cuBLAS, and NVRTC libraries can be loaded.
pub(crate) fn cuda_libs_present() -> bool {
    // SAFETY: `is_culib_present` only probes `libloading`; it does not call into CUDA.
    unsafe {
        cudarc::driver::sys::is_culib_present()
            && cudarc::cublas::sys::is_culib_present()
            && cudarc::nvrtc::sys::is_culib_present()
    }
}

/// Maps a panicking CUDA init to [`Error::UnsupportedDevice`].
pub(crate) fn map_cuda_init_panic<T>(
    caught: Result<Result<T, Error>, Box<dyn std::any::Any + Send>>,
) -> Result<T, Error> {
    match caught {
        Ok(inner) => inner,
        Err(_) => Err(Error::UnsupportedDevice),
    }
}

impl CudaBackend {
    pub fn new() -> Result<Self, Error> {
        #[cfg(windows)]
        windows_dll_search::add_cuda_bin();
        if !cuda_libs_present() {
            return Err(Error::UnsupportedDevice);
        }
        map_cuda_init_panic(std::panic::catch_unwind(std::panic::AssertUnwindSafe(Self::new_inner)))
    }

    fn new_inner() -> Result<Self, Error> {
        let ctx = CudaContext::new(0).map_err(|_| Error::UnsupportedDevice)?;
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).map_err(|_| Error::UnsupportedDevice)?;

        // SAFETY: `cublasSetMathMode` takes the live handle `CudaBlas::new` created.
        let math = unsafe {
            cublas_sys::cublasSetMathMode(
                *blas.handle(),
                cublas_sys::cublasMath_t::CUBLAS_DEFAULT_MATH,
            )
        };
        if math != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(Error::UnsupportedDevice);
        }

        let source = format!("{GELU_SRC}\n{TRAIN_SRC}");
        let ptx = compile_ptx_with_opts(
            source,
            CompileOptions { arch: Some("compute_61"), ..Default::default() },
        )
        .map_err(|_| Error::UnsupportedDevice)?;
        // SAFETY: `new` checked that both libraries load; `culib` loads them once and returns
        // the process-wide entry-point tables.
        let (driver, cublas) = unsafe { (driver_sys::culib(), cublas_sys::culib()) };
        let image = CString::new(ptx.to_src()).map_err(|_| Error::UnsupportedDevice)?;
        // SAFETY: `CudaContext::new` made `ctx` current on this thread; `image` is a
        // NUL-terminated PTX image that outlives the call.
        let module = unsafe { driver_result::module::load_data(image.as_ptr().cast()) }
            .map_err(|_| Error::UnsupportedDevice)?;
        let mut kernels = Kernels {
            driver,
            ctx: ctx.cu_ctx(),
            module,
            gelu: std::ptr::null_mut(),
            gelu_backward: std::ptr::null_mut(),
            add_bias: std::ptr::null_mut(),
            bias_grad: std::ptr::null_mut(),
            mse_grad: std::ptr::null_mut(),
            sgd_step: std::ptr::null_mut(),
            to_bf16: std::ptr::null_mut(),
            to_f32: std::ptr::null_mut(),
        };
        let function = |name: &str| {
            let name = CString::new(name).map_err(|_| Error::UnsupportedDevice)?;
            // SAFETY: `module` is loaded in the current context.
            unsafe { driver_result::module::get_function(module, name) }
                .map_err(|_| Error::UnsupportedDevice)
        };
        kernels.gelu = function("gelu_kernel")?;
        kernels.gelu_backward = function("gelu_backward_kernel")?;
        kernels.add_bias = function("add_bias_kernel")?;
        kernels.bias_grad = function("bias_grad_kernel")?;
        kernels.mse_grad = function("mse_grad_kernel")?;
        kernels.sgd_step = function("sgd_step_kernel")?;
        kernels.to_bf16 = function("convert_f32_bf16_kernel")?;
        kernels.to_f32 = function("convert_bf16_f32_kernel")?;

        let loss = stream.alloc_zeros::<u8>(4).map_err(|_| Error::UnsupportedDevice)?;
        let loss_ptr = {
            let (ptr, sync) = loss.device_ptr(&stream);
            drop(sync);
            ptr
        };
        let mut up_done: driver_sys::CUevent = std::ptr::null_mut();
        // SAFETY: the context is current; `up_done` is a valid out-pointer.
        let event = unsafe {
            (driver.cuEventCreate)(
                &mut up_done,
                driver_sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32,
            )
        };
        if event != driver_sys::CUresult::CUDA_SUCCESS {
            return Err(Error::UnsupportedDevice);
        }
        // Recorded once, so the first `upload_into` has an event to wait on.
        // SAFETY: `up_done` was just created in this context; the stream is live.
        unsafe {
            let _ = (driver.cuEventRecord)(up_done, stream.cu_stream());
        }
        let staging = Staging {
            driver,
            ctx: ctx.cu_ctx(),
            down: Pinned { ptr: std::ptr::null_mut(), cap: 0 },
            up: Pinned { ptr: std::ptr::null_mut(), cap: 0 },
            up_done,
            _loss: loss,
            loss_ptr,
        };

        Ok(Self {
            kernels,
            staging: Exclusive::new(staging),
            pool: Arc::new(FreeList::new()),
            stream,
            blas,
            cublas,
            #[cfg(any(test, feature = "debug-counters"))]
            stats: Mutex::new(Stats {
                allocs: 0,
                downloads: 0,
                uploads: 0,
                launches: Vec::new(),
                mem_allocs: 0,
                device_host_copies: 0,
                pinned_allocs: 0,
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

    fn bump(&self, counter: Counter) {
        #[cfg(any(test, feature = "debug-counters"))]
        {
            let mut stats = self.stats.lock().expect("cuda stats");
            match counter {
                Counter::MemAllocs => stats.mem_allocs += 1,
                Counter::DeviceHostCopies => stats.device_host_copies += 1,
                Counter::PinnedAllocs => stats.pinned_allocs += 1,
            }
        }
        #[cfg(not(any(test, feature = "debug-counters")))]
        {
            let _ = counter;
        }
    }

    /// Device allocator calls so far: `alloc` and `upload` misses on the free list (test and
    /// `debug-counters` builds).
    #[cfg(any(test, feature = "debug-counters"))]
    pub fn debug_mem_allocs(&self) -> u64 {
        self.stats.lock().expect("cuda stats").mem_allocs
    }

    /// Device-to-host and host-to-device copies made by `download_into` and `upload_into` so
    /// far (test and `debug-counters` builds).
    #[cfg(any(test, feature = "debug-counters"))]
    pub fn debug_device_host_copies(&self) -> u64 {
        self.stats.lock().expect("cuda stats").device_host_copies
    }

    /// Pinned staging allocations so far (test and `debug-counters` builds).
    #[cfg(any(test, feature = "debug-counters"))]
    pub fn debug_pinned_allocs(&self) -> u64 {
        self.stats.lock().expect("cuda stats").pinned_allocs
    }

    /// The cuBLAS math mode the handle runs in.
    #[cfg(test)]
    pub(crate) fn math_mode_for_test(&self) -> cublas_sys::cublasMath_t {
        let mut mode = cublas_sys::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH;
        // SAFETY: the handle is live; `mode` is a valid out-pointer.
        let status = unsafe { (self.cublas.cublasGetMathMode)(*self.blas.handle(), &mut mode) };
        assert_eq!(status, cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        mode
    }

    /// A zero-filled device buffer of `max(nbytes, 1)` bytes: a free one of that size if a
    /// dropped store returned one, else a new allocation.
    fn device_buffer(&self, nbytes: usize, zero: bool) -> Result<CudaSlice<u8>, Error> {
        let len = nbytes.max(1);
        if let Some(mut slice) = self.pool.take(len) {
            if zero {
                self.stream.memset_zeros(&mut slice).map_err(|_| Error::Schedule)?;
            }
            return Ok(slice);
        }
        self.bump(Counter::MemAllocs);
        if zero {
            self.stream.alloc_zeros::<u8>(len).map_err(|_| Error::Schedule)
        } else {
            // SAFETY: the caller overwrites every byte before any read.
            unsafe { self.stream.alloc::<u8>(len) }.map_err(|_| Error::Schedule)
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

    /// Makes the backend's context current on this thread. Worker threads that did not run
    /// `new` need it before cuBLAS or a launch. Two driver calls; no cudarc wrapper.
    fn bind(&self) -> Result<(), Error> {
        let driver = self.kernels.driver;
        let mut current: driver_sys::CUcontext = std::ptr::null_mut();
        // SAFETY: `current` is a valid out-pointer; `kernels.ctx` is the live primary context.
        unsafe {
            if (driver.cuCtxGetCurrent)(&mut current) != driver_sys::CUresult::CUDA_SUCCESS {
                return Err(Error::Schedule);
            }
            if current != self.kernels.ctx
                && (driver.cuCtxSetCurrent)(self.kernels.ctx) != driver_sys::CUresult::CUDA_SUCCESS
            {
                return Err(Error::Schedule);
            }
        }
        Ok(())
    }

    /// Launches `function` over `n` elements on the backend's stream with `params`.
    fn launch(
        &self,
        function: driver_sys::CUfunction,
        n: usize,
        params: &mut [*mut c_void],
    ) -> Result<(), Error> {
        let cfg = LaunchConfig::for_num_elems(n.max(1) as u32);
        self.launch_with(function, cfg, params)
    }

    /// `ceil(n / BLOCK)` blocks of `BLOCK` threads.
    fn blocks(n: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (n.div_ceil(BLOCK), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Launches `function` with `cfg` on the backend's stream with `params`.
    fn launch_with(
        &self,
        function: driver_sys::CUfunction,
        cfg: LaunchConfig,
        params: &mut [*mut c_void],
    ) -> Result<(), Error> {
        self.bind()?;
        // SAFETY: `function` belongs to the module loaded in this context, and `params` holds
        // one pointer per kernel parameter, each to a live value of the parameter's type.
        let status = unsafe {
            (self.kernels.driver.cuLaunchKernel)(
                function,
                cfg.grid_dim.0,
                cfg.grid_dim.1,
                cfg.grid_dim.2,
                cfg.block_dim.0,
                cfg.block_dim.1,
                cfg.block_dim.2,
                cfg.shared_mem_bytes,
                self.stream.cu_stream(),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if status != driver_sys::CUresult::CUDA_SUCCESS {
            return Err(Error::Schedule);
        }
        Ok(())
    }

    fn make_store(&self, slice: CudaSlice<u8>, nbytes: usize) -> Result<CudaStore, Error> {
        let ptr = {
            let (ptr, sync) = slice.device_ptr(&self.stream);
            drop(sync);
            ptr
        };
        if ptr == 0 {
            self.pool.put(slice.len(), slice);
            return Err(Error::Schedule);
        }
        Ok(CudaStore {
            buf: Arc::new(DeviceBuf {
                slice: ManuallyDrop::new(slice),
                ptr: ptr as u64,
                nbytes,
                pool: Arc::clone(&self.pool),
            }),
        })
    }

    /// Checks a transfer of `len` bytes against `tensor`: `Shape` for a length mismatch, then
    /// `UnsupportedLayout`, then `Shape` if the buffer is short. Allocates nothing.
    fn transfer_bytes(tensor: &Tensor<CudaStore>, len: usize) -> Result<usize, Error> {
        let nbytes = checked_nbytes(tensor.dtype, &tensor.shape)?;
        if len != nbytes {
            return Err(Error::Shape);
        }
        if !packed_layout(tensor) {
            return Err(Error::UnsupportedLayout);
        }
        if nbytes > tensor.storage.buf.nbytes {
            return Err(Error::Shape);
        }
        Ok(nbytes)
    }

    fn sync_stream(&self) -> Result<(), Error> {
        // SAFETY: the stream belongs to the backend's live context.
        let status = unsafe { (self.kernels.driver.cuStreamSynchronize)(self.stream.cu_stream()) };
        if status != driver_sys::CUresult::CUDA_SUCCESS {
            return Err(Error::Schedule);
        }
        Ok(())
    }
}

#[cfg(test)]
impl CudaBackend {
    /// The context whose deferred-error atomic (`error_state`, read and cleared by cudarc's
    /// `check_err` inside `bind_to_thread`) the compute paths must leave untouched.
    pub(crate) fn context_for_test(&self) -> &Arc<CudaContext> {
        self.stream.context()
    }
}

impl Backend for CudaBackend {
    type Store = CudaStore;

    fn kind(&self) -> DeviceKind {
        DeviceKind::Cuda
    }

    fn alloc(&self, dtype: DType, shape: &[usize]) -> Result<Tensor<Self::Store>, Error> {
        let nbytes = checked_nbytes(dtype, shape)?;
        self.bind()?;
        let slice = self.device_buffer(nbytes, true)?;
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
        self.bind()?;
        // Every byte is written below: the payload, or zeros for an empty tensor.
        let mut slice = self.device_buffer(nbytes, false)?;
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

        // alpha == 0 must not reference A or B: pass k = 0 to cuBLAS.
        let k_eff = if alpha == 0.0 { 0 } else { k };

        // Row-major C = A B via column-major identity C^T = B^T A^T: one sgemm
        // with operands swapped. A transpose_matrix view is an OP_T flag.
        // Leading dims satisfy ld ≥ max(1, rows) even for zero extents.
        let (trans_b, ldb) = match b_layout {
            MatrixLayout::RowMajor => (cublas_sys::cublasOperation_t::CUBLAS_OP_N, n.max(1)),
            MatrixLayout::Transposed => (cublas_sys::cublasOperation_t::CUBLAS_OP_T, k.max(1)),
        };
        let (trans_a, lda) = match a_layout {
            MatrixLayout::RowMajor => (cublas_sys::cublasOperation_t::CUBLAS_OP_N, k.max(1)),
            MatrixLayout::Transposed => (cublas_sys::cublasOperation_t::CUBLAS_OP_T, m.max(1)),
        };
        let ldc = n.max(1);

        // Dim/ld i32 checks run before the empty-output early return so a
        // zero-output gemm whose other extent exceeds i32::MAX is Error::Shape.
        let m_i = i32_dim(m)?;
        let n_i = i32_dim(n)?;
        let k_i = i32_dim(k_eff)?;
        let lda_i = i32_dim(lda)?;
        let ldb_i = i32_dim(ldb)?;
        let ldc_i = i32_dim(ldc)?;

        // Empty output: nothing to write and no cuBLAS launch.
        if m == 0 || n == 0 {
            return Ok(());
        }

        // Worker threads that did not run `new` must bind the context first.
        self.bind()?;

        // SAFETY: pointers are live device allocations for a/b/c; dims/ld match
        // the layouts above; Backend callers do not race writes on these ids.
        // cuBLAS with beta==0 does not read C (NaN-safe). With alpha==0, k_eff
        // is 0 so A and B are not referenced.
        let status = unsafe {
            (self.cublas.cublasSgemm_v2)(
                *self.blas.handle(),
                trans_b,
                trans_a,
                n_i,
                m_i,
                k_i,
                &alpha,
                b.storage.buf.ptr as *const f32,
                ldb_i,
                a.storage.buf.ptr as *const f32,
                lda_i,
                &beta,
                c.storage.buf.ptr as *mut f32,
                ldc_i,
            )
        };
        if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(Error::Schedule);
        }

        self.record("gemm-f32-cublas");
        Ok(())
    }

    fn gelu(&self, x: &Tensor<Self::Store>, y: &mut Tensor<Self::Store>) -> Result<(), Error> {
        let n = prepare_f32_row_major(&[x, y])?;
        if x.storage.storage_id() == y.storage.storage_id() {
            return Err(Error::Alias);
        }
        let mut n_i32 = i32::try_from(n).map_err(|_| Error::Shape)?;
        let (mut y_ptr, mut x_ptr) = (y.storage.buf.ptr, x.storage.buf.ptr);
        // gelu_kernel(float* y, const float* x, int n): writes `y`, reads `x`, `n` f32 each.
        // Backend callers do not race these storage_ids.
        let mut params: [*mut c_void; 3] = [
            (&mut y_ptr as *mut u64).cast(),
            (&mut x_ptr as *mut u64).cast(),
            (&mut n_i32 as *mut i32).cast(),
        ];
        self.launch(self.kernels.gelu, n, &mut params)?;
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
        let mut n_i32 = i32::try_from(n).map_err(|_| Error::Shape)?;
        let (mut dx_ptr, mut pre_ptr, mut dy_ptr) =
            (dx.storage.buf.ptr, pre.storage.buf.ptr, dy.storage.buf.ptr);
        // gelu_backward_kernel(float* dx, const float* pre, const float* dy, int n): writes
        // `dx`, reads `pre` and `dy`, `n` f32 each. Backend callers do not race these ids.
        let mut params: [*mut c_void; 4] = [
            (&mut dx_ptr as *mut u64).cast(),
            (&mut pre_ptr as *mut u64).cast(),
            (&mut dy_ptr as *mut u64).cast(),
            (&mut n_i32 as *mut i32).cast(),
        ];
        self.launch(self.kernels.gelu_backward, n, &mut params)?;
        self.record("gelu-backward-cuda");
        Ok(())
    }

    fn sync(&self) -> Result<(), Error> {
        self.stream.synchronize().map_err(|_| Error::Schedule)
    }

    fn download_into(&self, tensor: &Tensor<Self::Store>, out: &mut [u8]) -> Result<(), Error> {
        let len = Self::transfer_bytes(tensor, out.len())?;
        self.bind()?;
        let mut staging = self.staging.lock();
        if staging.ensure(false, len)? {
            self.bump(Counter::PinnedAllocs);
        }
        if len > 0 {
            // SAFETY: `down` holds at least `len` pinned bytes; the source is `len` bytes of a
            // live device buffer. The stream orders the copy after every earlier call.
            let status = unsafe {
                (self.kernels.driver.cuMemcpyDtoHAsync_v2)(
                    staging.down.ptr.cast(),
                    tensor.storage.buf.ptr,
                    len,
                    self.stream.cu_stream(),
                )
            };
            if status != driver_sys::CUresult::CUDA_SUCCESS {
                return Err(Error::Schedule);
            }
            self.bump(Counter::DeviceHostCopies);
        }
        self.sync_stream()?;
        if len > 0 {
            // SAFETY: the copy into `down` has completed; `out` is `len` bytes of caller memory.
            unsafe { std::ptr::copy_nonoverlapping(staging.down.ptr, out.as_mut_ptr(), len) };
        }
        drop(staging);
        self.bump_downloads();
        Ok(())
    }

    fn upload_into(&self, bytes: &[u8], tensor: &mut Tensor<Self::Store>) -> Result<(), Error> {
        let len = Self::transfer_bytes(tensor, bytes.len())?;
        self.bind()?;
        if len > 0 {
            let mut staging = self.staging.lock();
            if staging.ensure(true, len)? {
                self.bump(Counter::PinnedAllocs);
            }
            let driver = self.kernels.driver;
            // SAFETY: `up_done` fires once the previous upload has read `up`, so the host may
            // overwrite it; `up` holds at least `len` bytes. The device copy is stream-ordered
            // after every earlier call, and `up_done` is recorded after it.
            unsafe {
                if (driver.cuEventSynchronize)(staging.up_done)
                    != driver_sys::CUresult::CUDA_SUCCESS
                {
                    return Err(Error::Schedule);
                }
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), staging.up.ptr, len);
                let status = (driver.cuMemcpyHtoDAsync_v2)(
                    tensor.storage.buf.ptr,
                    staging.up.ptr.cast(),
                    len,
                    self.stream.cu_stream(),
                );
                if status != driver_sys::CUresult::CUDA_SUCCESS {
                    return Err(Error::Schedule);
                }
                if (driver.cuEventRecord)(staging.up_done, self.stream.cu_stream())
                    != driver_sys::CUresult::CUDA_SUCCESS
                {
                    return Err(Error::Schedule);
                }
            }
            self.bump(Counter::DeviceHostCopies);
        }
        self.bump_uploads();
        Ok(())
    }

    fn add_bias(
        &self,
        x: &Tensor<Self::Store>,
        b: &Tensor<Self::Store>,
        y: &mut Tensor<Self::Store>,
    ) -> Result<(), Error> {
        if x.shape.len() != 2
            || y.shape != x.shape
            || b.shape.len() != 1
            || b.shape[0] != x.shape[1]
        {
            return Err(Error::Shape);
        }
        all_f32(&[x, b, y])?;
        let b_id = b.storage.storage_id();
        if b_id == x.storage.storage_id() || b_id == y.storage.storage_id() {
            return Err(Error::Alias);
        }
        let n = packed(&[x, b, y])?;
        let mut cols = i32::try_from(x.shape[1]).map_err(|_| Error::Shape)?;
        if n > 0 {
            let mut n_i32 = n as i32;
            let (mut y_ptr, mut x_ptr, mut b_ptr) =
                (y.storage.buf.ptr, x.storage.buf.ptr, b.storage.buf.ptr);
            // add_bias_kernel(float* y, const float* x, const float* b, int n, int cols).
            let mut params: [*mut c_void; 5] = [
                (&mut y_ptr as *mut u64).cast(),
                (&mut x_ptr as *mut u64).cast(),
                (&mut b_ptr as *mut u64).cast(),
                (&mut n_i32 as *mut i32).cast(),
                (&mut cols as *mut i32).cast(),
            ];
            self.launch_with(self.kernels.add_bias, Self::blocks(n), &mut params)?;
        }
        self.record("add-bias-cuda");
        Ok(())
    }

    fn bias_grad(
        &self,
        dpre: &Tensor<Self::Store>,
        db: &mut Tensor<Self::Store>,
    ) -> Result<(), Error> {
        if dpre.shape.len() != 2 || db.shape.len() != 1 || db.shape[0] != dpre.shape[1] {
            return Err(Error::Shape);
        }
        all_f32(&[dpre, db])?;
        if dpre.storage.storage_id() == db.storage.storage_id() {
            return Err(Error::Alias);
        }
        packed(&[dpre, db])?;
        let mut rows = i32::try_from(dpre.shape[0]).map_err(|_| Error::Shape)?;
        let mut cols = i32::try_from(dpre.shape[1]).map_err(|_| Error::Shape)?;
        if rows > 0 && cols > 0 {
            let (mut db_ptr, mut d_ptr) = (db.storage.buf.ptr, dpre.storage.buf.ptr);
            // bias_grad_kernel(float* db, const float* dpre, int rows, int cols).
            let mut params: [*mut c_void; 4] = [
                (&mut db_ptr as *mut u64).cast(),
                (&mut d_ptr as *mut u64).cast(),
                (&mut rows as *mut i32).cast(),
                (&mut cols as *mut i32).cast(),
            ];
            self.launch_with(self.kernels.bias_grad, Self::blocks(cols as u32), &mut params)?;
        }
        self.record("bias-grad-cuda");
        Ok(())
    }

    fn mse_grad(
        &self,
        y: &Tensor<Self::Store>,
        target: &Tensor<Self::Store>,
        dy: &mut Tensor<Self::Store>,
        loss: &mut f32,
    ) -> Result<(), Error> {
        if target.shape != y.shape
            || dy.shape != y.shape
            || checked_nbytes(DType::F32, &y.shape)? == 0
        {
            return Err(Error::Shape);
        }
        all_f32(&[y, target, dy])?;
        if dy.storage.storage_id() == target.storage.storage_id() {
            return Err(Error::Alias);
        }
        let mut n = packed(&[y, target, dy])? as i32;
        let mut staging = self.staging.lock();
        self.bind()?;
        if staging.ensure(false, 4)? {
            self.bump(Counter::PinnedAllocs);
        }
        let (mut dy_ptr, mut loss_ptr, mut y_ptr, mut t_ptr) =
            (dy.storage.buf.ptr, staging.loss_ptr, y.storage.buf.ptr, target.storage.buf.ptr);
        // mse_grad_kernel(float* dy, float* loss, const float* y, const float* target, int n).
        let mut params: [*mut c_void; 5] = [
            (&mut dy_ptr as *mut u64).cast(),
            (&mut loss_ptr as *mut u64).cast(),
            (&mut y_ptr as *mut u64).cast(),
            (&mut t_ptr as *mut u64).cast(),
            (&mut n as *mut i32).cast(),
        ];
        let one_block = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (MSE_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        self.launch_with(self.kernels.mse_grad, one_block, &mut params)?;
        // SAFETY: `down` holds at least 4 pinned bytes; the loss is a live 4-byte device
        // buffer written by the kernel just queued.
        let status = unsafe {
            (self.kernels.driver.cuMemcpyDtoHAsync_v2)(
                staging.down.ptr.cast(),
                staging.loss_ptr,
                4,
                self.stream.cu_stream(),
            )
        };
        if status != driver_sys::CUresult::CUDA_SUCCESS {
            return Err(Error::Schedule);
        }
        self.sync_stream()?;
        // SAFETY: the copy has completed; `down` holds the loss's 4 bytes.
        *loss = unsafe { staging.down.ptr.cast::<f32>().read_unaligned() };
        drop(staging);
        self.record("mse-grad-cuda");
        Ok(())
    }

    fn sgd_step(
        &self,
        w: &mut Tensor<Self::Store>,
        dw: &mut Tensor<Self::Store>,
        lr: f32,
    ) -> Result<(), Error> {
        if dw.shape != w.shape {
            return Err(Error::Shape);
        }
        all_f32(&[w, dw])?;
        if w.storage.storage_id() == dw.storage.storage_id() {
            return Err(Error::Alias);
        }
        let n = packed(&[w, dw])?;
        if n > 0 {
            let (mut w_ptr, mut g_ptr, mut lr, mut n_i32) =
                (w.storage.buf.ptr, dw.storage.buf.ptr, lr, n as i32);
            // sgd_step_kernel(float* w, float* dw, float lr, int n).
            let mut params: [*mut c_void; 4] = [
                (&mut w_ptr as *mut u64).cast(),
                (&mut g_ptr as *mut u64).cast(),
                (&mut lr as *mut f32).cast(),
                (&mut n_i32 as *mut i32).cast(),
            ];
            self.launch_with(self.kernels.sgd_step, Self::blocks(n), &mut params)?;
        }
        self.record("sgd-step-cuda");
        Ok(())
    }

    fn convert(&self, x: &Tensor<Self::Store>, y: &mut Tensor<Self::Store>) -> Result<(), Error> {
        if y.shape != x.shape {
            return Err(Error::Shape);
        }
        let (function, launch) = match (x.dtype, y.dtype) {
            (DType::F32, DType::BF16) => (self.kernels.to_bf16, "convert-f32-bf16-cuda"),
            (DType::BF16, DType::F32) => (self.kernels.to_f32, "convert-bf16-f32-cuda"),
            _ => return Err(Error::Dtype),
        };
        if x.storage.storage_id() == y.storage.storage_id() {
            return Err(Error::Alias);
        }
        let n = packed(&[x, y])?;
        if n > 0 {
            let (mut y_ptr, mut x_ptr, mut n_i32) =
                (y.storage.buf.ptr, x.storage.buf.ptr, n as i32);
            // convert_*_kernel(out* y, const in* x, int n).
            let mut params: [*mut c_void; 3] = [
                (&mut y_ptr as *mut u64).cast(),
                (&mut x_ptr as *mut u64).cast(),
                (&mut n_i32 as *mut i32).cast(),
            ];
            self.launch_with(function, Self::blocks(n), &mut params)?;
        }
        self.record(launch);
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

fn i32_dim(value: usize) -> Result<i32, Error> {
    i32::try_from(value).map_err(|_| Error::Shape)
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

/// Row-major contiguous at offset 0, checked without building the stride vector.
fn packed_layout(t: &Tensor<CudaStore>) -> bool {
    if t.offset != 0 || t.strides.len() != t.shape.len() {
        return false;
    }
    let mut stride = 1usize;
    for (&extent, &have) in t.shape.iter().zip(&t.strides).rev() {
        if have != stride {
            return false;
        }
        stride = stride.wrapping_mul(extent);
    }
    true
}

fn all_f32(tensors: &[&Tensor<CudaStore>]) -> Result<(), Error> {
    if tensors.iter().any(|t| t.dtype != DType::F32) {
        return Err(Error::Dtype);
    }
    Ok(())
}

/// Row-major contiguous at offset 0 (else `UnsupportedLayout`), each buffer holds the tensor's
/// bytes (else `Shape`); returns the first tensor's element count, which must fit an `i32`
/// kernel argument (else `Shape`).
fn packed(tensors: &[&Tensor<CudaStore>]) -> Result<u32, Error> {
    if !tensors.iter().all(|t| packed_layout(t)) {
        return Err(Error::UnsupportedLayout);
    }
    for t in tensors {
        if checked_nbytes(t.dtype, &t.shape)? > t.storage.buf.nbytes {
            return Err(Error::Shape);
        }
    }
    let count = checked_nbytes(tensors[0].dtype, &tensors[0].shape)? / tensors[0].dtype.size_of();
    let count = i32::try_from(count).map_err(|_| Error::Shape)?;
    Ok(count as u32)
}

#[cfg(windows)]
mod windows_dll_search {
    use std::os::windows::ffi::OsStrExt;

    #[link(name = "kernel32")]
    extern "system" {
        fn SetDllDirectoryW(path: *const u16) -> i32;
    }

    /// Adds `%CUDA_PATH%\\bin` to the DLL search path, so cudarc's dynamic loading finds
    /// cuBLAS and NVRTC from the installed CUDA 12.x runtime even when `PATH` does not name it.
    /// `nvcuda.dll` comes with the driver in System32. Without `CUDA_PATH` this does nothing,
    /// and a library that still cannot be found makes `new` return `UnsupportedDevice`.
    pub(super) fn add_cuda_bin() {
        let Some(root) = std::env::var_os("CUDA_PATH") else {
            return;
        };
        let bin = std::path::Path::new(&root).join("bin");
        if !bin.is_dir() {
            return;
        }
        let wide: Vec<u16> = bin.as_os_str().encode_wide().chain(Some(0)).collect();
        // SAFETY: `wide` is a NUL-terminated UTF-16 path that outlives the call.
        unsafe {
            SetDllDirectoryW(wide.as_ptr());
        }
    }
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
