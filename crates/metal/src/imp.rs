//! macOS implementation: buffers are shared Metal buffers, f32 gemm delegates to the legacy
//! `linalg` contraction kernel, and GELU runs two small kernels under `shaders/`.
//!
//! The backend's own code on the gemm, gelu and gelu_backward paths performs no atomic. Every
//! metal-rs method goes through objc 0.2.7's `sel!`, which loads a per-call-site
//! `static AtomicUsize` on each message. So buffer ids and lengths are read once at allocation,
//! selectors are registered once in [`MetalBackend::new`], and [`MetalBackend::dispatch`] sends
//! through `objc_msgSend` directly. The legacy gemm the spec requires is outside this rule.

use std::ffi::c_void;

use linalg::abi::{alloc_shared, LinalgError, MetalContext, TensorView};
use linalg::ops::contraction::gemm as legacy_gemm;
use linalg_core::{row_major_strides, Backend, DType, DeviceKind, Error, Identified, Tensor};
use metal::foreign_types::{ForeignType, ForeignTypeRef};
use metal::objc::runtime::{Object, Sel};
use metal::{ComputePipelineState, MTLCommandBufferStatus, MTLSize};

const GELU_SOURCE: &str = include_str!("../shaders/gelu.metal");

#[link(name = "objc", kind = "dylib")]
extern "C" {
    /// The Objective-C message trampoline. Each use casts it to the message's exact C signature.
    fn objc_msgSend();
}

type Id = *mut Object;

/// `objc_msgSend` as the function-pointer type `F`, which must be the message's C signature
/// (receiver, selector, then the arguments).
///
/// # Safety
/// `F` must be an `unsafe extern "C" fn` type matching the method that will be sent.
unsafe fn msg_send_fn<F: Copy>() -> F {
    let trampoline: unsafe extern "C" fn() = objc_msgSend;
    std::mem::transmute_copy(&trampoline)
}

/// One shared Metal buffer. `Clone` retains the buffer and copies nothing.
///
/// `metal::Buffer` is `Send + Sync`, so no unsafe impl is needed here.
#[derive(Clone)]
pub struct MetalStore {
    pub(crate) buffer: metal::Buffer,
    /// `buffer.contents()`, read once at allocation. A shared buffer's contents never move.
    contents: u64,
    /// `buffer.length()` in bytes, read once at allocation.
    length: u64,
}

impl Identified for MetalStore {
    fn storage_id(&self) -> u64 {
        self.contents
    }
}

/// Every selector `dispatch` sends, registered once.
struct Selectors {
    command_buffer: Sel,
    compute_command_encoder: Sel,
    set_compute_pipeline_state: Sel,
    set_buffer: Sel,
    set_bytes: Sel,
    dispatch_threads: Sel,
    end_encoding: Sel,
    commit: Sel,
    wait_until_completed: Sel,
    status: Sel,
}

impl Selectors {
    fn register() -> Self {
        Self {
            command_buffer: Sel::register("commandBuffer"),
            compute_command_encoder: Sel::register("computeCommandEncoder"),
            set_compute_pipeline_state: Sel::register("setComputePipelineState:"),
            set_buffer: Sel::register("setBuffer:offset:atIndex:"),
            set_bytes: Sel::register("setBytes:length:atIndex:"),
            dispatch_threads: Sel::register("dispatchThreads:threadsPerThreadgroup:"),
            end_encoding: Sel::register("endEncoding"),
            commit: Sel::register("commit"),
            wait_until_completed: Sel::register("waitUntilCompleted"),
            status: Sel::register("status"),
        }
    }
}

/// A GELU pipeline and its threadgroup width, read once.
struct Kernel {
    state: ComputePipelineState,
    width: u64,
}

#[cfg(any(test, feature = "debug-counters"))]
#[derive(Default)]
struct Stats {
    allocs: u64,
    downloads: u64,
    uploads: u64,
    launches: Vec<&'static str>,
}

pub struct MetalBackend {
    ctx: MetalContext,
    gelu: Kernel,
    gelu_backward: Kernel,
    sel: Selectors,
    #[cfg(any(test, feature = "debug-counters"))]
    stats: std::sync::Mutex<Stats>,
}

impl MetalBackend {
    pub fn new() -> Result<MetalBackend, Error> {
        let ctx = MetalContext::new().map_err(|_| Error::UnsupportedDevice)?;
        let options = metal::CompileOptions::new();
        options.set_fast_math_enabled(false);
        let library = ctx
            .device
            .new_library_with_source(GELU_SOURCE, &options)
            .map_err(|_| Error::UnsupportedDevice)?;
        let pipeline = |name: &str| -> Result<Kernel, Error> {
            let function =
                library.get_function(name, None).map_err(|_| Error::UnsupportedDevice)?;
            let state = ctx
                .device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|_| Error::UnsupportedDevice)?;
            let width = state.max_total_threads_per_threadgroup().min(256);
            Ok(Kernel { state, width })
        };
        let gelu = pipeline("gelu_f32")?;
        let gelu_backward = pipeline("gelu_backward_f32")?;
        Ok(MetalBackend {
            ctx,
            gelu,
            gelu_backward,
            sel: Selectors::register(),
            #[cfg(any(test, feature = "debug-counters"))]
            stats: std::sync::Mutex::new(Stats::default()),
        })
    }

    #[cfg(any(test, feature = "debug-counters"))]
    fn with_stats<R>(&self, f: impl FnOnce(&mut Stats) -> R) -> R {
        f(&mut self.stats.lock().expect("metal stats"))
    }

    fn bump_allocs(&self) {
        #[cfg(any(test, feature = "debug-counters"))]
        self.with_stats(|s| s.allocs += 1);
    }

    fn bump_uploads(&self) {
        #[cfg(any(test, feature = "debug-counters"))]
        self.with_stats(|s| s.uploads += 1);
    }

    fn bump_downloads(&self) {
        #[cfg(any(test, feature = "debug-counters"))]
        self.with_stats(|s| s.downloads += 1);
    }

    fn record(&self, name: &'static str) {
        #[cfg(any(test, feature = "debug-counters"))]
        self.with_stats(|s| s.launches.push(name));
        #[cfg(not(any(test, feature = "debug-counters")))]
        let _ = name;
    }

    /// Creates a zeroed row-major tensor. Zero-element tensors still own a distinct buffer.
    fn new_tensor(&self, dtype: DType, shape: &[usize]) -> Result<Tensor<MetalStore>, Error> {
        let bytes = nbytes(dtype, shape)?;
        let buffer = alloc_shared::<u8>(&self.ctx, bytes);
        let (contents, length) = (buffer.contents() as u64, buffer.length());
        Ok(Tensor {
            storage: MetalStore { buffer, contents, length },
            dtype,
            shape: shape.to_vec(),
            strides: row_major_strides(shape),
            offset: 0,
        })
    }

    /// Validates same-shape contiguous f32 tensors and returns the element count.
    fn elementwise_count(tensors: &[&Tensor<MetalStore>]) -> Result<u32, Error> {
        if tensors.iter().any(|t| t.dtype != DType::F32) {
            return Err(Error::Dtype);
        }
        let shape = &tensors[0].shape;
        if tensors.iter().any(|t| t.shape != *shape) {
            return Err(Error::Shape);
        }
        let packed = row_major_strides(shape);
        if tensors.iter().any(|t| t.offset != 0 || t.strides != packed) {
            return Err(Error::UnsupportedLayout);
        }
        let count = numel(shape)?;
        let bytes = count.checked_mul(4).ok_or(Error::Shape)?;
        if tensors.iter().any(|t| (t.storage.length as usize) < bytes) {
            return Err(Error::Shape);
        }
        u32::try_from(count).map_err(|_| Error::Shape)
    }

    /// One compute dispatch of `kernel` over `count` threads with `buffers` at indices
    /// `0..len` and `count` at index `len`; waits for completion. The same messages metal-rs
    /// sends, through `objc_msgSend` with the selectors registered in `new`.
    fn dispatch(
        &self,
        kernel: &Kernel,
        buffers: &[&metal::BufferRef],
        count: u32,
    ) -> Result<(), Error> {
        let sel = &self.sel;
        // SAFETY: each cast matches the method's Objective-C signature (NSUInteger is u64 and
        // MTLSize is #[repr(C)] on Apple Silicon). The queue, pipeline and buffers are live for
        // the call; `commandBuffer` and `computeCommandEncoder` return autoreleased objects, as
        // through metal-rs. `count` outlives `setBytes:`, which copies it.
        unsafe {
            let object: unsafe extern "C" fn(Id, Sel) -> Id = msg_send_fn();
            let unit: unsafe extern "C" fn(Id, Sel) = msg_send_fn();
            let set_state: unsafe extern "C" fn(Id, Sel, Id) = msg_send_fn();
            let set_buffer: unsafe extern "C" fn(Id, Sel, Id, u64, u64) = msg_send_fn();
            let set_bytes: unsafe extern "C" fn(Id, Sel, *const c_void, u64, u64) = msg_send_fn();
            let dispatch: unsafe extern "C" fn(Id, Sel, MTLSize, MTLSize) = msg_send_fn();
            let status: unsafe extern "C" fn(Id, Sel) -> u64 = msg_send_fn();

            let cmd = object(self.ctx.queue.as_ptr().cast(), sel.command_buffer);
            let enc = object(cmd, sel.compute_command_encoder);
            set_state(enc, sel.set_compute_pipeline_state, kernel.state.as_ptr().cast());
            for (index, buffer) in buffers.iter().enumerate() {
                set_buffer(enc, sel.set_buffer, buffer.as_ptr().cast(), 0, index as u64);
            }
            set_bytes(
                enc,
                sel.set_bytes,
                (&count as *const u32).cast(),
                std::mem::size_of::<u32>() as u64,
                buffers.len() as u64,
            );
            dispatch(
                enc,
                sel.dispatch_threads,
                MTLSize { width: u64::from(count), height: 1, depth: 1 },
                MTLSize { width: kernel.width, height: 1, depth: 1 },
            );
            unit(enc, sel.end_encoding);
            unit(cmd, sel.commit);
            unit(cmd, sel.wait_until_completed);
            if status(cmd, sel.status) == MTLCommandBufferStatus::Error as u64 {
                return Err(Error::UnsupportedDevice);
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Layout {
    RowMajor,
    Transposed,
}

fn layout_of(t: &Tensor<MetalStore>) -> Option<Layout> {
    if t.offset != 0 || t.shape.len() != 2 || t.strides.len() != 2 {
        return None;
    }
    if t.strides == row_major_strides(&t.shape) {
        return Some(Layout::RowMajor);
    }
    // `transpose_matrix` of row-major `[cols, rows]` has strides `[1, rows]`.
    if t.strides[0] == 1 && t.strides[1] == t.shape[0] {
        return Some(Layout::Transposed);
    }
    None
}

fn map_legacy(err: LinalgError) -> Error {
    match err {
        LinalgError::InvalidArgument(_) => Error::Shape,
        LinalgError::Overlap(_) => Error::Alias,
        LinalgError::NoDevice | LinalgError::ShaderCompile(_) => Error::UnsupportedDevice,
    }
}

/// The legacy view of `t`, built from the cached contents address (`TensorView::new` would
/// message the buffer for it).
fn view<'a>(t: &'a Tensor<MetalStore>) -> TensorView<'a, f32> {
    TensorView {
        buffer: &t.storage.buffer,
        host_ptr: t.storage.contents as usize as *mut f32,
        offset: 0,
        shape: t.shape.clone(),
        strides: t.strides.clone(),
    }
}

fn numel(shape: &[usize]) -> Result<usize, Error> {
    shape.iter().try_fold(1usize, |acc, &e| acc.checked_mul(e)).ok_or(Error::Shape)
}

fn nbytes(dtype: DType, shape: &[usize]) -> Result<usize, Error> {
    numel(shape)?.checked_mul(dtype.size_of()).ok_or(Error::Shape)
}

impl Backend for MetalBackend {
    type Store = MetalStore;

    fn kind(&self) -> DeviceKind {
        DeviceKind::Metal
    }

    fn alloc(&self, dtype: DType, shape: &[usize]) -> Result<Tensor<MetalStore>, Error> {
        let tensor = self.new_tensor(dtype, shape)?;
        self.bump_allocs();
        Ok(tensor)
    }

    fn upload(
        &self,
        dtype: DType,
        shape: &[usize],
        row_major: &[u8],
    ) -> Result<Tensor<MetalStore>, Error> {
        if row_major.len() != nbytes(dtype, shape)? {
            return Err(Error::Dtype);
        }
        let tensor = self.new_tensor(dtype, shape)?;
        // SAFETY: the fresh buffer holds at least `row_major.len()` bytes and no GPU work uses it.
        unsafe {
            std::ptr::copy_nonoverlapping(
                row_major.as_ptr(),
                tensor.storage.buffer.contents().cast::<u8>(),
                row_major.len(),
            );
        }
        self.bump_uploads();
        Ok(tensor)
    }

    fn download(&self, tensor: &Tensor<MetalStore>) -> Result<Vec<u8>, Error> {
        if tensor.strides != row_major_strides(&tensor.shape) {
            return Err(Error::UnsupportedLayout);
        }
        let size = tensor.dtype.size_of();
        let start = tensor.offset.checked_mul(size).ok_or(Error::Shape)?;
        let len = nbytes(tensor.dtype, &tensor.shape)?;
        let end = start.checked_add(len).ok_or(Error::Shape)?;
        if end > tensor.storage.buffer.length() as usize {
            return Err(Error::Shape);
        }
        // SAFETY: `start..end` lies inside the buffer, and every launch here waits for
        // completion before returning, so no GPU write is in flight.
        let out = unsafe {
            std::slice::from_raw_parts(
                tensor.storage.buffer.contents().cast::<u8>().add(start),
                len,
            )
            .to_vec()
        };
        self.bump_downloads();
        Ok(out)
    }

    fn gemm(
        &self,
        a: &Tensor<MetalStore>,
        b: &Tensor<MetalStore>,
        c: &mut Tensor<MetalStore>,
        alpha: f32,
        beta: f32,
    ) -> Result<(), Error> {
        for t in [a, b, &*c] {
            if t.shape.len() != 2 || t.strides.len() != 2 {
                return Err(Error::Shape);
            }
        }
        if a.shape[1] != b.shape[0] || c.shape[0] != a.shape[0] || c.shape[1] != b.shape[1] {
            return Err(Error::Shape);
        }
        if a.dtype != b.dtype || a.dtype != c.dtype {
            return Err(Error::Dtype);
        }
        if a.dtype != DType::F32 {
            return Err(Error::Dtype);
        }
        let (ids_a, ids_b, ids_c) =
            (a.storage.storage_id(), b.storage.storage_id(), c.storage.storage_id());
        if ids_a == ids_b || ids_a == ids_c || ids_b == ids_c {
            return Err(Error::Alias);
        }
        let layouts = (layout_of(a), layout_of(b), layout_of(c));
        if !matches!(
            layouts,
            (Some(Layout::RowMajor), Some(Layout::RowMajor), Some(Layout::RowMajor))
                | (Some(Layout::Transposed), Some(Layout::RowMajor), Some(Layout::RowMajor))
                | (Some(Layout::RowMajor), Some(Layout::Transposed), Some(Layout::RowMajor))
        ) {
            return Err(Error::UnsupportedLayout);
        }
        if c.shape[0] == 0 || c.shape[1] == 0 {
            return Ok(());
        }
        let result = legacy_gemm::<f32>(&self.ctx, alpha, &view(a), &view(b), beta, &view(c));
        result.map_err(map_legacy)?;
        self.record("gemm-f32-metal");
        Ok(())
    }

    fn gelu(&self, x: &Tensor<MetalStore>, y: &mut Tensor<MetalStore>) -> Result<(), Error> {
        let count = Self::elementwise_count(&[x, y])?;
        if count == 0 {
            return Ok(());
        }
        self.dispatch(&self.gelu, &[&x.storage.buffer, &y.storage.buffer], count)?;
        self.record("gelu-metal");
        Ok(())
    }

    fn gelu_backward(
        &self,
        pre: &Tensor<MetalStore>,
        dy: &Tensor<MetalStore>,
        dx: &mut Tensor<MetalStore>,
    ) -> Result<(), Error> {
        let count = Self::elementwise_count(&[pre, dy, dx])?;
        if count == 0 {
            return Ok(());
        }
        self.dispatch(
            &self.gelu_backward,
            &[&pre.storage.buffer, &dy.storage.buffer, &dx.storage.buffer],
            count,
        )?;
        self.record("gelu-backward-metal");
        Ok(())
    }

    // Not yet implemented by this backend; see the core contract.
    fn download_into(&self, _tensor: &Tensor<Self::Store>, _out: &mut [u8]) -> Result<(), Error> {
        Err(Error::UnsupportedDevice)
    }

    fn upload_into(&self, _bytes: &[u8], _tensor: &mut Tensor<Self::Store>) -> Result<(), Error> {
        Err(Error::UnsupportedDevice)
    }

    fn add_bias(
        &self,
        _x: &Tensor<Self::Store>,
        _b: &Tensor<Self::Store>,
        _y: &mut Tensor<Self::Store>,
    ) -> Result<(), Error> {
        Err(Error::UnsupportedDevice)
    }

    fn bias_grad(
        &self,
        _dpre: &Tensor<Self::Store>,
        _db: &mut Tensor<Self::Store>,
    ) -> Result<(), Error> {
        Err(Error::UnsupportedDevice)
    }

    fn mse_grad(
        &self,
        _y: &Tensor<Self::Store>,
        _target: &Tensor<Self::Store>,
        _dy: &mut Tensor<Self::Store>,
        _loss: &mut f32,
    ) -> Result<(), Error> {
        Err(Error::UnsupportedDevice)
    }

    fn sgd_step(
        &self,
        _w: &mut Tensor<Self::Store>,
        _dw: &mut Tensor<Self::Store>,
        _lr: f32,
    ) -> Result<(), Error> {
        Err(Error::UnsupportedDevice)
    }

    fn convert(&self, _x: &Tensor<Self::Store>, _y: &mut Tensor<Self::Store>) -> Result<(), Error> {
        Err(Error::UnsupportedDevice)
    }

    fn debug_allocs(&self) -> u64 {
        #[cfg(any(test, feature = "debug-counters"))]
        {
            self.with_stats(|s| s.allocs)
        }
        #[cfg(not(any(test, feature = "debug-counters")))]
        {
            0
        }
    }

    fn debug_launches(&self) -> Vec<&'static str> {
        #[cfg(any(test, feature = "debug-counters"))]
        {
            self.with_stats(|s| s.launches.clone())
        }
        #[cfg(not(any(test, feature = "debug-counters")))]
        {
            Vec::new()
        }
    }

    fn debug_downloads(&self) -> u64 {
        #[cfg(any(test, feature = "debug-counters"))]
        {
            self.with_stats(|s| s.downloads)
        }
        #[cfg(not(any(test, feature = "debug-counters")))]
        {
            0
        }
    }

    fn debug_uploads(&self) -> u64 {
        #[cfg(any(test, feature = "debug-counters"))]
        {
            self.with_stats(|s| s.uploads)
        }
        #[cfg(not(any(test, feature = "debug-counters")))]
        {
            0
        }
    }
}
