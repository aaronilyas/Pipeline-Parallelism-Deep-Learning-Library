//! macOS implementation: buffers are shared Metal buffers, f32 gemm delegates to the legacy
//! `linalg` contraction kernel, and GELU and the training ops run small kernels under
//! `shaders/`.
//!
//! # Ordering
//!
//! Every op commits one command buffer to the backend's one queue and returns without waiting.
//! The queue runs command buffers in commit order, and the shared buffers are hazard-tracked, so
//! launch order is execution order. The last committed command buffer is kept in
//! [`Shared`](crate::shared_state::Shared); `sync`, `download`, `download_into`, `upload_into`,
//! and `mse_grad` wait on it once.
//!
//! # Messages
//!
//! Every metal-rs method goes through objc 0.2.7's `sel!`, which loads a per-call-site
//! `static AtomicUsize` on each message. So buffer ids and lengths are read once at allocation,
//! selectors are registered once in [`MetalBackend::new`], and [`MetalBackend::dispatch`] sends
//! through `objc_msgSend` directly. The legacy gemm the spec requires is outside this rule. The
//! one lock on these paths is the swap of the last command buffer.

use std::ffi::c_void;

use linalg::abi::{alloc_shared, LinalgError, MetalContext, TensorView};
use linalg::ops::contraction::gemm_nowait as legacy_gemm_nowait;
use linalg_core::{row_major_strides, Backend, DType, DeviceKind, Error, Identified, Tensor};
use metal::foreign_types::{ForeignType, ForeignTypeRef};
use metal::objc::runtime::{Object, Sel};
use metal::{ComputePipelineState, MTLCommandBufferStatus, MTLSize};

use crate::shared_state::Shared;

const GELU_SOURCE: &str = include_str!("../shaders/gelu.metal");
const TRAIN_SOURCE: &str = include_str!("../shaders/train.metal");

/// Threads in `mse_grad_f32`'s one threadgroup; the shader's `MSE_THREADS` must match.
const MSE_THREADS: u64 = 1024;

#[link(name = "objc", kind = "dylib")]
extern "C" {
    /// The Objective-C message trampoline. Each use casts it to the message's exact C signature.
    fn objc_msgSend();
    fn objc_autoreleasePoolPush() -> *mut c_void;
    fn objc_autoreleasePoolPop(pool: *mut c_void);
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

/// One retained Objective-C object (here a command buffer), released on drop with the selector
/// registered in [`MetalBackend::new`].
pub(crate) struct Retained {
    ptr: Id,
    release: Sel,
}

// SAFETY: a command buffer may be waited on and released from any thread; `Retained` owns one
// reference and only messages it through `objc_msgSend`.
unsafe impl Send for Retained {}

impl Drop for Retained {
    fn drop(&mut self) {
        // SAFETY: `ptr` holds one retain that this drop balances.
        unsafe {
            let unit: unsafe extern "C" fn(Id, Sel) = msg_send_fn();
            unit(self.ptr, self.release);
        }
    }
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
    retain: Sel,
    release: Sel,
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
            retain: Sel::register("retain"),
            release: Sel::register("release"),
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
    waits: u64,
    host_copies: u64,
}

pub struct MetalBackend {
    ctx: MetalContext,
    gelu: Kernel,
    gelu_backward: Kernel,
    add_bias: Kernel,
    bias_grad: Kernel,
    mse_grad: Kernel,
    sgd_step: Kernel,
    to_bf16: Kernel,
    to_f32: Kernel,
    sel: Selectors,
    /// The last committed command buffer, and the loss buffer `mse_grad` writes.
    shared: Shared<Retained, MetalStore>,
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
        let train = ctx
            .device
            .new_library_with_source(TRAIN_SOURCE, &options)
            .map_err(|_| Error::UnsupportedDevice)?;
        let pipeline_in = |library: &metal::Library, name: &str| -> Result<Kernel, Error> {
            let function =
                library.get_function(name, None).map_err(|_| Error::UnsupportedDevice)?;
            let state = ctx
                .device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|_| Error::UnsupportedDevice)?;
            let width = state.max_total_threads_per_threadgroup().min(256);
            Ok(Kernel { state, width })
        };
        let gelu = pipeline_in(&library, "gelu_f32")?;
        let gelu_backward = pipeline_in(&library, "gelu_backward_f32")?;
        let mse_grad = pipeline_in(&train, "mse_grad_f32")?;
        if mse_grad.state.max_total_threads_per_threadgroup() < MSE_THREADS {
            return Err(Error::UnsupportedDevice);
        }
        let loss_buffer = alloc_shared::<f32>(&ctx, 1);
        let (contents, length) = (loss_buffer.contents() as u64, loss_buffer.length());
        Ok(MetalBackend {
            add_bias: pipeline_in(&train, "add_bias_f32")?,
            bias_grad: pipeline_in(&train, "bias_grad_f32")?,
            mse_grad,
            sgd_step: pipeline_in(&train, "sgd_step_f32")?,
            to_bf16: pipeline_in(&train, "convert_f32_bf16")?,
            to_f32: pipeline_in(&train, "convert_bf16_f32")?,
            ctx,
            gelu,
            gelu_backward,
            sel: Selectors::register(),
            shared: Shared::new(MetalStore { buffer: loss_buffer, contents, length }),
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

    fn bump_waits(&self) {
        #[cfg(any(test, feature = "debug-counters"))]
        self.with_stats(|s| s.waits += 1);
    }

    fn bump_host_copies(&self) {
        #[cfg(any(test, feature = "debug-counters"))]
        self.with_stats(|s| s.host_copies += 1);
    }

    /// `waitUntilCompleted` calls so far (test and `debug-counters` builds).
    #[cfg(any(test, feature = "debug-counters"))]
    pub fn debug_waits(&self) -> u64 {
        self.with_stats(|s| s.waits)
    }

    /// Host-side copies between device memory and caller memory so far (test and
    /// `debug-counters` builds).
    #[cfg(any(test, feature = "debug-counters"))]
    pub fn debug_host_copies(&self) -> u64 {
        self.with_stats(|s| s.host_copies)
    }

    /// Makes `cmd` the last committed command buffer; the one it replaces is released after the
    /// lock is gone.
    fn committed(&self, cmd: Retained) {
        drop(self.shared.set_last(cmd));
    }

    /// Waits once on the last committed command buffer. Returns at once when nothing has been
    /// committed.
    fn wait_last(&self) -> Result<(), Error> {
        let sel = &self.sel;
        // SAFETY: `retain` and `waitUntilCompleted` take no arguments; `status` returns
        // NSUInteger. The extra retain keeps the buffer alive after the lock is released.
        unsafe {
            let object: unsafe extern "C" fn(Id, Sel) -> Id = msg_send_fn();
            let unit: unsafe extern "C" fn(Id, Sel) = msg_send_fn();
            let status: unsafe extern "C" fn(Id, Sel) -> u64 = msg_send_fn();
            let held = self.shared.with_last(|last| {
                last.map(|cmd| Retained { ptr: object(cmd.ptr, sel.retain), release: sel.release })
            });
            let Some(cmd) = held else {
                return Ok(());
            };
            unit(cmd.ptr, sel.wait_until_completed);
            self.bump_waits();
            if status(cmd.ptr, sel.status) == MTLCommandBufferStatus::Error as u64 {
                return Err(Error::UnsupportedDevice);
            }
        }
        Ok(())
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
    /// `0..len` and `count` at index `len`. Commits without waiting.
    fn dispatch(
        &self,
        kernel: &Kernel,
        buffers: &[&metal::BufferRef],
        count: u32,
    ) -> Result<(), Error> {
        self.dispatch_with(kernel, buffers, bytes_of(&count), u64::from(count), kernel.width)
    }

    /// One compute dispatch of `kernel`: `buffers` at indices `0..len`, `params` copied to index
    /// `len`, `grid` threads in groups of `group`. Commits the command buffer, makes it the last
    /// committed one, and returns without waiting. The same messages metal-rs sends, through
    /// `objc_msgSend` with the selectors registered in `new`, inside an autorelease pool so the
    /// autoreleased encoder and command buffer do not outlive the call.
    fn dispatch_with(
        &self,
        kernel: &Kernel,
        buffers: &[&metal::BufferRef],
        params: &[u8],
        grid: u64,
        group: u64,
    ) -> Result<(), Error> {
        let sel = &self.sel;
        // SAFETY: each cast matches the method's Objective-C signature (NSUInteger is u64 and
        // MTLSize is #[repr(C)] on Apple Silicon). The queue, pipeline and buffers are live for
        // the call; `commandBuffer` and `computeCommandEncoder` return autoreleased objects,
        // which the pool pop releases after `retain` has taken a reference to the command
        // buffer. `params` outlives `setBytes:`, which copies it.
        let cmd = unsafe {
            let object: unsafe extern "C" fn(Id, Sel) -> Id = msg_send_fn();
            let unit: unsafe extern "C" fn(Id, Sel) = msg_send_fn();
            let set_state: unsafe extern "C" fn(Id, Sel, Id) = msg_send_fn();
            let set_buffer: unsafe extern "C" fn(Id, Sel, Id, u64, u64) = msg_send_fn();
            let set_bytes: unsafe extern "C" fn(Id, Sel, *const c_void, u64, u64) = msg_send_fn();
            let dispatch: unsafe extern "C" fn(Id, Sel, MTLSize, MTLSize) = msg_send_fn();

            let pool = objc_autoreleasePoolPush();
            let cmd = object(self.ctx.queue.as_ptr().cast(), sel.command_buffer);
            let enc = object(cmd, sel.compute_command_encoder);
            set_state(enc, sel.set_compute_pipeline_state, kernel.state.as_ptr().cast());
            for (index, buffer) in buffers.iter().enumerate() {
                set_buffer(enc, sel.set_buffer, buffer.as_ptr().cast(), 0, index as u64);
            }
            set_bytes(
                enc,
                sel.set_bytes,
                params.as_ptr().cast(),
                params.len() as u64,
                buffers.len() as u64,
            );
            dispatch(
                enc,
                sel.dispatch_threads,
                MTLSize { width: grid, height: 1, depth: 1 },
                MTLSize { width: group, height: 1, depth: 1 },
            );
            unit(enc, sel.end_encoding);
            unit(cmd, sel.commit);
            let cmd = Retained { ptr: object(cmd, sel.retain), release: sel.release };
            objc_autoreleasePoolPop(pool);
            cmd
        };
        self.committed(cmd);
        Ok(())
    }
}

/// The bytes of `value`, for `setBytes:`.
fn bytes_of<T: Copy>(value: &T) -> &[u8] {
    // SAFETY: `value` is a live `T` of `size_of::<T>()` bytes; the params passed here have no
    // padding.
    unsafe { std::slice::from_raw_parts((value as *const T).cast(), std::mem::size_of::<T>()) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SgdArgs {
    lr: f32,
    count: u32,
}

/// Row-major contiguous at offset 0, checked without building the stride vector.
fn packed_layout(t: &Tensor<MetalStore>) -> bool {
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

fn all_f32(tensors: &[&Tensor<MetalStore>]) -> Result<(), Error> {
    if tensors.iter().any(|t| t.dtype != DType::F32) {
        return Err(Error::Dtype);
    }
    Ok(())
}

/// Row-major contiguous at offset 0 (else `UnsupportedLayout`), each buffer holds the tensor's
/// bytes (else `Shape`); returns the first tensor's element count as `u32` (else `Shape`).
fn packed(tensors: &[&Tensor<MetalStore>]) -> Result<u32, Error> {
    if !tensors.iter().all(|t| packed_layout(t)) {
        return Err(Error::UnsupportedLayout);
    }
    for t in tensors {
        if nbytes(t.dtype, &t.shape)? > t.storage.length as usize {
            return Err(Error::Shape);
        }
    }
    u32::try_from(numel(&tensors[0].shape)?).map_err(|_| Error::Shape)
}

/// `x` is `[B, N]`, `b` is `[N]`, and `y` is `[B, N]`.
fn bias_shapes(x: &Tensor<MetalStore>, b: &Tensor<MetalStore>, y: &Tensor<MetalStore>) -> bool {
    x.shape.len() == 2 && y.shape == x.shape && b.shape.len() == 1 && b.shape[0] == x.shape[1]
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
        self.wait_last()?;
        // SAFETY: `start..end` lies inside the buffer, and `wait_last` returned after every
        // committed command buffer completed, so no GPU write is in flight.
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
        let result =
            legacy_gemm_nowait::<f32>(&self.ctx, alpha, &view(a), &view(b), beta, &view(c));
        let cmd = result.map_err(map_legacy)?;
        self.committed(Retained { ptr: cmd.into_ptr().cast(), release: self.sel.release });
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

    fn sync(&self) -> Result<(), Error> {
        self.wait_last()
    }

    fn download_into(&self, tensor: &Tensor<Self::Store>, out: &mut [u8]) -> Result<(), Error> {
        let len = nbytes(tensor.dtype, &tensor.shape)?;
        if out.len() != len {
            return Err(Error::Shape);
        }
        if !packed_layout(tensor) {
            return Err(Error::UnsupportedLayout);
        }
        if len > tensor.storage.length as usize {
            return Err(Error::Shape);
        }
        self.wait_last()?;
        // SAFETY: `0..len` lies inside the shared buffer, and every committed command buffer
        // has completed, so no GPU write is in flight. `out` is caller memory.
        unsafe {
            std::ptr::copy_nonoverlapping(
                tensor.storage.contents as usize as *const u8,
                out.as_mut_ptr(),
                len,
            );
        }
        self.bump_host_copies();
        self.bump_downloads();
        Ok(())
    }

    fn upload_into(&self, bytes: &[u8], tensor: &mut Tensor<Self::Store>) -> Result<(), Error> {
        let len = nbytes(tensor.dtype, &tensor.shape)?;
        if bytes.len() != len {
            return Err(Error::Shape);
        }
        if !packed_layout(tensor) {
            return Err(Error::UnsupportedLayout);
        }
        if len > tensor.storage.length as usize {
            return Err(Error::Shape);
        }
        // Earlier GPU work may still read or write this buffer; the host write follows it.
        self.wait_last()?;
        // SAFETY: `0..len` lies inside the shared buffer and no GPU work is in flight.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                tensor.storage.contents as usize as *mut u8,
                len,
            );
        }
        self.bump_host_copies();
        self.bump_uploads();
        Ok(())
    }

    fn add_bias(
        &self,
        x: &Tensor<Self::Store>,
        b: &Tensor<Self::Store>,
        y: &mut Tensor<Self::Store>,
    ) -> Result<(), Error> {
        if !bias_shapes(x, b, y) {
            return Err(Error::Shape);
        }
        all_f32(&[x, b, y])?;
        let b_id = b.storage.storage_id();
        if b_id == x.storage.storage_id() || b_id == y.storage.storage_id() {
            return Err(Error::Alias);
        }
        let count = packed(&[x, b, y])?;
        let cols = u32::try_from(x.shape[1]).map_err(|_| Error::Shape)?;
        if count > 0 {
            let buffers = [&*x.storage.buffer, &*b.storage.buffer, &*y.storage.buffer];
            let dims = [count, cols];
            self.dispatch_with(
                &self.add_bias,
                &buffers,
                bytes_of(&dims),
                u64::from(count),
                self.add_bias.width,
            )?;
        }
        self.record("add-bias-metal");
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
        let rows = u32::try_from(dpre.shape[0]).map_err(|_| Error::Shape)?;
        let cols = u32::try_from(dpre.shape[1]).map_err(|_| Error::Shape)?;
        if rows > 0 && cols > 0 {
            let dims = [rows, cols];
            self.dispatch_with(
                &self.bias_grad,
                &[&dpre.storage.buffer, &db.storage.buffer],
                bytes_of(&dims),
                u64::from(cols),
                self.bias_grad.width,
            )?;
        }
        self.record("bias-grad-metal");
        Ok(())
    }

    fn mse_grad(
        &self,
        y: &Tensor<Self::Store>,
        target: &Tensor<Self::Store>,
        dy: &mut Tensor<Self::Store>,
        loss: &mut f32,
    ) -> Result<(), Error> {
        if target.shape != y.shape || dy.shape != y.shape || numel(&y.shape)? == 0 {
            return Err(Error::Shape);
        }
        all_f32(&[y, target, dy])?;
        if dy.storage.storage_id() == target.storage.storage_id() {
            return Err(Error::Alias);
        }
        let count = packed(&[y, target, dy])?;
        let out = self.shared.loss();
        let buffers =
            [&*y.storage.buffer, &*target.storage.buffer, &*dy.storage.buffer, &*out.buffer];
        self.dispatch_with(&self.mse_grad, &buffers, bytes_of(&count), MSE_THREADS, MSE_THREADS)?;
        self.wait_last()?;
        // SAFETY: the loss buffer holds one f32, the kernel that wrote it has completed, and
        // the guard keeps every other `mse_grad` off it.
        *loss = unsafe { (out.contents as usize as *const f32).read() };
        self.record("mse-grad-metal");
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
        let count = packed(&[w, dw])?;
        if count > 0 {
            let args = SgdArgs { lr, count };
            self.dispatch_with(
                &self.sgd_step,
                &[&w.storage.buffer, &dw.storage.buffer],
                bytes_of(&args),
                u64::from(count),
                self.sgd_step.width,
            )?;
        }
        self.record("sgd-step-metal");
        Ok(())
    }

    fn convert(&self, x: &Tensor<Self::Store>, y: &mut Tensor<Self::Store>) -> Result<(), Error> {
        if y.shape != x.shape {
            return Err(Error::Shape);
        }
        let (kernel, launch) = match (x.dtype, y.dtype) {
            (DType::F32, DType::BF16) => (&self.to_bf16, "convert-f32-bf16-metal"),
            (DType::BF16, DType::F32) => (&self.to_f32, "convert-bf16-f32-metal"),
            _ => return Err(Error::Dtype),
        };
        if x.storage.storage_id() == y.storage.storage_id() {
            return Err(Error::Alias);
        }
        let count = packed(&[x, y])?;
        if count > 0 {
            self.dispatch(kernel, &[&x.storage.buffer, &y.storage.buffer], count)?;
        }
        self.record(launch);
        Ok(())
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
