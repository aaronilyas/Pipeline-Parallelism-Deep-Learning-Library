//! Frozen data model shared by every primitive.
//!
//! This file is owned by the scaffold and is byte-identical on every branch. Primitives
//! build on [`MetalContext`], [`TensorView`], [`alloc_shared`], [`compute_pipeline`], and
//! [`LinalgError`]; they never extend this file.
//!
//! Memory model: every tensor lives in a `StorageModeShared` buffer that the CPU and GPU
//! address directly (unified memory, zero copies). A [`TensorView`] describes a strided
//! window into such a buffer. Kernels bind the buffer at byte offset 0 and receive the
//! view's offset, shape, and strides (all in elements) through a `#[repr(C)]` params
//! struct. Host code reads or writes `host_ptr` only while no command buffer that touches
//! the buffer is in flight.

use std::collections::HashMap;
use std::fmt;
use std::mem::{align_of, size_of};
use std::ops::Range;
use std::sync::{Mutex, OnceLock};

/// Errors returned by every fallible operation in this crate.
///
/// Primitives return an error (and encode nothing) on invalid metadata; they never panic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LinalgError {
    /// No Metal device is available on this machine.
    NoDevice,
    /// A shape, stride, offset, extent, or parameter is invalid for the requested operation.
    InvalidArgument(String),
    /// A destination overlaps a source in a way the operation does not allow.
    Overlap(String),
    /// Runtime shader compilation or pipeline creation failed.
    ShaderCompile(String),
}

impl fmt::Display for LinalgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoDevice => write!(f, "no Metal device available"),
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            Self::Overlap(msg) => write!(f, "overlapping views: {msg}"),
            Self::ShaderCompile(msg) => write!(f, "shader compilation failed: {msg}"),
        }
    }
}

impl std::error::Error for LinalgError {}

/// Crate-wide result type.
pub type Result<T> = std::result::Result<T, LinalgError>;

/// A Metal device and the command queue all primitives encode onto.
#[rustfmt::skip]
pub struct MetalContext { pub device: metal::Device, pub queue: metal::CommandQueue }

impl MetalContext {
    /// Opens the system default Metal device and creates one command queue on it.
    pub fn new() -> Result<Self> {
        let device = metal::Device::system_default().ok_or(LinalgError::NoDevice)?;
        let queue = device.new_command_queue();
        Ok(Self { device, queue })
    }
}

/// A strided view of elements of type `T` inside a shared Metal buffer.
///
/// The element at multi-index `idx` lives at `host_ptr.add(offset + Σ idx[d] * strides[d])`
/// on the CPU and at the same element index on the GPU (buffer bound at byte offset 0).
/// `host_ptr` always equals the buffer base, never an offset-adjusted pointer.
#[rustfmt::skip]
pub struct TensorView<'a, T> {
    pub buffer: &'a metal::BufferRef,
    pub host_ptr: *mut T,     // == buffer.contents() as *mut T — buffer BASE, never offset-adjusted
    pub offset: usize,        // in elements
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,  // in elements; element at idx = host_ptr.add(offset + Σ idx[d]*strides[d])
}

impl<T> Clone for TensorView<'_, T> {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer,
            host_ptr: self.host_ptr,
            offset: self.offset,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }
}

impl<T> fmt::Debug for TensorView<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorView")
            .field("host_ptr", &self.host_ptr)
            .field("buffer_len", &self.buffer.length())
            .field("offset", &self.offset)
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .finish()
    }
}

impl<'a, T> TensorView<'a, T> {
    /// Builds a view with explicit metadata; `host_ptr` is taken from the buffer base.
    ///
    /// No validation happens here; primitives call [`TensorView::validate`].
    pub fn new(
        buffer: &'a metal::BufferRef,
        offset: usize,
        shape: Vec<usize>,
        strides: Vec<usize>,
    ) -> Self {
        Self { buffer, host_ptr: buffer.contents() as *mut T, offset, shape, strides }
    }

    /// Builds a contiguous row-major view of `shape` starting at element `offset`.
    pub fn row_major(buffer: &'a metal::BufferRef, offset: usize, shape: Vec<usize>) -> Self {
        let strides = row_major_strides(&shape);
        Self::new(buffer, offset, shape, strides)
    }

    /// Number of axes.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Number of elements (the product of the extents; 1 for rank 0).
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// True when some extent is zero.
    pub fn is_empty(&self) -> bool {
        self.shape.contains(&0)
    }

    /// Element index (relative to the buffer base) of multi-index `idx`.
    ///
    /// The caller guarantees `idx` has `rank()` entries within the extents.
    pub fn element_index(&self, idx: &[usize]) -> usize {
        self.offset + idx.iter().zip(&self.strides).map(|(i, s)| i * s).sum::<usize>()
    }

    /// Checks that the metadata is self-consistent and every reachable element lies inside
    /// the buffer, with no arithmetic overflow.
    pub fn validate(&self) -> Result<()> {
        let invalid = |msg: String| Err(LinalgError::InvalidArgument(msg));
        if size_of::<T>() == 0 {
            return invalid("zero-sized element type".into());
        }
        if self.shape.len() != self.strides.len() {
            return invalid(format!(
                "rank mismatch: shape {:?} vs strides {:?}",
                self.shape, self.strides
            ));
        }
        if self.host_ptr != self.buffer.contents() as *mut T {
            return invalid("host_ptr is not the buffer base".into());
        }
        if (self.host_ptr as usize) % align_of::<T>() != 0 {
            return invalid("buffer base is misaligned for the element type".into());
        }
        let capacity = self.buffer.length() as usize / size_of::<T>();
        if self.is_empty() {
            return if self.offset <= capacity {
                Ok(())
            } else {
                invalid(format!("offset {} beyond buffer capacity {capacity}", self.offset))
            };
        }
        let last = self.max_element_index().ok_or_else(|| {
            LinalgError::InvalidArgument(format!("index overflow in view {self:?}"))
        })?;
        if last >= capacity {
            return invalid(format!("view reaches element {last}, buffer holds {capacity}"));
        }
        Ok(())
    }

    /// Range of element indices (relative to the buffer base) the view can touch, or
    /// `None` for an empty view. Assumes the view passed [`TensorView::validate`].
    pub fn element_range(&self) -> Option<Range<usize>> {
        if self.is_empty() {
            return None;
        }
        self.max_element_index().map(|last| self.offset..last + 1)
    }

    /// Range of bytes (relative to the buffer base) the view can touch, or `None` for an
    /// empty view.
    pub fn byte_range(&self) -> Option<Range<usize>> {
        self.element_range().map(|r| r.start * size_of::<T>()..r.end * size_of::<T>())
    }

    /// Conservative overlap test: true when both views live in the same buffer and their
    /// byte ranges intersect. Empty views overlap nothing.
    pub fn may_overlap<U>(&self, other: &TensorView<'_, U>) -> bool {
        if self.buffer.contents() != other.buffer.contents() {
            return false;
        }
        match (self.byte_range(), other.byte_range()) {
            (Some(a), Some(b)) => a.start < b.end && b.start < a.end,
            _ => false,
        }
    }

    fn max_element_index(&self) -> Option<usize> {
        self.shape.iter().zip(&self.strides).try_fold(self.offset, |acc, (&n, &s)| {
            n.checked_sub(1)?.checked_mul(s).and_then(|d| acc.checked_add(d))
        })
    }
}

/// Row-major (C order) strides, in elements, for `shape`.
pub fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

/// Allocation granularity of [`alloc_shared`], in bytes.
pub const ALLOC_GRANULE: usize = 16 * 1024;

/// Allocates a zeroed `StorageModeShared` buffer holding at least `len` elements of `T`.
///
/// The byte length is rounded up to a multiple of [`ALLOC_GRANULE`] (at least one granule,
/// so `len == 0` is legal).
///
/// # Panics
/// Panics if the byte length overflows `usize`.
pub fn alloc_shared<T>(ctx: &MetalContext, len: usize) -> metal::Buffer {
    let bytes = len.checked_mul(size_of::<T>()).expect("alloc_shared: byte length overflows usize");
    let rounded = bytes
        .max(1)
        .checked_next_multiple_of(ALLOC_GRANULE)
        .expect("alloc_shared: byte length overflows usize");
    let buffer =
        ctx.device.new_buffer(rounded as u64, metal::MTLResourceOptions::StorageModeShared);
    // SAFETY: a fresh shared buffer of `rounded` bytes, not yet visible to the GPU.
    unsafe { std::ptr::write_bytes(buffer.contents() as *mut u8, 0, rounded) };
    buffer
}

type LibraryKey = (u64, &'static str);
type PipelineKey = (u64, &'static str, String);

fn libraries() -> &'static Mutex<HashMap<LibraryKey, metal::Library>> {
    static CACHE: OnceLock<Mutex<HashMap<LibraryKey, metal::Library>>> = OnceLock::new();
    CACHE.get_or_init(Default::default)
}

fn pipelines() -> &'static Mutex<HashMap<PipelineKey, metal::ComputePipelineState>> {
    static CACHE: OnceLock<Mutex<HashMap<PipelineKey, metal::ComputePipelineState>>> =
        OnceLock::new();
    CACHE.get_or_init(Default::default)
}

/// Returns the compute pipeline for kernel `function` of `module`, compiling and caching
/// on first use.
///
/// Libraries are compiled once per (device, module) from `source` with the options that
/// `options` builds (each module owns its `CompileOptions`, e.g. fast math off); pipelines
/// are cached per (device, module, function). Contexts on the same device share the cache.
/// Kernel variants are distinct `[[host_name]]` entry points, not function constants.
pub fn compute_pipeline(
    ctx: &MetalContext,
    module: &'static str,
    source: &str,
    options: impl FnOnce() -> metal::CompileOptions,
    function: &str,
) -> Result<metal::ComputePipelineState> {
    let device_id = ctx.device.registry_id();
    let key = (device_id, module, function.to_owned());
    if let Some(pipeline) = pipelines().lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return Ok(pipeline.clone());
    }
    let library = {
        let mut libs = libraries().lock().unwrap_or_else(|e| e.into_inner());
        match libs.get(&(device_id, module)) {
            Some(lib) => lib.clone(),
            None => {
                let lib = ctx
                    .device
                    .new_library_with_source(source, &options())
                    .map_err(|e| LinalgError::ShaderCompile(format!("{module}: {e}")))?;
                libs.insert((device_id, module), lib.clone());
                lib
            }
        }
    };
    let func = library
        .get_function(function, None)
        .map_err(|e| LinalgError::ShaderCompile(format!("{module}::{function}: {e}")))?;
    let pipeline = ctx
        .device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| LinalgError::ShaderCompile(format!("{module}::{function}: {e}")))?;
    pipelines().lock().unwrap_or_else(|e| e.into_inner()).insert(key, pipeline.clone());
    Ok(pipeline)
}
