//! Elementwise reciprocal over ℝ and ℂ for scalars, strided vectors, and batched diagonals.
//!
//! [`reciprocal`] maps every element `x` of a strided source view to `x⁻¹` in a destination
//! view of the same shape, either in place (source and destination are the same view) or out
//! of place (the views do not overlap). [`encode_reciprocal`] records the same work into a
//! caller-owned command buffer without committing it.
//!
//! # Real `f32`
//!
//! The result matches `(1.0 / x as f64) as f32` bit for bit, including signed zeros,
//! infinities, and subnormals. A NaN stays quiet (a signaling NaN is quieted) and keeps its
//! sign and payload. This GPU flushes subnormal operands and results, so the kernel
//! classifies the input from its bits, divides the normalized 24-bit significand with a
//! round-to-nearest-even 64-bit integer quotient, and packs a subnormal result the same way.
//! No Newton correction is used: the integer quotient is already correctly rounded.
//!
//! # Complex `[f32; 2]`
//!
//! Finite nonzero values use Smith's algorithm on a soft float (normal significand plus an
//! explicit power-of-two exponent), so a subnormal part is never flushed:
//!
//! - `|a| ≥ |b|`: `r = b/a`, `d = a + b·r`, `z⁻¹ = (1/d, −r/d)`
//! - otherwise: `r = a/b`, `d = a·r + b`, `z⁻¹ = (r/d, −1/d)`
//!
//! Zero, infinite, and NaN inputs follow C99 Annex G, as pinned by
//! `complex_specials_follow_documented_table`:
//!
//! - an infinite part, even beside a NaN, yields `(copysign(0, re), copysign(0, −im))`
//! - otherwise a NaN part yields a canonical quiet `(NaN, NaN)`
//! - `1/(±0 ± 0i) = (copysign(∞, re), copysign(0, −im))`, and a single zero part uses the
//!   same signs, so `1/(x + 0i)` matches the real reciprocal with imaginary part `−0`

use std::ffi::c_void;

use crate::abi::{compute_pipeline, LinalgError, MetalContext, Result, TensorView};

const SHADER: &str = include_str!("../../shaders/reciprocal.metal");
const MODULE: &str = "reciprocal";

/// Highest view rank the kernels address directly (one dispatch-grid axis per view axis).
pub const MAX_RANK: usize = 3;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for [f32; 2] {}
}

/// Element types the reciprocal kernels support: real `f32` and complex `[f32; 2]` = (re, im).
pub trait ReciprocalElement: sealed::Sealed + Copy + 'static {
    /// `[[host_name]]` of the kernel variant for this element type.
    const KERNEL: &'static str;
}

impl ReciprocalElement for f32 {
    const KERNEL: &'static str = "reciprocal_f32";
}

impl ReciprocalElement for [f32; 2] {
    const KERNEL: &'static str = "reciprocal_c32";
}

/// `MTLMathModeSafe`: no optimization may violate IEEE 754.
const MATH_MODE_SAFE: isize = 0;
/// `MTLMathFloatingPointFunctionsPrecise`: single-precision functions default to `precise::`.
const MATH_FUNCTIONS_PRECISE: isize = 1;

/// Compile options of the reciprocal shader: MSL 3.1, fast math off, and (where the OS
/// offers the selectors, macOS 15+) safe math mode with precise floating-point functions,
/// so that `/` is an IEEE division.
pub fn compile_options() -> metal::CompileOptions {
    use metal::foreign_types::ForeignType;
    use metal::objc::runtime::{Object, Sel};
    use metal::objc::Message;

    let options = metal::CompileOptions::new();
    options.set_language_version(metal::MTLLanguageVersion::V3_1);
    options.set_fast_math_enabled(false);
    let setters = [
        ("setMathMode:", MATH_MODE_SAFE),
        ("setMathFloatingPointFunctions:", MATH_FUNCTIONS_PRECISE),
    ];
    for (name, value) in setters {
        let selector = Sel::register(name);
        // SAFETY: `options` is a live `MTLCompileOptions`; each selector is sent only when
        // the object responds to it and takes one `NSInteger`-backed enum, returning void.
        unsafe {
            let object = &*(options.as_ptr() as *const Object);
            // `instance_method` is how this objc crate reports that the selector exists.
            // macOS 15+ has both setters; earlier systems keep fast-math-off alone.
            if object.class().instance_method(selector).is_some() {
                let _ = options.send_message::<(isize,), ()>(selector, (value,));
            }
        }
    }
    options
}

/// `#[repr(C)]` mirror of `RecipParams` in `shaders/reciprocal.metal`.
#[repr(C)]
#[derive(Clone, Copy)]
struct RecipParams {
    src_offset: u32,
    dst_offset: u32,
    rank: u32,
    count: u32,
    shape: [u32; MAX_RANK],
    src_strides: [u32; MAX_RANK],
    dst_strides: [u32; MAX_RANK],
}

const _: () = assert!(std::mem::size_of::<RecipParams>() == 52);
const _: () = assert!(std::mem::align_of::<RecipParams>() == 4);

/// Computes `dst[i] = 1 / src[i]` for every multi-index `i` and waits for completion.
///
/// Returns an error, and launches nothing, when the views are invalid, differ in shape,
/// overlap without being the same view, or the destination is not injective. See the
/// [module docs](self) for rounding and the complex specials.
pub fn reciprocal<T: ReciprocalElement>(
    ctx: &MetalContext,
    src: &TensorView<'_, T>,
    dst: &TensorView<'_, T>,
) -> Result<()> {
    let cmd = ctx.queue.new_command_buffer();
    encode_reciprocal(ctx, cmd, src, dst)?;
    cmd.commit();
    cmd.wait_until_completed();
    if cmd.status() != metal::MTLCommandBufferStatus::Completed {
        return Err(LinalgError::InvalidArgument(format!(
            "reciprocal command buffer finished with status {:?}",
            cmd.status()
        )));
    }
    Ok(())
}

/// Encodes `dst[i] = 1 / src[i]` for every multi-index `i` into `cmd`.
///
/// Creates and ends its own compute encoder. Does not commit, wait, or read or write host
/// memory, so several primitives may encode into one command buffer. An empty view encodes
/// nothing. Invalid metadata and partial overlap return an error and encode nothing.
pub fn encode_reciprocal<T: ReciprocalElement>(
    ctx: &MetalContext,
    cmd: &metal::CommandBufferRef,
    src: &TensorView<'_, T>,
    dst: &TensorView<'_, T>,
) -> Result<()> {
    let Some(params) = prepare(src, dst)? else {
        return Ok(());
    };
    let pipeline = compute_pipeline(ctx, MODULE, SHADER, compile_options, T::KERNEL)?;
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(src.buffer), 0);
    encoder.set_buffer(1, Some(dst.buffer), 0);
    encoder.set_bytes(
        2,
        std::mem::size_of::<RecipParams>() as u64,
        &params as *const RecipParams as *const c_void,
    );
    let groups = threadgroup_width(&pipeline, params.count);
    encoder.dispatch_threads(
        metal::MTLSize::new(u64::from(params.count), 1, 1),
        metal::MTLSize::new(groups, 1, 1),
    );
    encoder.end_encoding();
    Ok(())
}

fn prepare<T>(src: &TensorView<'_, T>, dst: &TensorView<'_, T>) -> Result<Option<RecipParams>> {
    src.validate()?;
    dst.validate()?;
    if src.shape != dst.shape {
        return Err(LinalgError::InvalidArgument(format!(
            "reciprocal shape mismatch: src {:?} vs dst {:?}",
            src.shape, dst.shape
        )));
    }
    if src.rank() > MAX_RANK {
        return Err(LinalgError::InvalidArgument(format!(
            "reciprocal rank {} exceeds {MAX_RANK}",
            src.rank()
        )));
    }
    if !same_view(src, dst) && src.may_overlap(dst) {
        return Err(LinalgError::Overlap(
            "reciprocal views overlap; in-place requires the same buffer, offset, shape, and strides"
                .into(),
        ));
    }
    if src.is_empty() {
        return Ok(None);
    }
    if !is_injective(&dst.shape, &dst.strides) {
        return Err(LinalgError::InvalidArgument(
            "reciprocal destination is not injective; parallel stores would alias".into(),
        ));
    }
    let count = count_u32(&src.shape).ok_or_else(|| {
        LinalgError::InvalidArgument("reciprocal element count does not fit in u32".into())
    })?;
    if !indexes_fit_u32(src) || !indexes_fit_u32(dst) {
        return Err(LinalgError::InvalidArgument(
            "reciprocal index does not fit in u32; refusing to launch a 32-bit kernel".into(),
        ));
    }
    Ok(Some(RecipParams {
        src_offset: src.offset as u32,
        dst_offset: dst.offset as u32,
        rank: src.rank() as u32,
        count,
        shape: pack(&src.shape),
        src_strides: pack(&src.strides),
        dst_strides: pack(&dst.strides),
    }))
}

fn same_view<T>(a: &TensorView<'_, T>, b: &TensorView<'_, T>) -> bool {
    std::ptr::eq(a.buffer.contents(), b.buffer.contents())
        && a.offset == b.offset
        && a.shape == b.shape
        && a.strides == b.strides
}

fn count_u32(shape: &[usize]) -> Option<u32> {
    let mut count = 1u32;
    for &extent in shape {
        count = count.checked_mul(u32::try_from(extent).ok()?)?;
    }
    Some(count)
}

fn indexes_fit_u32<T>(view: &TensorView<'_, T>) -> bool {
    if view.offset > u32::MAX as usize {
        return false;
    }
    if view.shape.iter().any(|&extent| extent > u32::MAX as usize)
        || view.strides.iter().any(|&stride| stride > u32::MAX as usize)
    {
        return false;
    }
    match view.element_range() {
        Some(range) => range.end - 1 <= u32::MAX as usize,
        None => true,
    }
}

/// True when distinct multi-indices land on distinct elements. Size-1 axes are ignored.
/// Strides are non-negative. Axes are sorted by stride; each stride must clear the highest
/// offset the tighter axes can form.
fn is_injective(shape: &[usize], strides: &[usize]) -> bool {
    let mut axes: Vec<(usize, usize)> = shape
        .iter()
        .copied()
        .zip(strides.iter().copied())
        .filter(|&(extent, _)| extent > 1)
        .map(|(extent, stride)| (stride, extent))
        .collect();
    axes.sort_by_key(|&(stride, _)| stride);
    let mut max_offset = 0usize;
    for (stride, extent) in axes {
        if stride <= max_offset {
            return false;
        }
        let step = match stride.checked_mul(extent - 1) {
            Some(step) => step,
            None => return false,
        };
        max_offset = match max_offset.checked_add(step) {
            Some(next) => next,
            None => return false,
        };
    }
    true
}

fn pack(values: &[usize]) -> [u32; MAX_RANK] {
    let mut packed = [0u32; MAX_RANK];
    for (slot, &value) in packed.iter_mut().zip(values) {
        *slot = value as u32;
    }
    packed
}

fn threadgroup_width(pipeline: &metal::ComputePipelineStateRef, count: u32) -> u64 {
    let simd = pipeline.thread_execution_width().max(1);
    let limit = pipeline.max_total_threads_per_threadgroup().max(1);
    let mut width = simd.saturating_mul((256 / simd).max(1));
    if width > limit {
        width = (limit / simd) * simd;
    }
    if width == 0 {
        width = simd.min(limit);
    }
    width.min(u64::from(count)).max(1)
}
