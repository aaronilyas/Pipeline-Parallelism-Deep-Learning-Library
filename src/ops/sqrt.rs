//! Elementwise square root for scalars, strided vectors, and batched diagonals.
//!
//! Accepts `f32` views of rank 0 through 8. A scalar is a rank-0 view (one element at
//! `offset`). A batched diagonal is a rank-2 view of shape `[batch, n]` whose strides are
//! `[batch_stride, ld + 1]`: element `(b, i)` is diagonal entry `i` of matrix `b` when each
//! matrix is column-major with leading dimension `ld`. Any other non-negative stride pattern
//! of rank at most 8 is accepted, including a rank-1 strided vector.
//!
//! In-place means the same view is passed as source and destination (same buffer, offset,
//! shape, and strides). Any other byte-range overlap is [`LinalgError::Overlap`]. The
//! destination must be injective so parallel stores do not alias. A source may use a zero
//! stride (several logical elements read one address).
//!
//! Indexing is 32-bit. If the element count or any reachable index does not fit in `u32`,
//! the call returns [`LinalgError::InvalidArgument`] and launches nothing.
//!
//! Each finite result matches `(x as f64).sqrt() as f32` bitwise, including signed zeros.
//! `precise::sqrt` on the GPU flushes subnormal inputs to zero, so positive subnormals are
//! scaled by 2^48 in integer arithmetic, square-rooted, and scaled by 2^-24. NaNs match any
//! NaN; payloads are not required to equal the Rust cast (the GPU returns a canonical quiet
//! NaN for invalid inputs).

use std::ffi::c_void;

use metal::objc::{msg_send, sel, sel_impl};

use crate::abi::{compute_pipeline, LinalgError, MetalContext, Result, TensorView};

const SHADER: &str = include_str!("../../shaders/sqrt.metal");
const MODULE: &str = "sqrt";
const KERNEL: &str = "sqrt_f32";
/// Maximum rank accepted by [`sqrt`]. The params struct stores eight extents and stride pairs.
const MAX_RANK: usize = 8;

/// `#[repr(C)]` mirror of `SqrtParams` in `shaders/sqrt.metal`.
#[repr(C)]
#[derive(Clone, Copy)]
struct SqrtParams {
    src_offset: u32,
    dst_offset: u32,
    rank: u32,
    count: u32,
    shape: [u32; MAX_RANK],
    src_strides: [u32; MAX_RANK],
    dst_strides: [u32; MAX_RANK],
}

const _: () = assert!(std::mem::size_of::<SqrtParams>() == 112);
const _: () = assert!(std::mem::align_of::<SqrtParams>() == 4);

/// Writes the square root of `src` into `dst` and waits until the GPU has finished.
///
/// See the [module docs](self) for the view contract, the overlap rule, and rounding.
pub fn sqrt(
    ctx: &MetalContext,
    src: &TensorView<'_, f32>,
    dst: &TensorView<'_, f32>,
) -> Result<()> {
    let cmd = ctx.queue.new_command_buffer();
    encode_sqrt(ctx, cmd, src, dst)?;
    cmd.commit();
    cmd.wait_until_completed();
    if cmd.status() != metal::MTLCommandBufferStatus::Completed {
        return Err(LinalgError::InvalidArgument(format!(
            "sqrt command buffer finished with status {:?}",
            cmd.status()
        )));
    }
    Ok(())
}

/// Encodes the square root of `src` into `dst` on `cmd` without committing or waiting.
///
/// Creates and ends its own compute encoder. Does not read or write host memory, so several
/// primitives may encode into one command buffer. An empty view encodes nothing.
pub fn encode_sqrt(
    ctx: &MetalContext,
    cmd: &metal::CommandBufferRef,
    src: &TensorView<'_, f32>,
    dst: &TensorView<'_, f32>,
) -> Result<()> {
    let Some(params) = prepare(src, dst)? else {
        return Ok(());
    };
    let pipeline = compute_pipeline(ctx, MODULE, SHADER, compile_options, KERNEL)?;
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(src.buffer), 0);
    encoder.set_buffer(1, Some(dst.buffer), 0);
    encoder.set_bytes(
        2,
        std::mem::size_of::<SqrtParams>() as u64,
        &params as *const SqrtParams as *const c_void,
    );
    let groups = threadgroup_width(&pipeline, params.count);
    encoder.dispatch_threads(
        metal::MTLSize::new(u64::from(params.count), 1, 1),
        metal::MTLSize::new(groups, 1, 1),
    );
    encoder.end_encoding();
    Ok(())
}

fn compile_options() -> metal::CompileOptions {
    let opts = metal::CompileOptions::new();
    opts.set_fast_math_enabled(false);
    opts.set_language_version(metal::MTLLanguageVersion::V3_1);
    // SAFETY: `setMathMode:` and `setMathFloatingPointFunctions:` exist on MTLCompileOptions
    // since macOS 15. 0 is MTLMathModeSafe, 1 is MTLMathFloatingPointFunctionsPrecise.
    // `fastMathEnabled` is deprecated there; `mathMode` is the switch the compiler reads.
    // `sel_impl` still gates on `feature = "cargo-clippy"`, which this toolchain reports as
    // an unexpected cfg. The allow covers only that macro-generated check.
    #[allow(unexpected_cfgs)]
    unsafe {
        let _: () = msg_send![&*opts, setMathMode: 0isize];
        let _: () = msg_send![&*opts, setMathFloatingPointFunctions: 1isize];
    }
    opts
}

fn prepare(src: &TensorView<'_, f32>, dst: &TensorView<'_, f32>) -> Result<Option<SqrtParams>> {
    src.validate()?;
    dst.validate()?;
    if src.shape != dst.shape {
        return Err(LinalgError::InvalidArgument(format!(
            "sqrt shape mismatch: src {:?} vs dst {:?}",
            src.shape, dst.shape
        )));
    }
    if src.rank() > MAX_RANK {
        return Err(LinalgError::InvalidArgument(format!(
            "sqrt rank {} exceeds {MAX_RANK}",
            src.rank()
        )));
    }
    if !same_view(src, dst) && src.may_overlap(dst) {
        return Err(LinalgError::Overlap(
            "sqrt views overlap; in-place requires the same buffer, offset, shape, and strides"
                .into(),
        ));
    }
    if src.is_empty() {
        return Ok(None);
    }
    if !is_injective(&dst.shape, &dst.strides) {
        return Err(LinalgError::InvalidArgument(
            "sqrt destination is not injective; parallel stores would alias".into(),
        ));
    }
    let count = count_u32(&src.shape).ok_or_else(|| {
        LinalgError::InvalidArgument("sqrt element count does not fit in u32".into())
    })?;
    if !indexes_fit_u32(src) || !indexes_fit_u32(dst) {
        return Err(LinalgError::InvalidArgument(
            "sqrt index does not fit in u32; refusing to launch a 32-bit kernel".into(),
        ));
    }
    Ok(Some(SqrtParams {
        src_offset: src.offset as u32,
        dst_offset: dst.offset as u32,
        rank: src.rank() as u32,
        count,
        shape: pack(&src.shape),
        src_strides: pack(&src.strides),
        dst_strides: pack(&dst.strides),
    }))
}

fn same_view(a: &TensorView<'_, f32>, b: &TensorView<'_, f32>) -> bool {
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

fn indexes_fit_u32(view: &TensorView<'_, f32>) -> bool {
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
