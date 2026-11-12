//! Index of the element of largest magnitude in a strided 1-D view (pivot selection).
//!
//! `k = argmaxᵢ |xᵢ|` over a rank-1 [`TensorView`], written as a 0-based `u32` into a
//! device-resident output slot. Real input is `f32`; complex input is `[f32; 2]` =
//! `(re, im)` ranked by a [`Magnitude`].
//!
//! Ordering (identical in every tier of the kernel, hence bitwise deterministic):
//! a NaN magnitude outranks every number; otherwise the larger magnitude wins; equal
//! magnitudes (including `-0` and `+0`, and equal infinities) go to the lowest index. Among
//! several NaNs the lowest index wins.
//!
//! # Execution
//!
//! Lengths up to [`SINGLE_PASS_MAX_LEN`] run one single-threadgroup dispatch that writes the
//! output slot directly. Longer inputs run two dispatches inside one compute encoder: a
//! first pass of up to [`MAX_GROUPS`] threadgroups, each emitting one candidate into module
//! scratch, and a second pass in which a single threadgroup folds the candidates.
//!
//! # Scratch
//!
//! Only the two-pass path uses scratch: [`MAX_GROUPS`] candidates, held in a per-device pool.
//! A scratch buffer belongs to the command buffer it was last encoded into and is handed to
//! another command buffer only after the owner has completed (or failed), so concurrent
//! command buffers and concurrent threads never share it. Several calls encoded into the
//! *same* command buffer share one scratch buffer; Metal's automatic hazard tracking of the
//! (default, tracked) shared buffer orders the encoders that touch it, so each call sees only
//! its own candidates. A command buffer that is encoded into and then abandoned without
//! being committed keeps its scratch buffer reserved.

use std::collections::HashMap;
use std::mem::size_of;
use std::sync::{Mutex, OnceLock};

use metal::foreign_types::ForeignTypeRef;
use metal::{MTLCommandBufferStatus, MTLLanguageVersion, MTLSize};

use crate::abi::{alloc_shared, compute_pipeline, LinalgError, MetalContext, Result, TensorView};

/// Threads per threadgroup of every argmax kernel.
pub const THREADGROUP_SIZE: usize = 256;

/// Largest length handled by the single-dispatch, single-threadgroup fast path.
pub const SINGLE_PASS_MAX_LEN: usize = 4096;

/// Upper bound on threadgroups launched by the first pass of the two-pass path.
pub const MAX_GROUPS: usize = 1024;

/// Elements each thread is expected to accumulate before the first pass stops adding groups.
const ITEMS_PER_THREAD: usize = 4;

const MODULE: &str = "argmax";
const SOURCE: &str = include_str!("../../shaders/argmax.metal");

/// How the magnitude of a complex element is ranked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Magnitude {
    /// `|re| + |im|`, computed in `f32` (saturates to `+inf` on overflow).
    /// Subnormal parts are included; they are not flushed to zero.
    L1,
    /// `max · sqrt(1 + (min/max)²)` with `max = max(|re|, |im|)`, computed in `f32`.
    /// Subnormal parts are included; they are not flushed to zero.
    Hypot,
}

/// Launch parameters; mirrors `Params` in `shaders/argmax.metal` field for field.
#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    offset: u64,
    stride: u64,
    out_offset: u64,
    n: u32,
    groups: u32,
}

/// One first-pass winner in scratch; mirrors `Candidate` in `shaders/argmax.metal`.
#[repr(C)]
struct Candidate(#[allow(dead_code)] u64);

/// Argmax of `|x|` over a real view; commits and waits. See [`encode_argmax`].
pub fn argmax(ctx: &MetalContext, x: &TensorView<f32>, out: &TensorView<u32>) -> Result<()> {
    run_sync(ctx, |cmd| encode_argmax(ctx, cmd, x, out))
}

/// Encodes the argmax of `|x|` over a real view into `cmd` without committing.
///
/// `x` must be a rank-1 view with `1 ..= u32::MAX` elements; `out` must hold exactly one
/// element (rank 0, or rank 1 with shape `[1]`) and must not overlap `x`. On success the
/// 0-based index of the largest `|x|` is written to `out` once `cmd` completes. See the module
/// docs for the ordering rules and the scratch guarantee.
///
/// # Errors
/// [`LinalgError::InvalidArgument`] for malformed or out-of-range metadata,
/// [`LinalgError::Overlap`] if `out` overlaps `x`, [`LinalgError::ShaderCompile`] if the
/// kernels cannot be built. Nothing is encoded on error.
pub fn encode_argmax(
    ctx: &MetalContext,
    cmd: &metal::CommandBufferRef,
    x: &TensorView<f32>,
    out: &TensorView<u32>,
) -> Result<()> {
    encode(ctx, cmd, x, out, "f32")
}

/// Argmax of the complex magnitude of `x`; commits and waits. See [`encode_argmax_complex`].
pub fn argmax_complex(
    ctx: &MetalContext,
    x: &TensorView<[f32; 2]>,
    magnitude: Magnitude,
    out: &TensorView<u32>,
) -> Result<()> {
    run_sync(ctx, |cmd| encode_argmax_complex(ctx, cmd, x, magnitude, out))
}

/// Encodes the argmax of the complex magnitude of `x` into `cmd` without committing.
///
/// Elements are `(re, im)` pairs ranked by `magnitude`; a pair with a NaN part has NaN
/// magnitude. Requirements, results and errors are those of [`encode_argmax`].
pub fn encode_argmax_complex(
    ctx: &MetalContext,
    cmd: &metal::CommandBufferRef,
    x: &TensorView<[f32; 2]>,
    magnitude: Magnitude,
    out: &TensorView<u32>,
) -> Result<()> {
    let variant = match magnitude {
        Magnitude::L1 => "c32_l1",
        Magnitude::Hypot => "c32_hypot",
    };
    encode(ctx, cmd, x, out, variant)
}

/// Encodes into a fresh command buffer, commits, and waits.
fn run_sync(
    ctx: &MetalContext,
    encode_into: impl FnOnce(&metal::CommandBufferRef) -> Result<()>,
) -> Result<()> {
    metal::objc::rc::autoreleasepool(|| {
        let cmd = ctx.queue.new_command_buffer();
        encode_into(cmd)?;
        cmd.commit();
        cmd.wait_until_completed();
        match cmd.status() {
            MTLCommandBufferStatus::Completed => Ok(()),
            status => Err(LinalgError::InvalidArgument(format!(
                "argmax command buffer did not complete: {status:?}"
            ))),
        }
    })
}

/// Validates the views and encodes the kernels of `variant` (`f32`, `c32_l1`, `c32_hypot`).
fn encode<T>(
    ctx: &MetalContext,
    cmd: &metal::CommandBufferRef,
    x: &TensorView<T>,
    out: &TensorView<u32>,
    variant: &str,
) -> Result<()> {
    let invalid = |msg: &str| Err(LinalgError::InvalidArgument(msg.to_owned()));
    if x.rank() != 1 {
        return invalid(&format!("argmax input must be rank 1, got shape {:?}", x.shape));
    }
    let len = x.shape[0];
    if len == 0 {
        return invalid("argmax input is empty");
    }
    let Ok(n) = u32::try_from(len) else {
        return invalid(&format!("argmax input length {len} exceeds u32::MAX"));
    };
    x.validate()?;
    out.validate()?;
    if out.rank() > 1 || out.len() != 1 {
        return invalid(&format!("argmax output must hold one element, got shape {:?}", out.shape));
    }
    if x.may_overlap(out) {
        return Err(LinalgError::Overlap("argmax output overlaps its input".to_owned()));
    }

    let single = len <= SINGLE_PASS_MAX_LEN;
    let groups = if single {
        1
    } else {
        len.div_ceil(THREADGROUP_SIZE * ITEMS_PER_THREAD).clamp(1, MAX_GROUPS)
    };
    let mut params = Params {
        offset: x.offset as u64,
        stride: x.strides[0] as u64,
        out_offset: out.offset as u64,
        n,
        groups: groups as u32,
    };
    let first =
        pipeline(ctx, &format!("argmax_{variant}_{}", if single { "single" } else { "partial" }))?;
    let last = if single { None } else { Some(pipeline(ctx, "argmax_final")?) };
    let scratch = if single { None } else { Some(acquire_scratch(ctx, cmd)) };

    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&first);
    encoder.set_buffer(0, Some(x.buffer), 0);
    encoder.set_buffer(1, Some(scratch.as_deref().unwrap_or(out.buffer)), 0);
    encoder.set_bytes(2, size_of::<Params>() as u64, (&mut params as *mut Params).cast());
    let threads = MTLSize::new(THREADGROUP_SIZE as u64, 1, 1);
    encoder.dispatch_thread_groups(MTLSize::new(groups as u64, 1, 1), threads);
    if let (Some(last), Some(scratch)) = (last, scratch.as_deref()) {
        encoder.set_compute_pipeline_state(&last);
        encoder.set_buffer(0, Some(scratch), 0);
        encoder.set_buffer(1, Some(out.buffer), 0);
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), threads);
    }
    encoder.end_encoding();
    Ok(())
}

/// Fetches a kernel from the per-device cache, checking it can run a full threadgroup.
fn pipeline(ctx: &MetalContext, function: &str) -> Result<metal::ComputePipelineState> {
    let pipeline = compute_pipeline(ctx, MODULE, SOURCE, compile_options, function)?;
    if (pipeline.max_total_threads_per_threadgroup() as usize) < THREADGROUP_SIZE {
        return Err(LinalgError::ShaderCompile(format!(
            "{MODULE}::{function}: device cannot run {THREADGROUP_SIZE} threads per threadgroup"
        )));
    }
    Ok(pipeline)
}

/// Strict IEEE math so NaN, infinity, and signed-zero handling survives compilation.
fn compile_options() -> metal::CompileOptions {
    let options = metal::CompileOptions::new();
    options.set_fast_math_enabled(false);
    options.set_language_version(MTLLanguageVersion::V3_0);
    options
}

/// A scratch buffer in the per-device pool and the command buffer that last used it.
struct ScratchSlot {
    buffer: metal::Buffer,
    owner: metal::CommandBuffer,
}

/// Per-device scratch pools, keyed by the device registry id.
fn pools() -> &'static Mutex<HashMap<u64, Vec<ScratchSlot>>> {
    static POOLS: OnceLock<Mutex<HashMap<u64, Vec<ScratchSlot>>>> = OnceLock::new();
    POOLS.get_or_init(Default::default)
}

/// Returns a scratch buffer that is safe to use from `cmd`: the buffer `cmd` already owns,
/// else one whose owner has finished, else a newly allocated one.
fn acquire_scratch(ctx: &MetalContext, cmd: &metal::CommandBufferRef) -> metal::Buffer {
    let mut pools = pools().lock().unwrap_or_else(|e| e.into_inner());
    let slots = pools.entry(ctx.device.registry_id()).or_default();
    let finished = |s: &ScratchSlot| {
        matches!(
            s.owner.status(),
            MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
        )
    };
    let reusable = slots
        .iter()
        .position(|s| s.owner.as_ptr() == cmd.as_ptr())
        .or_else(|| slots.iter().position(finished));
    match reusable {
        Some(i) => {
            slots[i].owner = cmd.to_owned();
            slots[i].buffer.clone()
        }
        None => {
            let buffer = alloc_shared::<Candidate>(ctx, MAX_GROUPS);
            slots.push(ScratchSlot { buffer: buffer.clone(), owner: cmd.to_owned() });
            buffer
        }
    }
}
