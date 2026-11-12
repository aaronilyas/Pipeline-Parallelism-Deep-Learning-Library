//! Axis permutation views, contiguous materialization, and complex adjoint.

use std::ffi::c_void;
use std::mem::size_of;

use crate::abi::{
    compute_pipeline, row_major_strides, LinalgError, MetalContext, Result, TensorView,
};

/// Shader source, compiled at runtime and cached per device.
const SOURCE: &str = include_str!("../../shaders/permutation.metal");
/// Module name under which the shader library is cached.
const MODULE: &str = "permutation";
/// Maximum number of axes the kernels handle after adjacent contiguous axes are merged.
/// Mirrors `MAX_RANK` in the shader.
const MAX_RANK: usize = 8;
/// Edge of the square tile the transpose kernel stages through threadgroup memory.
const TILE: u64 = 32;
/// Rows of the transpose threadgroup; each thread moves `TILE / TILE_ROWS` tile rows.
const TILE_ROWS: u64 = 8;
/// Threads per threadgroup of the strided-copy kernel.
const COPY_GROUP: u64 = 256;
/// Kernels index with 32-bit arithmetic; every element index and count must stay below this.
const INDEX_LIMIT: usize = 1 << 31;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for u32 {}
    impl Sealed for i32 {}
    impl Sealed for [f32; 2] {}
}

/// Element types the permutation kernels move as raw bit patterns.
///
/// Four-byte types move as one 32-bit word and `[f32; 2]` (complex f32, `(re, im)`) as two.
/// The set is closed: it is sealed to the types the kernels are instantiated for.
pub trait Element: Copy + sealed::Sealed {
    /// True for complex types, whose adjoint and conjugate flip the imaginary sign bit.
    const COMPLEX: bool;
}

impl Element for f32 {
    const COMPLEX: bool = false;
}
impl Element for u32 {
    const COMPLEX: bool = false;
}
impl Element for i32 {
    const COMPLEX: bool = false;
}
impl Element for [f32; 2] {
    const COMPLEX: bool = true;
}

/// Parameters of the strided-copy kernels. Mirrors `CopyParams` in the shader.
#[repr(C)]
struct CopyParams {
    count: u32,
    rank: u32,
    src_offset: u32,
    dst_offset: u32,
    shape: [u32; MAX_RANK],
    src_strides: [u32; MAX_RANK],
}

/// Parameters of the tiled transpose kernels. Mirrors `TileParams` in the shader.
#[repr(C)]
struct TileParams {
    src_offset: u32,
    dst_offset: u32,
    extent_a: u32,
    extent_b: u32,
    src_stride_b: u32,
    dst_stride_a: u32,
    outer_rank: u32,
    outer_shape: [u32; MAX_RANK],
    outer_src: [u32; MAX_RANK],
    outer_dst: [u32; MAX_RANK],
}

/// Builds the options the shader is compiled with: pinned language version, no fast math.
fn compile_options() -> metal::CompileOptions {
    let options = metal::CompileOptions::new();
    options.set_language_version(metal::MTLLanguageVersion::V3_0);
    options.set_fast_math_enabled(false);
    options
}

fn invalid<S: Into<String>>(msg: S) -> LinalgError {
    LinalgError::InvalidArgument(msg.into())
}

fn to_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| invalid(format!("{what} {value} does not fit 32-bit indexing")))
}

fn narrow<const N: usize>(values: &[usize], what: &str) -> Result<[u32; N]> {
    let mut out = [0u32; N];
    for (slot, &v) in out.iter_mut().zip(values) {
        *slot = to_u32(v, what)?;
    }
    Ok(out)
}

/// Drops unit axes and merges adjacent axes that are contiguous in the source, so the
/// destination (row-major over `shape`) and the source are described by the fewest axes.
/// Never returns an empty list: a single element becomes one axis of extent 1.
fn coalesce(shape: &[usize], strides: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut out_shape: Vec<usize> = Vec::new();
    let mut out_strides: Vec<usize> = Vec::new();
    for (&n, &s) in shape.iter().zip(strides) {
        if n == 1 {
            continue;
        }
        if let (Some(outer_n), Some(outer_s)) = (out_shape.last_mut(), out_strides.last_mut()) {
            if s.checked_mul(n) == Some(*outer_s) {
                *outer_n *= n;
                *outer_s = s;
                continue;
            }
        }
        out_shape.push(n);
        out_strides.push(s);
    }
    if out_shape.is_empty() {
        out_shape.push(1);
        out_strides.push(1);
    }
    (out_shape, out_strides)
}

/// Kernel entry point for a path, element width in 32-bit words, and conjugation.
fn kernel_name(tiled: bool, words: usize, conj: bool) -> &'static str {
    match (tiled, words, conj) {
        (false, 1, _) => "permutation_copy_w1",
        (false, _, false) => "permutation_copy_w2",
        (false, _, true) => "permutation_copy_conj_w2",
        (true, 1, _) => "permutation_tile_w1",
        (true, _, false) => "permutation_tile_w2",
        (true, _, true) => "permutation_tile_conj_w2",
    }
}

fn encode_params<P>(
    cmd: &metal::CommandBufferRef,
    pso: &metal::ComputePipelineStateRef,
    src: &metal::BufferRef,
    dst: &metal::BufferRef,
    params: &P,
    dispatch: impl FnOnce(&metal::ComputeCommandEncoderRef),
) {
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pso);
    enc.set_buffer(0, Some(src), 0);
    enc.set_buffer(1, Some(dst), 0);
    enc.set_bytes(2, size_of::<P>() as u64, params as *const P as *const c_void);
    dispatch(enc);
    enc.end_encoding();
}

/// Validates the pair and encodes `dst = [conj] src` with `dst` contiguous row-major.
fn encode_gather<T: Element>(
    ctx: &MetalContext,
    cmd: &metal::CommandBufferRef,
    src: &TensorView<'_, T>,
    dst: &TensorView<'_, T>,
    conj: bool,
) -> Result<()> {
    src.validate()?;
    dst.validate()?;
    if src.shape != dst.shape {
        return Err(invalid(format!(
            "destination shape {:?} does not match source shape {:?}",
            dst.shape, src.shape
        )));
    }
    let dense = row_major_strides(&dst.shape);
    let contiguous =
        dst.shape.iter().zip(dst.strides.iter().zip(&dense)).all(|(&n, (s, d))| n == 1 || s == d);
    if !contiguous {
        return Err(invalid(format!(
            "destination must be contiguous row-major, got strides {:?} for shape {:?}",
            dst.strides, dst.shape
        )));
    }
    if src.may_overlap(dst) {
        return Err(LinalgError::Overlap("destination overlaps the source".into()));
    }
    let (Some(src_range), Some(dst_range)) = (src.element_range(), dst.element_range()) else {
        return Ok(());
    };
    if src_range.end > INDEX_LIMIT || dst_range.end > INDEX_LIMIT {
        return Err(invalid(
            "view reaches beyond 2^31 elements, which 32-bit indexing cannot address",
        ));
    }

    let (shape, strides) = coalesce(&src.shape, &src.strides);
    let rank = shape.len();
    if rank > MAX_RANK {
        return Err(invalid(format!(
            "rank {rank} exceeds {MAX_RANK} after merging contiguous axes"
        )));
    }
    let words = size_of::<T>() / size_of::<u32>();
    let conj = conj && T::COMPLEX;
    let dst_strides = row_major_strides(&shape);
    let src_offset = to_u32(src.offset, "source offset")?;
    let dst_offset = to_u32(dst.offset, "destination offset")?;

    // The destination's unit-stride axis is the last one. When the source is not unit
    // stride there but has a unit-stride axis elsewhere, transpose tiles of those two axes.
    let b = rank - 1;
    let tile_axis = if strides[b] == 1 { None } else { (0..b).find(|&a| strides[a] == 1) };

    let Some(a) = tile_axis else {
        let params = CopyParams {
            count: to_u32(dst.len(), "element count")?,
            rank: rank as u32,
            src_offset,
            dst_offset,
            shape: narrow(&shape, "extent")?,
            src_strides: narrow(&strides, "source stride")?,
        };
        let pso = compute_pipeline(
            ctx,
            MODULE,
            SOURCE,
            compile_options,
            kernel_name(false, words, conj),
        )?;
        let group = COPY_GROUP.min(pso.max_total_threads_per_threadgroup());
        encode_params(cmd, &pso, src.buffer, dst.buffer, &params, |enc| {
            enc.dispatch_threads(
                metal::MTLSize::new(dst.len() as u64, 1, 1),
                metal::MTLSize::new(group, 1, 1),
            );
        });
        return Ok(());
    };

    let outer: Vec<usize> = (0..rank).filter(|&d| d != a && d != b).collect();
    let pick = |from: &[usize]| outer.iter().map(|&d| from[d]).collect::<Vec<_>>();
    let outer_shape = pick(&shape);
    let params = TileParams {
        src_offset,
        dst_offset,
        extent_a: to_u32(shape[a], "extent")?,
        extent_b: to_u32(shape[b], "extent")?,
        src_stride_b: to_u32(strides[b], "source stride")?,
        dst_stride_a: to_u32(dst_strides[a], "destination stride")?,
        outer_rank: outer.len() as u32,
        outer_shape: narrow(&outer_shape, "extent")?,
        outer_src: narrow(&pick(&strides), "source stride")?,
        outer_dst: narrow(&pick(&dst_strides), "destination stride")?,
    };
    let groups = metal::MTLSize::new(
        (shape[b] as u64).div_ceil(TILE),
        (shape[a] as u64).div_ceil(TILE),
        outer_shape.iter().product::<usize>() as u64,
    );
    let pso =
        compute_pipeline(ctx, MODULE, SOURCE, compile_options, kernel_name(true, words, conj))?;
    encode_params(cmd, &pso, src.buffer, dst.buffer, &params, |enc| {
        enc.dispatch_thread_groups(groups, metal::MTLSize::new(TILE, TILE_ROWS, 1));
    });
    Ok(())
}

/// Commits and waits for a command buffer that `encode` filled.
fn run_sync(
    ctx: &MetalContext,
    encode: impl FnOnce(&metal::CommandBufferRef) -> Result<()>,
) -> Result<()> {
    let cmd = ctx.queue.new_command_buffer();
    encode(cmd)?;
    cmd.commit();
    cmd.wait_until_completed();
    Ok(())
}

/// `src` with its last two axes swapped, or an error if it has fewer than two axes.
fn swap_last_two<'a, T>(src: &TensorView<'a, T>) -> Result<TensorView<'a, T>> {
    let rank = src.rank();
    if rank < 2 {
        return Err(invalid(format!("adjoint needs at least 2 axes, got {rank}")));
    }
    let mut perm: Vec<usize> = (0..rank).collect();
    perm.swap(rank - 2, rank - 1);
    permute_view(src, &perm)
}

/// Reorders the axes of `view` without touching memory (O(rank), no GPU work).
///
/// Axis `i` of the result is axis `perm[i]` of `view`. The buffer, `host_ptr` and offset
/// are unchanged. Fails if `perm` is not a permutation of `0..view.rank()`.
pub fn permute_view<'a, T>(view: &TensorView<'a, T>, perm: &[usize]) -> Result<TensorView<'a, T>> {
    let rank = view.shape.len();
    if view.strides.len() != rank {
        return Err(invalid(format!(
            "rank mismatch: shape {:?} vs strides {:?}",
            view.shape, view.strides
        )));
    }
    let mut seen = vec![false; rank];
    let is_permutation = perm.len() == rank
        && perm.iter().all(|&p| p < rank && !std::mem::replace(&mut seen[p], true));
    if !is_permutation {
        return Err(invalid(format!("{perm:?} is not a permutation of 0..{rank}")));
    }
    Ok(TensorView {
        buffer: view.buffer,
        host_ptr: view.host_ptr,
        offset: view.offset,
        shape: perm.iter().map(|&p| view.shape[p]).collect(),
        strides: perm.iter().map(|&p| view.strides[p]).collect(),
    })
}

/// Copies `src` (any strided view) into the row-major view `dst` of the same shape and
/// waits for completion.
///
/// Fails on invalid metadata or if `dst` may overlap `src`.
pub fn materialize<T: Element>(
    ctx: &MetalContext,
    src: &TensorView<'_, T>,
    dst: &TensorView<'_, T>,
) -> Result<()> {
    run_sync(ctx, |cmd| encode_materialize(ctx, cmd, src, dst))
}

/// Encodes [`materialize`] into `cmd` without committing or waiting.
pub fn encode_materialize<T: Element>(
    ctx: &MetalContext,
    cmd: &metal::CommandBufferRef,
    src: &TensorView<'_, T>,
    dst: &TensorView<'_, T>,
) -> Result<()> {
    encode_gather(ctx, cmd, src, dst, false)
}

/// Writes the conjugate of `src` into the row-major view `dst` and waits for completion.
///
/// Conjugation XORs the imaginary sign bit, so it is bitwise self-inverse; for real
/// element types it is a plain copy.
pub fn conjugate<T: Element>(
    ctx: &MetalContext,
    src: &TensorView<'_, T>,
    dst: &TensorView<'_, T>,
) -> Result<()> {
    run_sync(ctx, |cmd| encode_conjugate(ctx, cmd, src, dst))
}

/// Encodes [`conjugate`] into `cmd` without committing or waiting.
pub fn encode_conjugate<T: Element>(
    ctx: &MetalContext,
    cmd: &metal::CommandBufferRef,
    src: &TensorView<'_, T>,
    dst: &TensorView<'_, T>,
) -> Result<()> {
    encode_gather(ctx, cmd, src, dst, true)
}

/// Writes the batched adjoint of `src` (shape `[.., m, n]`) into the row-major view `dst`
/// (shape `[.., n, m]`) and waits for completion.
///
/// The adjoint transposes the last two axes and conjugates complex elements; for real
/// types it is the transpose. Fails if `dst` may overlap `src`.
pub fn adjoint<T: Element>(
    ctx: &MetalContext,
    src: &TensorView<'_, T>,
    dst: &TensorView<'_, T>,
) -> Result<()> {
    run_sync(ctx, |cmd| encode_adjoint(ctx, cmd, src, dst))
}

/// Encodes [`adjoint`] into `cmd` without committing or waiting.
pub fn encode_adjoint<T: Element>(
    ctx: &MetalContext,
    cmd: &metal::CommandBufferRef,
    src: &TensorView<'_, T>,
    dst: &TensorView<'_, T>,
) -> Result<()> {
    encode_gather(ctx, cmd, &swap_last_two(src)?, dst, true)
}
