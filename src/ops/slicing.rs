//! Zero-copy slicing views plus GPU packing and concatenation.
//!
//! [`slice`] adjusts view metadata on the CPU: the result borrows the same buffer and
//! `host_ptr` (the buffer base), and its offset grows by `Σ start · stride`. [`pack`] copies
//! one view onto another of the same shape, and [`concat`] stitches parts along one axis.
//! Both are raw bit copies for 2, 4, 8, and 16-byte elements, so NaN payloads and `-0.0`
//! survive. The `encode_*` functions record one compute encoder and do not commit, wait, or
//! read host memory; the plain wrappers commit and wait.
//!
//! Kernel indexing is 32-bit. A view whose reachable indices do not fit returns
//! [`crate::abi::LinalgError::InvalidArgument`] instead of launching. The vector pack path
//! copies `ulong4` (32 bytes) then `ulong2` (16 bytes), with a scalar head and tail in the
//! same dispatch, when the inner stride is 1 and the two rows share a 16-byte alignment.

use std::ffi::c_void;
use std::mem::size_of;
use std::ops::Range;

use crate::abi::{compute_pipeline, LinalgError, MetalContext, Result, TensorView};

/// Shader source, compiled at runtime and cached per device under [`MODULE`].
const SOURCE: &str = include_str!("../../shaders/slicing.metal");
/// Module name under which the shader library is cached.
const MODULE: &str = "slicing";
/// Maximum rank accepted by [`pack`] and [`concat`]. Mirrors `MAX_RANK` in the shader.
pub const MAX_RANK: usize = 8;
/// Maximum number of parts in one [`concat`] dispatch. Mirrors `MAX_CONCAT_PARTS`.
pub const MAX_CONCAT_PARTS: usize = 16;
/// Threads per threadgroup for the 1D kernels, capped by the device limit.
const COPY_GROUP: u64 = 256;
/// Kernels index with 32-bit arithmetic. Every reachable element index stays below this.
const INDEX_LIMIT: usize = 1 << 31;

/// Parameters of the scalar pack kernels. Mirrors `PackParams` in the shader.
#[repr(C)]
struct PackParams {
    count: u32,
    rank: u32,
    src_offset: u32,
    dst_offset: u32,
    shape: [u32; MAX_RANK],
    src_strides: [u32; MAX_RANK],
    dst_strides: [u32; MAX_RANK],
}

/// Parameters of the vector pack kernels. Mirrors `VecParams` in the shader.
#[repr(C)]
struct VecParams {
    n_rows: u32,
    inner: u32,
    src_offset: u32,
    dst_offset: u32,
    outer_rank: u32,
    threads_per_row: u32,
    head: u32,
    n32: u32,
    n16: u32,
    tail: u32,
    outer_shape: [u32; MAX_RANK],
    src_outer: [u32; MAX_RANK],
    dst_outer: [u32; MAX_RANK],
}

/// Parameters of the concat kernels. Mirrors `ConcatParams` in the shader.
///
/// The part id for each coordinate along the concat axis is a device table (buffer 18),
/// not a field here, because that table grows with the destination extent and can exceed
/// the `set_bytes` size limit. Empty parts carry zero offset and strides: no coordinate
/// ever selects them.
#[repr(C)]
struct ConcatParams {
    count: u32,
    rank: u32,
    axis: u32,
    n_parts: u32,
    dst_offset: u32,
    dst_shape: [u32; MAX_RANK],
    dst_strides: [u32; MAX_RANK],
    part_offset: [u32; MAX_CONCAT_PARTS],
    part_strides: [[u32; MAX_RANK]; MAX_CONCAT_PARTS],
}

const _: () = assert!(size_of::<PackParams>() == 112);
const _: () = assert!(size_of::<VecParams>() == 136);
const _: () = assert!(size_of::<ConcatParams>() == 660);

/// Scalar head, 32-byte body, 16-byte body, and scalar tail of one unit-stride row.
struct RowSplit {
    head: u32,
    n32: u32,
    n16: u32,
    tail: u32,
    threads_per_row: u32,
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
    if values.len() > N {
        return Err(invalid(format!("{what} has {} entries, the limit is {N}", values.len())));
    }
    let mut out = [0u32; N];
    for (slot, &value) in out.iter_mut().zip(values) {
        *slot = to_u32(value, what)?;
    }
    Ok(out)
}

fn narrow_opt<const N: usize>(values: &[usize]) -> Option<[u32; N]> {
    if values.len() > N {
        return None;
    }
    let mut out = [0u32; N];
    for (slot, &value) in out.iter_mut().zip(values) {
        *slot = u32::try_from(value).ok()?;
    }
    Some(out)
}

fn logical_len(shape: &[usize]) -> Option<usize> {
    shape.iter().try_fold(1usize, |acc, &extent| acc.checked_mul(extent))
}

fn elem_size<T>() -> Result<usize> {
    match size_of::<T>() {
        n @ (2 | 4 | 8 | 16) => Ok(n),
        n => Err(invalid(format!("element size {n} is not 2, 4, 8, or 16 bytes"))),
    }
}

fn kernel_name(op: &str, elem: usize) -> Result<&'static str> {
    Ok(match (op, elem) {
        ("pack", 2) => "slicing_pack_e2",
        ("pack", 4) => "slicing_pack_e4",
        ("pack", 8) => "slicing_pack_e8",
        ("pack", 16) => "slicing_pack_e16",
        ("vec", 2) => "slicing_pack_vec_e2",
        ("vec", 4) => "slicing_pack_vec_e4",
        ("vec", 8) => "slicing_pack_vec_e8",
        ("vec", 16) => "slicing_pack_vec_e16",
        ("concat", 2) => "slicing_concat_e2",
        ("concat", 4) => "slicing_concat_e4",
        ("concat", 8) => "slicing_concat_e8",
        ("concat", 16) => "slicing_concat_e16",
        _ => return Err(invalid(format!("no {op} kernel for {elem}-byte elements"))),
    })
}

/// Product of the extents, rejecting a count that 32-bit linear indexing cannot name.
fn fits<T>(view: &TensorView<'_, T>, what: &str) -> Result<()> {
    to_u32(view.offset, &format!("{what} offset"))?;
    for &extent in &view.shape {
        to_u32(extent, &format!("{what} extent"))?;
    }
    for &stride in &view.strides {
        to_u32(stride, &format!("{what} stride"))?;
    }
    let len = logical_len(&view.shape)
        .ok_or_else(|| invalid(format!("{what} element count overflows")))?;
    if len >= INDEX_LIMIT {
        return Err(invalid(format!("{what} has {len} elements, past the 2^31 index limit")));
    }
    if let Some(range) = view.element_range() {
        if range.end > INDEX_LIMIT {
            return Err(invalid(format!(
                "{what} reaches element {}, past the 2^31 index limit",
                range.end
            )));
        }
    } else if !view.is_empty() {
        return Err(invalid(format!("{what} index overflow")));
    }
    Ok(())
}

/// True when logical element `i` of a non-empty view lives at element index `offset + i`.
///
/// Unit extents are skipped. Column-major layouts are not contiguous: the last axis must
/// have stride 1 and each outer axis must follow with no gap and no overlap.
fn is_contiguous<T>(view: &TensorView<'_, T>) -> bool {
    let mut expect = 1usize;
    for (&extent, &stride) in view.shape.iter().zip(view.strides.iter()).rev() {
        if extent <= 1 {
            continue;
        }
        if stride != expect {
            return false;
        }
        let Some(next) = expect.checked_mul(extent) else {
            return false;
        };
        expect = next;
    }
    true
}

fn base_aligned(buffer: &metal::BufferRef) -> bool {
    buffer.contents() as usize % 16 == 0
}

fn misalign_of(offset: usize, elem: usize) -> Option<usize> {
    offset.checked_mul(elem).map(|bytes| bytes % 16)
}

/// True when advancing by `stride` elements keeps a 16-byte alignment. A zero stride does.
fn preserves_alignment(stride: usize, elem: usize) -> bool {
    stride.checked_mul(elem).is_some_and(|bytes| bytes % 16 == 0)
}

/// True when two distinct multi-indices can name the same element.
///
/// Axes of extent 0 or 1 are ignored. The rest are ordered by increasing stride; a stride
/// narrower than the span already covered collides, and so does an overflowing span. Some
/// injective layouts whose strides interleave are rejected as well: a parallel store must
/// not guess about them. Sources may alias; only the destination is checked.
fn destination_aliases<T>(view: &TensorView<'_, T>) -> bool {
    let mut axes: Vec<usize> = (0..view.rank()).filter(|&axis| view.shape[axis] > 1).collect();
    axes.sort_by_key(|&axis| view.strides[axis]);
    let mut span = 1usize;
    for axis in axes {
        let stride = view.strides[axis];
        if stride < span {
            return true;
        }
        let Some(reach) = stride.checked_mul(view.shape[axis] - 1) else {
            return true;
        };
        let Some(next) = reach.checked_add(span) else {
            return true;
        };
        span = next;
    }
    false
}

/// Splits one unit-stride row into a scalar head, `ulong4` body, `ulong2` body, and tail.
///
/// `misalign` is the row start's byte offset modulo 16. The head is the fewest elements that
/// reach a 16-byte boundary, or the whole row when it ends first. `None` means the caller
/// must use the scalar kernel. A 16-byte element never has a head or a tail; a leftover
/// element is one `ulong2`. `n16` is 0 or 1.
fn row_layout(misalign: usize, inner: usize, elem: usize) -> Option<RowSplit> {
    if !matches!(elem, 2 | 4 | 8 | 16) || misalign >= 16 || inner == 0 {
        return None;
    }
    let per16 = 16 / elem;
    let per32 = 32 / elem;
    let head = if misalign == 0 {
        0
    } else {
        let gap = 16 - misalign;
        if gap % elem != 0 {
            return None;
        }
        inner.min(gap / elem)
    };
    if head < inner && (misalign + head * elem) % 16 != 0 {
        return None;
    }
    let rest = inner - head;
    let n32 = rest / per32;
    let rem = rest % per32;
    let n16 = rem / per16;
    let tail = rem % per16;
    let threads = head + n32 + n16 + tail;
    Some(RowSplit {
        head: u32::try_from(head).ok()?,
        n32: u32::try_from(n32).ok()?,
        n16: u32::try_from(n16).ok()?,
        tail: u32::try_from(tail).ok()?,
        threads_per_row: u32::try_from(threads).ok()?,
    })
}

/// Outer axes of a vector-pack plan, outermost first.
struct OuterAxes<'a> {
    shape: &'a [usize],
    src: &'a [usize],
    dst: &'a [usize],
}

fn assemble_vec(
    src_offset: usize,
    dst_offset: usize,
    inner: usize,
    n_rows: usize,
    outer: OuterAxes<'_>,
    split: RowSplit,
) -> Option<VecParams> {
    Some(VecParams {
        n_rows: u32::try_from(n_rows).ok()?,
        inner: u32::try_from(inner).ok()?,
        src_offset: u32::try_from(src_offset).ok()?,
        dst_offset: u32::try_from(dst_offset).ok()?,
        outer_rank: u32::try_from(outer.shape.len()).ok()?,
        threads_per_row: split.threads_per_row,
        head: split.head,
        n32: split.n32,
        n16: split.n16,
        tail: split.tail,
        outer_shape: narrow_opt(outer.shape)?,
        src_outer: narrow_opt(outer.src)?,
        dst_outer: narrow_opt(outer.dst)?,
    })
}

/// One contiguous run of `len` elements, when both views are logically contiguous and share
/// a 16-byte misalignment. Element `i` then lives at `offset + i` on both sides.
fn plan_linear<T>(
    src: &TensorView<'_, T>,
    dst: &TensorView<'_, T>,
    elem: usize,
    misalign: usize,
) -> Option<VecParams> {
    if !is_contiguous(src) || !is_contiguous(dst) {
        return None;
    }
    let inner = logical_len(&src.shape)?;
    if inner == 0 {
        return None;
    }
    let split = row_layout(misalign, inner, elem)?;
    assemble_vec(
        src.offset,
        dst.offset,
        inner,
        1,
        OuterAxes { shape: &[], src: &[], dst: &[] },
        split,
    )
}

/// One row per outer index when the last stride is 1 on both sides and every outer stride
/// preserves 16-byte alignment. Row 0 starts at each view's offset, so equal misalignments
/// stay equal on later rows.
fn plan_rows<T>(
    src: &TensorView<'_, T>,
    dst: &TensorView<'_, T>,
    elem: usize,
    misalign: usize,
) -> Option<VecParams> {
    let rank = src.rank();
    if rank == 0 || src.strides[rank - 1] != 1 || dst.strides[rank - 1] != 1 {
        return None;
    }
    let inner = src.shape[rank - 1];
    if inner == 0 {
        return None;
    }
    let mut n_rows = 1usize;
    let mut outer_shape = Vec::with_capacity(rank - 1);
    let mut src_outer = Vec::with_capacity(rank - 1);
    let mut dst_outer = Vec::with_capacity(rank - 1);
    for axis in 0..rank - 1 {
        if !preserves_alignment(src.strides[axis], elem)
            || !preserves_alignment(dst.strides[axis], elem)
        {
            return None;
        }
        n_rows = n_rows.checked_mul(src.shape[axis])?;
        outer_shape.push(src.shape[axis]);
        src_outer.push(src.strides[axis]);
        dst_outer.push(dst.strides[axis]);
    }
    if n_rows == 0 {
        return None;
    }
    let split = row_layout(misalign, inner, elem)?;
    assemble_vec(
        src.offset,
        dst.offset,
        inner,
        n_rows,
        OuterAxes { shape: &outer_shape, src: &src_outer, dst: &dst_outer },
        split,
    )
}

/// Vector plan, or `None` when the scalar kernel must be used.
///
/// A scalar head can bring both addresses onto a 16-byte boundary only when the rows start
/// at the same misalignment. Differing misalignments, a non-unit inner stride, or a base
/// that is not 16-byte aligned all take the scalar path.
fn vector_plan<T>(
    src: &TensorView<'_, T>,
    dst: &TensorView<'_, T>,
    elem: usize,
) -> Option<VecParams> {
    if !base_aligned(src.buffer) || !base_aligned(dst.buffer) {
        return None;
    }
    let misalign = misalign_of(src.offset, elem)?;
    if misalign != misalign_of(dst.offset, elem)? {
        return None;
    }
    plan_linear(src, dst, elem, misalign).or_else(|| plan_rows(src, dst, elem, misalign))
}

/// Rank 0 is one element. The scalar kernel's loop requires rank at least 1, with a zero
/// stride so that single element stays at `offset`.
fn expand(shape: &[usize], strides: &[usize]) -> (Vec<usize>, Vec<usize>) {
    if shape.is_empty() {
        (vec![1], vec![0])
    } else {
        (shape.to_vec(), strides.to_vec())
    }
}

fn scalar_params<T>(src: &TensorView<'_, T>, dst: &TensorView<'_, T>) -> Result<PackParams> {
    let (shape, src_strides) = expand(&src.shape, &src.strides);
    let (_, dst_strides) = expand(&dst.shape, &dst.strides);
    let count = logical_len(&src.shape).ok_or_else(|| invalid("element count overflows"))?;
    Ok(PackParams {
        count: to_u32(count, "element count")?,
        rank: shape.len() as u32,
        src_offset: to_u32(src.offset, "source offset")?,
        dst_offset: to_u32(dst.offset, "destination offset")?,
        shape: narrow(&shape, "extent")?,
        src_strides: narrow(&src_strides, "source stride")?,
        dst_strides: narrow(&dst_strides, "destination stride")?,
    })
}

fn group_width(pso: &metal::ComputePipelineStateRef, n: u64) -> u64 {
    let limit = pso.max_total_threads_per_threadgroup().max(1);
    n.min(COPY_GROUP).min(limit).max(1)
}

fn encode_binary<P>(
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

/// Commits and waits for a command buffer that `encode` filled. An encode error commits nothing.
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

/// Returns the sub-view of `view` selected by one range per axis.
///
/// `ranges` has one entry per axis (an empty slice for rank 0). The result borrows the same
/// buffer: `host_ptr` is the buffer base and the strides are unchanged. For a non-empty
/// result the offset is exactly `view.offset + Σ start · stride`. Inverted ranges,
/// out-of-bounds ranges, and the wrong number of ranges return
/// [`crate::abi::LinalgError::InvalidArgument`]. This is O(rank) and does no GPU work.
///
/// Every in-bounds empty range is legal, and the result is a no-op input to [`pack`] and
/// [`concat`]. An empty view addresses nothing, so its offset is unobservable: the offset
/// is `view.offset + Σ start · stride` when that is within the buffer, and is clamped to the
/// buffer capacity (in elements) when it is not, or when the sum overflows `usize`. The
/// clamp keeps the result valid under [`TensorView::validate`], which accepts an empty view
/// only when its offset does not exceed the capacity.
pub fn slice<'a, T>(
    view: &TensorView<'a, T>,
    ranges: &[Range<usize>],
) -> Result<TensorView<'a, T>> {
    view.validate()?;
    if ranges.len() != view.rank() {
        return Err(invalid(format!(
            "slice expected {} ranges, got {}",
            view.rank(),
            ranges.len()
        )));
    }
    // `None` once the offset arithmetic overflows; that is an error only for a non-empty view.
    let mut offset = Some(view.offset);
    let mut shape = Vec::with_capacity(view.rank());
    for (axis, range) in ranges.iter().enumerate() {
        if range.start > range.end {
            return Err(invalid(format!("inverted range {range:?} on axis {axis}")));
        }
        if range.end > view.shape[axis] {
            return Err(invalid(format!(
                "range {range:?} exceeds extent {} on axis {axis}",
                view.shape[axis]
            )));
        }
        offset = offset.and_then(|acc| {
            range.start.checked_mul(view.strides[axis]).and_then(|step| acc.checked_add(step))
        });
        shape.push(range.end - range.start);
    }
    let capacity = view.buffer.length() as usize / size_of::<T>();
    let offset = if shape.contains(&0) {
        offset.map_or(capacity, |exact| exact.min(capacity))
    } else {
        offset.ok_or_else(|| invalid("slice offset overflow"))?
    };
    let out = TensorView::new(view.buffer, offset, shape, view.strides.clone());
    out.validate()?;
    Ok(out)
}

/// Copies `src` onto `dst` elementwise and waits until the GPU finishes.
///
/// Shapes must match. Strides may differ. See [`encode_pack`].
pub fn pack<T>(ctx: &MetalContext, src: &TensorView<'_, T>, dst: &TensorView<'_, T>) -> Result<()> {
    run_sync(ctx, |cmd| encode_pack(ctx, cmd, src, dst))
}

/// Encodes [`pack`] into `cmd` without committing, waiting, or reading host memory.
///
/// One compute encoder copies every logical element. Bits move as 2, 4, 8, or 16-byte
/// integers. The destination must not overlap the source and must be injective, so threads
/// do not race on a store. An empty view encodes nothing and does not compile a shader.
pub fn encode_pack<T>(
    ctx: &MetalContext,
    cmd: &metal::CommandBufferRef,
    src: &TensorView<'_, T>,
    dst: &TensorView<'_, T>,
) -> Result<()> {
    src.validate()?;
    dst.validate()?;
    let elem = elem_size::<T>()?;
    if src.shape != dst.shape {
        return Err(invalid(format!(
            "destination shape {:?} does not match source shape {:?}",
            dst.shape, src.shape
        )));
    }
    if src.rank() > MAX_RANK {
        return Err(invalid(format!("rank {} exceeds {MAX_RANK}", src.rank())));
    }
    if src.may_overlap(dst) {
        return Err(LinalgError::Overlap("destination overlaps the source".into()));
    }
    if src.is_empty() {
        return Ok(());
    }
    if destination_aliases(dst) {
        return Err(invalid(
            "destination layout aliases an element, so a parallel store would race",
        ));
    }
    fits(src, "source")?;
    fits(dst, "destination")?;

    if let Some(params) = vector_plan(src, dst, elem) {
        if params.n_rows > 0 && params.threads_per_row > 0 {
            let pso =
                compute_pipeline(ctx, MODULE, SOURCE, compile_options, kernel_name("vec", elem)?)?;
            let group = group_width(&pso, params.threads_per_row as u64);
            let grid = metal::MTLSize::new(params.threads_per_row as u64, params.n_rows as u64, 1);
            encode_binary(cmd, &pso, src.buffer, dst.buffer, &params, |enc| {
                enc.dispatch_threads(grid, metal::MTLSize::new(group, 1, 1));
            });
            return Ok(());
        }
    }

    let params = scalar_params(src, dst)?;
    if params.count == 0 {
        return Ok(());
    }
    let pso = compute_pipeline(ctx, MODULE, SOURCE, compile_options, kernel_name("pack", elem)?)?;
    let count = params.count as u64;
    let group = group_width(&pso, count);
    encode_binary(cmd, &pso, src.buffer, dst.buffer, &params, |enc| {
        enc.dispatch_threads(metal::MTLSize::new(count, 1, 1), metal::MTLSize::new(group, 1, 1));
    });
    Ok(())
}

/// Concatenates `parts` along `axis` into `dst` and waits until the GPU finishes.
///
/// Parts are laid out in order. Empty parts are legal. See [`encode_concat`].
pub fn concat<T>(
    ctx: &MetalContext,
    parts: &[&TensorView<'_, T>],
    axis: usize,
    dst: &TensorView<'_, T>,
) -> Result<()> {
    run_sync(ctx, |cmd| encode_concat(ctx, cmd, parts, axis, dst))
}

/// Encodes [`concat`] into `cmd` without committing, waiting, or reading host memory.
///
/// One dispatch covers every logical destination element. Each thread loads its source
/// part from a device table indexed by the coordinate along `axis`, then copies one element.
/// At most [`MAX_CONCAT_PARTS`] parts, including empty ones, which contribute nothing. The
/// destination must not overlap any part. An empty destination encodes nothing.
///
/// `ConcatParams` goes through `set_bytes`. The per-call part table lives in a device buffer
/// that is dropped when this function returns, so `cmd` must retain its resources: use a
/// command buffer from `new_command_buffer` (not `new_command_buffer_with_unretained_references`).
/// The caller's part and destination buffers must stay alive until the GPU finishes.
pub fn encode_concat<T>(
    ctx: &MetalContext,
    cmd: &metal::CommandBufferRef,
    parts: &[&TensorView<'_, T>],
    axis: usize,
    dst: &TensorView<'_, T>,
) -> Result<()> {
    if parts.is_empty() {
        return Err(invalid("concat needs at least one part"));
    }
    if parts.len() > MAX_CONCAT_PARTS {
        return Err(invalid(format!(
            "{} parts exceeds the limit of {MAX_CONCAT_PARTS}",
            parts.len()
        )));
    }
    dst.validate()?;
    let elem = elem_size::<T>()?;
    if dst.rank() == 0 || axis >= dst.rank() {
        return Err(invalid(format!("axis {axis} is out of range for rank {}", dst.rank())));
    }
    if dst.rank() > MAX_RANK {
        return Err(invalid(format!("rank {} exceeds {MAX_RANK}", dst.rank())));
    }

    let mut extents = [0usize; MAX_CONCAT_PARTS];
    let mut sum = 0usize;
    for (i, part) in parts.iter().enumerate() {
        part.validate()?;
        if part.rank() != dst.rank() {
            return Err(invalid(format!(
                "part {i} rank {} does not match destination rank {}",
                part.rank(),
                dst.rank()
            )));
        }
        for ax in 0..dst.rank() {
            if ax != axis && part.shape[ax] != dst.shape[ax] {
                return Err(invalid(format!(
                    "part {i} extent {} on axis {ax} does not match destination extent {}",
                    part.shape[ax], dst.shape[ax]
                )));
            }
        }
        if part.may_overlap(dst) {
            return Err(LinalgError::Overlap(format!("destination overlaps part {i}")));
        }
        extents[i] = part.shape[axis];
        sum = sum.checked_add(extents[i]).ok_or_else(|| invalid("concat extent overflows"))?;
    }
    if sum != dst.shape[axis] {
        return Err(invalid(format!(
            "parts cover {sum} along axis {axis}, destination extent is {}",
            dst.shape[axis]
        )));
    }
    if dst.is_empty() {
        return Ok(());
    }
    if destination_aliases(dst) {
        return Err(invalid(
            "destination layout aliases an element, so a parallel store would race",
        ));
    }
    fits(dst, "destination")?;
    for (i, part) in parts.iter().enumerate() {
        if !part.is_empty() {
            fits(part, &format!("part {i}"))?;
        }
    }

    let mut prefix = [0u32; MAX_CONCAT_PARTS + 1];
    let mut running = 0u32;
    for i in 0..parts.len() {
        running = running
            .checked_add(to_u32(extents[i], "part extent")?)
            .ok_or_else(|| invalid("concat prefix overflows 32-bit indexing"))?;
        prefix[i + 1] = running;
    }
    let mut params = ConcatParams {
        count: to_u32(
            logical_len(&dst.shape).ok_or_else(|| invalid("element count overflows"))?,
            "element count",
        )?,
        rank: dst.rank() as u32,
        axis: axis as u32,
        n_parts: parts.len() as u32,
        dst_offset: to_u32(dst.offset, "destination offset")?,
        dst_shape: narrow(&dst.shape, "destination extent")?,
        dst_strides: narrow(&dst.strides, "destination stride")?,
        part_offset: [0; MAX_CONCAT_PARTS],
        part_strides: [[0; MAX_RANK]; MAX_CONCAT_PARTS],
    };
    for (i, part) in parts.iter().enumerate().filter(|(_, part)| !part.is_empty()) {
        params.part_offset[i] = to_u32(part.offset, "part offset")?;
        params.part_strides[i] = narrow(&part.strides, "part stride")?;
    }
    if params.count == 0 {
        return Ok(());
    }

    // `(part, prefix)` per coordinate along `axis`. Its length is 2 · extent words, which can
    // exceed the 4 KB `set_bytes` limit, so the table is a device buffer.
    let extent = dst.shape[axis];
    let table_len = extent.checked_mul(2).ok_or_else(|| invalid("concat part table overflows"))?;
    let mut table = Vec::<u32>::new();
    if table.try_reserve_exact(table_len).is_err() {
        return Err(invalid("concat part table is too large"));
    }
    table.resize(table_len, 0);
    for i in 0..parts.len() {
        let base = prefix[i];
        for along in base..prefix[i + 1] {
            let slot = along as usize * 2;
            table[slot] = i as u32;
            table[slot + 1] = base;
        }
    }
    let table_bytes = table_len
        .checked_mul(size_of::<u32>())
        .ok_or_else(|| invalid("concat part table overflows"))?;
    let table_buf = ctx.device.new_buffer_with_data(
        table.as_ptr() as *const c_void,
        table_bytes as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let pso = compute_pipeline(ctx, MODULE, SOURCE, compile_options, kernel_name("concat", elem)?)?;
    let count = params.count as u64;
    let group = group_width(&pso, count);
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pso);
    for slot in 0..MAX_CONCAT_PARTS {
        let buffer = parts.get(slot).map(|part| part.buffer).unwrap_or(dst.buffer);
        enc.set_buffer(slot as u64, Some(buffer), 0);
    }
    enc.set_buffer(MAX_CONCAT_PARTS as u64, Some(dst.buffer), 0);
    enc.set_bytes(
        (MAX_CONCAT_PARTS + 1) as u64,
        size_of::<ConcatParams>() as u64,
        &params as *const ConcatParams as *const c_void,
    );
    enc.set_buffer((MAX_CONCAT_PARTS + 2) as u64, Some(&table_buf), 0);
    enc.dispatch_threads(metal::MTLSize::new(count, 1, 1), metal::MTLSize::new(group, 1, 1));
    enc.end_encoding();
    Ok(())
}

#[cfg(test)]
mod layout_tests {
    use super::row_layout;

    fn check(mis: usize, inner: usize, elem: usize, head: u32, n32: u32, n16: u32, tail: u32) {
        let split = row_layout(mis, inner, elem)
            .unwrap_or_else(|| panic!("row_layout({mis}, {inner}, {elem}) should split"));
        assert_eq!((split.head, split.n32, split.n16, split.tail), (head, n32, n16, tail));
        let per32 = 32 / elem;
        let per16 = 16 / elem;
        let covered = split.head as usize
            + split.n32 as usize * per32
            + split.n16 as usize * per16
            + split.tail as usize;
        assert_eq!(covered, inner);
        assert_eq!(split.threads_per_row, split.head + split.n32 + split.n16 + split.tail);
        assert!(split.n16 <= 1);
        if (split.head as usize) < inner {
            assert_eq!((mis + split.head as usize * elem) % 16, 0);
        }
    }

    #[test]
    fn head_vector_tail_matches_the_worked_examples() {
        // f32, element offset ≡ 1 (mod 4): 3 scalar + ulong4(8) + ulong2(4) + 1 scalar.
        check(4, 16, 4, 3, 1, 1, 1);
        check(0, 16, 4, 0, 2, 0, 0);
        check(8, 16, 4, 2, 1, 1, 2);
        check(12, 16, 4, 1, 1, 1, 3);
        check(4, 33, 4, 3, 3, 1, 2);
        // f64 offset 1, inner 8: head 1 + ulong4(4) + ulong2(2) + tail 1.
        check(8, 8, 8, 1, 1, 1, 1);
        // f16 offset 1, inner 24: 7 scalar + ulong4(16) + 1 scalar.
        check(2, 24, 2, 7, 1, 0, 1);
        // 16-byte elements: two ulong4 (2 elements each) and one ulong2, no tail.
        check(0, 5, 16, 0, 2, 1, 0);
        // The head consumes a row that ends before the next 16-byte boundary.
        check(4, 2, 4, 2, 0, 0, 0);
        check(8, 1, 8, 1, 0, 0, 0);
        assert!(row_layout(1, 16, 4).is_none());
        assert!(row_layout(0, 0, 4).is_none());
        assert!(row_layout(0, 4, 3).is_none());
    }
}
