//! Tensor contraction: `C = αAB + βC`, batched GEMM, and contraction over shared axes.
//!
//! Labels follow einsum rules. Axes are grouped into batch / M / N / K and adjacent axes
//! are merged when every operand that carries them is contiguous, so the kernel sees at
//! most four strides per group. `β = 0` does not read C, `α = 0` does not read A or B,
//! `K = 0` yields `C = βC`, and an empty M, N, or batch launches nothing.

// `msg_send!` expands a `cfg` the objc crate owns. This crate does not list that cfg.
#![allow(unexpected_cfgs)]

use std::any::TypeId;
use std::cell::RefCell;
use std::collections::HashSet;
use std::ffi::c_void;

use metal::foreign_types::ForeignTypeRef;
use metal::objc::runtime::Object;
use metal::objc::{msg_send, sel, sel_impl};

use crate::abi::{compute_pipeline, LinalgError, MetalContext, Result, TensorView};

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for half::f16 {}
}

/// Element types the contraction kernels support: `f32` and `half::f16`.
pub trait Element: sealed::Sealed + Copy + 'static {}

impl Element for f32 {}
impl Element for half::f16 {}

/// A view paired with one label per axis, einsum style (`"ijk"` labels a rank-3 view).
pub type Operand<'v, 'b, T> = (&'v TensorView<'b, T>, &'v str);

const SOURCE: &str = include_str!("../../shaders/contraction.metal");

/// Tile geometry of `contract_tiled_f32`. The shader constants must stay in lockstep.
const TILE_M: u32 = 64;
const TILE_N: u32 = 64;
const TILE_K: u32 = 32;
const TILE_THREADS: u32 = 512;
const GENERAL_TILE: u32 = 8;

/// Right-aligned mixed radix. Slot 3 is the fastest axis; unused slots have extent 1.
#[repr(C)]
#[derive(Clone, Copy)]
struct Axes4 {
    ext: [u32; 4],
    stride: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct TiledParams {
    a_offset: u32,
    b_offset: u32,
    c_offset: u32,
    m: u32,
    n: u32,
    k: u32,
    batch: u32,
    a_batch_stride: u32,
    a_m_stride: u32,
    a_k_stride: u32,
    b_batch_stride: u32,
    b_n_stride: u32,
    b_k_stride: u32,
    c_batch_stride: u32,
    c_m_stride: u32,
    c_n_stride: u32,
    alpha: f32,
    beta: f32,
    k_chunk: u32,
    splits: u32,
    vector_ok: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GeneralParams {
    a_offset: u32,
    b_offset: u32,
    c_offset: u32,
    m: u32,
    n: u32,
    k: u32,
    batch: u32,
    alpha: f32,
    beta: f32,
    a_batch: Axes4,
    a_m: Axes4,
    a_k: Axes4,
    b_batch: Axes4,
    b_n: Axes4,
    b_k: Axes4,
    c_batch: Axes4,
    c_m: Axes4,
    c_n: Axes4,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ScaleParams {
    c_offset: u32,
    count: u32,
    m: u32,
    n: u32,
    batch: u32,
    beta: f32,
    c_batch: Axes4,
    c_m: Axes4,
    c_n: Axes4,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ReduceParams {
    c_offset: u32,
    count: u32,
    m: u32,
    n: u32,
    batch: u32,
    splits: u32,
    alpha: f32,
    beta: f32,
    c_batch: Axes4,
    c_m: Axes4,
    c_n: Axes4,
}

const _: () = {
    assert!(std::mem::size_of::<TiledParams>() == 84);
    assert!(std::mem::size_of::<Axes4>() == 32);
    assert!(std::mem::size_of::<GeneralParams>() == 36 + 9 * 32);
    assert!(std::mem::size_of::<ScaleParams>() == 24 + 3 * 32);
    assert!(std::mem::size_of::<ReduceParams>() == 32 + 3 * 32);
};

struct Kernels {
    general: &'static str,
    scale: &'static str,
    zero: &'static str,
}

/// f16 products all go through `contract_general_f16`. The tiled, partial, and
/// reduce entry points exist only for f32, so they are named at those plans.
fn kernels<T: Element>() -> Kernels {
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        Kernels {
            general: "contract_general_f32",
            scale: "contract_scale_f32",
            zero: "contract_zero_f32",
        }
    } else {
        Kernels {
            general: "contract_general_f16",
            scale: "contract_scale_f16",
            zero: "contract_zero_f16",
        }
    }
}

fn invalid(msg: impl Into<String>) -> LinalgError {
    LinalgError::InvalidArgument(msg.into())
}

fn overlap(msg: impl Into<String>) -> LinalgError {
    LinalgError::Overlap(msg.into())
}

fn fit(x: usize, what: &str) -> Result<u32> {
    u32::try_from(x).map_err(|_| invalid(format!("{what} {x} does not fit in u32")))
}

fn mul_u32(a: u32, b: u32) -> Result<u32> {
    u32::try_from(u64::from(a) * u64::from(b)).map_err(|_| invalid("dimension product exceeds u32"))
}

fn shader_options() -> metal::CompileOptions {
    let opts = metal::CompileOptions::new();
    opts.set_fast_math_enabled(false);
    opts.set_language_version(metal::MTLLanguageVersion::V2_4);
    opts
}

fn pipeline(ctx: &MetalContext, function: &str) -> Result<metal::ComputePipelineState> {
    compute_pipeline(ctx, "contraction", SOURCE, shader_options, function)
}

fn set_params<T>(enc: &metal::ComputeCommandEncoderRef, index: u64, value: &T) {
    enc.set_bytes(index, std::mem::size_of::<T>() as u64, value as *const T as *const c_void);
}

/// Scratch kept alive until the command buffer that uses it has completed.
///
/// `encode_*` returns before the caller commits, so the partial buffer cannot be dropped
/// with the stack frame. Metal does not promise to retain a buffer that was only encoded,
/// and the completion-handler block type is not reachable without a new dependency. The
/// buffer stays in this thread-local list, which drops it once `status` is completed or error.
struct Live {
    command: usize,
    /// Kept so the partial buffer outlives the command buffer that encodes it.
    #[allow(dead_code)]
    buffer: metal::Buffer,
}

impl Drop for Live {
    fn drop(&mut self) {
        // SAFETY: `command` is an MTLCommandBuffer retained in `track`. This is the matching
        // release, and it runs only after the buffer is no longer in the list.
        unsafe {
            let () = msg_send![self.command as *mut Object, release];
        }
    }
}

thread_local! {
    static LIVE: RefCell<Vec<Live>> = const { RefCell::new(Vec::new()) };
}

fn track(cmd: &metal::CommandBufferRef, buffer: metal::Buffer) {
    LIVE.with(|live| {
        let mut live = live.borrow_mut();
        reap(&mut live);
        // SAFETY: `cmd` is a live command buffer. `retain` is balanced by `Live`'s drop.
        let command = unsafe {
            let ptr = cmd.as_ptr() as *mut Object;
            let () = msg_send![ptr, retain];
            ptr as usize
        };
        live.push(Live { command, buffer });
    });
}

fn reap(live: &mut Vec<Live>) {
    let mut i = 0;
    while i < live.len() {
        // SAFETY: each entry's pointer was retained and has not been released yet.
        // `status` returns NSUInteger: 4 is completed, 5 is error.
        let status: u64 = unsafe { msg_send![live[i].command as *mut Object, status] };
        if status >= 4 {
            live.swap_remove(i);
        } else {
            i += 1;
        }
    }
}

fn alloc_partial(ctx: &MetalContext, elems: u32) -> metal::Buffer {
    let bytes = (elems as usize).saturating_mul(std::mem::size_of::<f32>()).max(16);
    ctx.device.new_buffer(bytes as u64, metal::MTLResourceOptions::StorageModeShared)
}

#[derive(Clone, Copy)]
struct Axis {
    extent: usize,
    a: Option<usize>,
    b: Option<usize>,
    c: Option<usize>,
}

fn can_merge(prev: Axis, next: Axis) -> bool {
    for (prev_stride, next_stride) in [(prev.a, next.a), (prev.b, next.b), (prev.c, next.c)] {
        match (prev_stride, next_stride) {
            (Some(prev_stride), Some(next_stride)) => match next.extent.checked_mul(next_stride) {
                Some(prod) if prev_stride == prod => {}
                _ => return false,
            },
            (None, None) => {}
            _ => return false,
        }
    }
    true
}

/// Drops extent-1 axes and folds a slower axis into the next when both operands are contiguous.
fn merge_group(axes: &[Axis], group: &str) -> Result<Vec<Axis>> {
    if axes.iter().any(|ax| ax.extent == 0) {
        return Ok(vec![Axis { extent: 0, a: Some(0), b: Some(0), c: Some(0) }]);
    }
    let kept: Vec<Axis> = axes.iter().copied().filter(|ax| ax.extent != 1).collect();
    if kept.is_empty() {
        return Ok(vec![Axis { extent: 1, a: Some(0), b: Some(0), c: Some(0) }]);
    }
    let mut out = vec![kept[0]];
    for ax in kept.into_iter().skip(1) {
        let prev = out[out.len() - 1];
        if can_merge(prev, ax) {
            let extent = prev
                .extent
                .checked_mul(ax.extent)
                .ok_or_else(|| invalid(format!("{group} extent product overflows")))?;
            let last_i = out.len() - 1;
            let last = &mut out[last_i];
            last.extent = extent;
            last.a = ax.a;
            last.b = ax.b;
            last.c = ax.c;
        } else {
            out.push(ax);
        }
    }
    if out.len() > 4 {
        return Err(invalid(format!(
            "{group} group has {} strides after merging; the kernel supports 4",
            out.len()
        )));
    }
    Ok(out)
}

fn pack(axes: &[Axis], pick: impl Fn(Axis) -> usize) -> Result<Axes4> {
    let mut out = Axes4 { ext: [1; 4], stride: [0; 4] };
    let n = axes.len();
    for (i, ax) in axes.iter().copied().enumerate() {
        let slot = 4 - n + i;
        out.ext[slot] = fit(ax.extent, "extent")?;
        out.stride[slot] = fit(pick(ax), "stride")?;
    }
    Ok(out)
}

fn stride(slot: Option<usize>) -> usize {
    slot.unwrap_or(0)
}

fn product(axes: &[Axis]) -> Result<u32> {
    let mut acc: u32 = 1;
    for ax in axes {
        acc = mul_u32(acc, fit(ax.extent, "extent")?)?;
    }
    Ok(acc)
}

struct Groups {
    batch: Vec<Axis>,
    m: Vec<Axis>,
    n: Vec<Axis>,
    k: Vec<Axis>,
}

fn parse_labels(labels: &str, rank: usize) -> Result<Vec<char>> {
    let parsed: Vec<char> = labels.chars().collect();
    if parsed.len() != rank {
        return Err(invalid(format!("label count {} does not match rank {rank}", parsed.len())));
    }
    for (i, ch) in parsed.iter().enumerate() {
        if parsed[..i].contains(ch) {
            return Err(invalid(format!("repeated label '{ch}'")));
        }
    }
    Ok(parsed)
}

struct Labeled<'a> {
    shape: &'a [usize],
    strides: &'a [usize],
    labels: Vec<char>,
}

fn lookup(op: &Labeled<'_>, label: char) -> Option<(usize, usize)> {
    op.labels.iter().position(|ch| *ch == label).map(|i| (op.shape[i], op.strides[i]))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    Batch,
    M,
    N,
    K,
}

struct LabelInfo {
    role: Role,
    axis: Axis,
}

fn groups_from_labels(a: &Labeled<'_>, b: &Labeled<'_>, c: &Labeled<'_>) -> Result<Groups> {
    let mut order = Vec::new();
    for ch in a.labels.iter().chain(b.labels.iter()).chain(c.labels.iter()) {
        if !order.contains(ch) {
            order.push(*ch);
        }
    }
    let mut infos: Vec<(char, LabelInfo)> = Vec::with_capacity(order.len());
    for ch in order {
        let fa = lookup(a, ch);
        let fb = lookup(b, ch);
        let fc = lookup(c, ch);
        let role = match (fa.is_some(), fb.is_some(), fc.is_some()) {
            (true, true, true) => Role::Batch,
            (true, false, true) => Role::M,
            (false, true, true) => Role::N,
            (true, true, false) => Role::K,
            (false, false, true) => {
                return Err(invalid(format!("label '{ch}' appears only in C")));
            }
            _ => return Err(invalid(format!("label '{ch}' appears once"))),
        };
        let mut extent = None;
        for found in [fa, fb, fc].into_iter().flatten() {
            match extent {
                None => extent = Some(found.0),
                Some(prev) if prev != found.0 => {
                    return Err(invalid(format!("extent mismatch for label '{ch}'")));
                }
                Some(_) => {}
            }
        }
        infos.push((
            ch,
            LabelInfo {
                role,
                axis: Axis {
                    extent: extent.unwrap_or(1),
                    a: fa.map(|(_, s)| s),
                    b: fb.map(|(_, s)| s),
                    c: fc.map(|(_, s)| s),
                },
            },
        ));
    }
    let take = |labels: &[char], role: Role| -> Vec<Axis> {
        labels
            .iter()
            .filter_map(|ch| {
                infos
                    .iter()
                    .find(|(label, _)| label == ch)
                    .and_then(|(_, info)| (info.role == role).then_some(info.axis))
            })
            .collect()
    };
    Ok(Groups {
        batch: take(&c.labels, Role::Batch),
        m: take(&c.labels, Role::M),
        n: take(&c.labels, Role::N),
        k: take(&a.labels, Role::K),
    })
}

fn groups_from_gemm<T>(
    a: &TensorView<'_, T>,
    b: &TensorView<'_, T>,
    c: &TensorView<'_, T>,
) -> Result<Groups> {
    if a.rank() < 2 || b.rank() < 2 || c.rank() < 2 {
        return Err(invalid("rank below 2"));
    }
    if a.rank() != b.rank() || b.rank() != c.rank() {
        return Err(invalid("ranks differ"));
    }
    let rank = a.rank();
    let mut batch = Vec::with_capacity(rank - 2);
    for axis in 0..rank - 2 {
        if a.shape[axis] != b.shape[axis] || b.shape[axis] != c.shape[axis] {
            return Err(invalid("batch extents differ"));
        }
        batch.push(Axis {
            extent: a.shape[axis],
            a: Some(a.strides[axis]),
            b: Some(b.strides[axis]),
            c: Some(c.strides[axis]),
        });
    }
    let (m, k_a) = (a.shape[rank - 2], a.shape[rank - 1]);
    let (k_b, n) = (b.shape[rank - 2], b.shape[rank - 1]);
    if m != c.shape[rank - 2] || n != c.shape[rank - 1] || k_a != k_b {
        return Err(invalid("inner extents differ"));
    }
    Ok(Groups {
        batch,
        m: vec![Axis {
            extent: m,
            a: Some(a.strides[rank - 2]),
            b: None,
            c: Some(c.strides[rank - 2]),
        }],
        n: vec![Axis {
            extent: n,
            a: None,
            b: Some(b.strides[rank - 1]),
            c: Some(c.strides[rank - 1]),
        }],
        k: vec![Axis {
            extent: k_a,
            a: Some(a.strides[rank - 1]),
            b: Some(b.strides[rank - 2]),
            c: None,
        }],
    })
}

fn max_index<T>(view: &TensorView<'_, T>) -> Option<usize> {
    view.shape.iter().zip(&view.strides).try_fold(view.offset, |acc, (&extent, &stride)| {
        extent.checked_sub(1)?.checked_mul(stride)?.checked_add(acc)
    })
}

fn check_index_width<T>(view: &TensorView<'_, T>, name: &str) -> Result<()> {
    fit(view.offset, &format!("{name} offset"))?;
    if let Some(last) = max_index(view) {
        fit(last, &format!("{name} index"))?;
    }
    Ok(())
}

fn same_buffer<T, U>(a: &TensorView<'_, T>, b: &TensorView<'_, U>) -> bool {
    a.buffer.contents() == b.buffer.contents()
}

fn same_view<T>(a: &TensorView<'_, T>, b: &TensorView<'_, T>) -> bool {
    same_buffer(a, b) && a.offset == b.offset && a.shape == b.shape && a.strides == b.strides
}

/// Extent>1 axes sorted by ascending stride, when they form a unique mixed-radix address.
fn decodable<T>(view: &TensorView<'_, T>) -> Option<Vec<(usize, usize)>> {
    let mut axes: Vec<(usize, usize)> = view
        .shape
        .iter()
        .copied()
        .zip(view.strides.iter().copied())
        .filter(|(extent, _)| *extent > 1)
        .collect();
    axes.sort_by_key(|(_, stride)| *stride);
    if axes.iter().any(|(_, stride)| *stride == 0) {
        return None;
    }
    for i in 1..axes.len() {
        let (extent, stride) = axes[i - 1];
        let next = axes[i].1;
        if stride.checked_mul(extent).is_none_or(|span| next < span) {
            return None;
        }
    }
    Some(axes)
}

fn contains<T>(view: &TensorView<'_, T>, elem: usize, axes_asc: &[(usize, usize)]) -> bool {
    if elem < view.offset {
        return false;
    }
    let mut rem = elem - view.offset;
    for &(extent, stride) in axes_asc.iter().rev() {
        let coord = rem / stride;
        if coord >= extent {
            return false;
        }
        rem -= coord * stride;
    }
    rem == 0
}

fn for_each_index<T>(view: &TensorView<'_, T>, mut visit: impl FnMut(usize)) {
    if view.shape.contains(&0) {
        return;
    }
    let rank = view.rank();
    if rank == 0 {
        visit(view.offset);
        return;
    }
    let mut idx = vec![0usize; rank];
    loop {
        let elem = view.offset + idx.iter().zip(&view.strides).map(|(i, s)| i * s).sum::<usize>();
        visit(elem);
        let mut axis = rank - 1;
        loop {
            idx[axis] += 1;
            if idx[axis] < view.shape[axis] {
                break;
            }
            idx[axis] = 0;
            if axis == 0 {
                return;
            }
            axis -= 1;
        }
    }
}

const EXACT_LIMIT: usize = 2_000_000;

fn shares<T>(a: &TensorView<'_, T>, b: &TensorView<'_, T>) -> bool {
    if !same_buffer(a, b) {
        return false;
    }
    let (Some(ra), Some(rb)) = (a.element_range(), b.element_range()) else {
        return false;
    };
    if ra.start >= rb.end || rb.start >= ra.end {
        return false;
    }
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    if let Some(axes) = decodable(large) {
        let mut hit = false;
        for_each_index(small, |elem| {
            hit |= contains(large, elem, &axes);
        });
        return hit;
    }
    // Unusual strides (or a stride of 0). Exact when the views are modest; otherwise
    // report overlap rather than launch a kernel that would race.
    if large.len() > EXACT_LIMIT {
        return true;
    }
    let mut seen = HashSet::with_capacity(large.len());
    for_each_index(large, |elem| {
        seen.insert(elem);
    });
    let mut hit = false;
    for_each_index(small, |elem| {
        hit |= seen.contains(&elem);
    });
    hit
}

fn self_overlaps<T>(view: &TensorView<'_, T>) -> bool {
    if view.len() <= 1 || decodable(view).is_some() {
        return false;
    }
    if view.len() > EXACT_LIMIT {
        return true;
    }
    let mut seen = HashSet::with_capacity(view.len());
    let mut overlap = false;
    for_each_index(view, |elem| {
        if !seen.insert(elem) {
            overlap = true;
        }
    });
    overlap
}

fn check_destination_alias<T>(
    a: &TensorView<'_, T>,
    b: &TensorView<'_, T>,
    c: &TensorView<'_, T>,
    dims: (u32, u32, u32),
    reads_inputs: bool,
) -> Result<()> {
    if self_overlaps(c) {
        return Err(overlap("C writes the same element more than once"));
    }
    if !reads_inputs {
        return Ok(());
    }
    let (m, n, k) = dims;
    // In-place scale: C is A (or B) and every other extent is 1, so each output reads
    // only its own previous value. A and B may still alias each other.
    let inplace_a = k == 1 && n == 1 && same_view(c, a) && !shares(c, b);
    let inplace_b = k == 1 && m == 1 && same_view(c, b) && !shares(c, a);
    if !inplace_a && (same_view(c, a) || shares(c, a)) {
        return Err(overlap("C overlaps A"));
    }
    if !inplace_b && (same_view(c, b) || shares(c, b)) {
        return Err(overlap("C overlaps B"));
    }
    Ok(())
}

enum Plan {
    Empty,
    Fill(ScaleParams),
    Tiled(TiledParams),
    /// Split-K: the tiled kernel writes raw partials, then `reduce` applies α and β once.
    Split {
        tiled: TiledParams,
        reduce: ReduceParams,
        partial_elems: u32,
    },
    General(GeneralParams),
}

struct Merged {
    batch: Vec<Axis>,
    m: Vec<Axis>,
    n: Vec<Axis>,
    k: Vec<Axis>,
    batch_n: u32,
    m_n: u32,
    n_n: u32,
    k_n: u32,
}

fn merge_all(groups: Groups) -> Result<Merged> {
    let batch = merge_group(&groups.batch, "batch")?;
    let m = merge_group(&groups.m, "M")?;
    let n = merge_group(&groups.n, "N")?;
    let k = merge_group(&groups.k, "K")?;
    let batch_n = product(&batch)?;
    let m_n = product(&m)?;
    let n_n = product(&n)?;
    let k_n = product(&k)?;
    Ok(Merged { batch, m, n, k, batch_n, m_n, n_n, k_n })
}

fn vector_ok(a_off: u32, b_off: u32, p: &TiledParams) -> u32 {
    let aligned = |x: u32| x % 4 == 0;
    let ok = p.a_k_stride == 1
        && p.b_n_stride == 1
        && aligned(a_off)
        && aligned(b_off)
        && aligned(p.a_m_stride)
        && aligned(p.a_batch_stride)
        && aligned(p.b_k_stride)
        && aligned(p.b_batch_stride);
    u32::from(ok)
}

fn tiled_params<T>(
    a: &TensorView<'_, T>,
    b: &TensorView<'_, T>,
    c: &TensorView<'_, T>,
    merged: &Merged,
    alpha: f32,
    beta: f32,
) -> Result<TiledParams> {
    let (batch, m, n, k) = (merged.batch[0], merged.m[0], merged.n[0], merged.k[0]);
    let mut params = TiledParams {
        a_offset: fit(a.offset, "A offset")?,
        b_offset: fit(b.offset, "B offset")?,
        c_offset: fit(c.offset, "C offset")?,
        m: merged.m_n,
        n: merged.n_n,
        k: merged.k_n,
        batch: merged.batch_n,
        a_batch_stride: fit(stride(batch.a), "A batch stride")?,
        a_m_stride: fit(stride(m.a), "A M stride")?,
        a_k_stride: fit(stride(k.a), "A K stride")?,
        b_batch_stride: fit(stride(batch.b), "B batch stride")?,
        b_n_stride: fit(stride(n.b), "B N stride")?,
        b_k_stride: fit(stride(k.b), "B K stride")?,
        c_batch_stride: fit(stride(batch.c), "C batch stride")?,
        c_m_stride: fit(stride(m.c), "C M stride")?,
        c_n_stride: fit(stride(n.c), "C N stride")?,
        alpha,
        beta,
        k_chunk: merged.k_n,
        splits: 1,
        vector_ok: 0,
    };
    params.vector_ok = vector_ok(params.a_offset, params.b_offset, &params);
    Ok(params)
}

fn general_params<T>(
    a: &TensorView<'_, T>,
    b: &TensorView<'_, T>,
    c: &TensorView<'_, T>,
    merged: &Merged,
    alpha: f32,
    beta: f32,
) -> Result<GeneralParams> {
    Ok(GeneralParams {
        a_offset: fit(a.offset, "A offset")?,
        b_offset: fit(b.offset, "B offset")?,
        c_offset: fit(c.offset, "C offset")?,
        m: merged.m_n,
        n: merged.n_n,
        k: merged.k_n,
        batch: merged.batch_n,
        alpha,
        beta,
        a_batch: pack(&merged.batch, |ax| stride(ax.a))?,
        a_m: pack(&merged.m, |ax| stride(ax.a))?,
        a_k: pack(&merged.k, |ax| stride(ax.a))?,
        b_batch: pack(&merged.batch, |ax| stride(ax.b))?,
        b_n: pack(&merged.n, |ax| stride(ax.b))?,
        b_k: pack(&merged.k, |ax| stride(ax.b))?,
        c_batch: pack(&merged.batch, |ax| stride(ax.c))?,
        c_m: pack(&merged.m, |ax| stride(ax.c))?,
        c_n: pack(&merged.n, |ax| stride(ax.c))?,
    })
}

fn scale_params<T>(c: &TensorView<'_, T>, merged: &Merged, beta: f32) -> Result<ScaleParams> {
    let count = mul_u32(mul_u32(merged.batch_n, merged.m_n)?, merged.n_n)?;
    Ok(ScaleParams {
        c_offset: fit(c.offset, "C offset")?,
        count,
        m: merged.m_n,
        n: merged.n_n,
        batch: merged.batch_n,
        beta,
        c_batch: pack(&merged.batch, |ax| stride(ax.c))?,
        c_m: pack(&merged.m, |ax| stride(ax.c))?,
        c_n: pack(&merged.n, |ax| stride(ax.c))?,
    })
}

/// `(splits, k_chunk)`. One split when the output already fills the GPU or K is short.
fn split_shape(m: u32, n: u32, k: u32) -> (u32, u32) {
    let tiles = m.div_ceil(TILE_M).saturating_mul(n.div_ceil(TILE_N));
    const TARGET: u32 = 80;
    if tiles >= TARGET || k < TILE_K * 4 {
        return (1, k);
    }
    let want = TARGET.div_ceil(tiles.max(1)).clamp(1, 128);
    let max_by_k = (k / (TILE_K * 4)).max(1);
    let rough = want.min(max_by_k).max(1);
    let chunk = k.div_ceil(rough).div_ceil(TILE_K).saturating_mul(TILE_K);
    if chunk == 0 || chunk >= k {
        return (1, k);
    }
    (k.div_ceil(chunk), chunk)
}

fn plan_merged<T: Element>(
    a: &TensorView<'_, T>,
    b: &TensorView<'_, T>,
    c: &TensorView<'_, T>,
    merged: Merged,
    alpha: f32,
    beta: f32,
) -> Result<Plan> {
    if merged.m_n == 0 || merged.n_n == 0 || merged.batch_n == 0 {
        return Ok(Plan::Empty);
    }
    let reads_inputs = alpha != 0.0 && merged.k_n != 0;
    check_destination_alias(a, b, c, (merged.m_n, merged.n_n, merged.k_n), reads_inputs)?;
    if !reads_inputs {
        return Ok(Plan::Fill(scale_params(c, &merged, beta)?));
    }
    let single = merged.batch.len() == 1
        && merged.m.len() == 1
        && merged.n.len() == 1
        && merged.k.len() == 1;
    // f16 rounds once from a double-float dot. The tiled f32 accumulator is a
    // couple of f32 ulps off, enough to cross an f16 midpoint.
    let half_out = TypeId::of::<T>() == TypeId::of::<half::f16>();
    if !single || half_out {
        return Ok(Plan::General(general_params(a, b, c, &merged, alpha, beta)?));
    }
    let mut tiled = tiled_params(a, b, c, &merged, alpha, beta)?;
    let (splits, chunk) = split_shape(merged.m_n, merged.n_n, merged.k_n);
    if splits == 1 {
        return Ok(Plan::Tiled(tiled));
    }
    let per_plane = mul_u32(mul_u32(merged.batch_n, merged.m_n)?, merged.n_n)?;
    let partial_elems = match mul_u32(per_plane, splits) {
        Ok(elems) => elems,
        Err(_) => return Ok(Plan::Tiled(tiled)),
    };
    if mul_u32(merged.batch_n, splits).is_err() {
        return Ok(Plan::Tiled(tiled));
    }
    tiled.k_chunk = chunk;
    tiled.splits = splits;
    // Partials are addressed densely, so the vector-load alignment rule is unchanged.
    let reduce = ReduceParams {
        c_offset: tiled.c_offset,
        count: per_plane,
        m: merged.m_n,
        n: merged.n_n,
        batch: merged.batch_n,
        splits,
        alpha,
        beta,
        c_batch: pack(&merged.batch, |ax| stride(ax.c))?,
        c_m: pack(&merged.m, |ax| stride(ax.c))?,
        c_n: pack(&merged.n, |ax| stride(ax.c))?,
    };
    Ok(Plan::Split { tiled, reduce, partial_elems })
}

fn prepare_views<T: Element>(
    a: &TensorView<'_, T>,
    b: &TensorView<'_, T>,
    c: &TensorView<'_, T>,
) -> Result<()> {
    a.validate()?;
    b.validate()?;
    c.validate()?;
    check_index_width(a, "A")?;
    check_index_width(b, "B")?;
    check_index_width(c, "C")?;
    Ok(())
}

fn plan_contract<T: Element>(
    alpha: f32,
    a: Operand<'_, '_, T>,
    b: Operand<'_, '_, T>,
    beta: f32,
    c: Operand<'_, '_, T>,
) -> Result<Plan> {
    prepare_views(a.0, b.0, c.0)?;
    let a_labels = parse_labels(a.1, a.0.rank())?;
    let b_labels = parse_labels(b.1, b.0.rank())?;
    let c_labels = parse_labels(c.1, c.0.rank())?;
    let groups = groups_from_labels(
        &Labeled { shape: &a.0.shape, strides: &a.0.strides, labels: a_labels },
        &Labeled { shape: &b.0.shape, strides: &b.0.strides, labels: b_labels },
        &Labeled { shape: &c.0.shape, strides: &c.0.strides, labels: c_labels },
    )?;
    plan_merged(a.0, b.0, c.0, merge_all(groups)?, alpha, beta)
}

fn plan_gemm<T: Element>(
    alpha: f32,
    a: &TensorView<'_, T>,
    b: &TensorView<'_, T>,
    beta: f32,
    c: &TensorView<'_, T>,
) -> Result<Plan> {
    prepare_views(a, b, c)?;
    let groups = groups_from_gemm(a, b, c)?;
    plan_merged(a, b, c, merge_all(groups)?, alpha, beta)
}

fn dispatch_threads(enc: &metal::ComputeCommandEncoderRef, count: u32) {
    let width = u64::from(count);
    let group = u64::from(256.min(count).max(1));
    enc.dispatch_threads(metal::MTLSize::new(width, 1, 1), metal::MTLSize::new(group, 1, 1));
}

fn encode_plan<T: Element>(
    ctx: &MetalContext,
    cmd: &metal::CommandBufferRef,
    plan: Plan,
    a: &TensorView<'_, T>,
    b: &TensorView<'_, T>,
    c: &TensorView<'_, T>,
) -> Result<()> {
    let kernels = kernels::<T>();
    match plan {
        Plan::Empty => Ok(()),
        Plan::Fill(params) => {
            let name = if params.beta == 0.0 { kernels.zero } else { kernels.scale };
            let state = pipeline(ctx, name)?;
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&state);
            enc.set_buffer(0, Some(c.buffer), 0);
            set_params(enc, 1, &params);
            dispatch_threads(enc, params.count);
            enc.end_encoding();
            Ok(())
        }
        Plan::Tiled(params) => {
            let state = pipeline(ctx, "contract_tiled_f32")?;
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&state);
            enc.set_buffer(0, Some(a.buffer), 0);
            enc.set_buffer(1, Some(b.buffer), 0);
            enc.set_buffer(2, Some(c.buffer), 0);
            set_params(enc, 3, &params);
            let depth = u64::from(params.batch);
            enc.dispatch_thread_groups(
                metal::MTLSize::new(
                    u64::from(params.n.div_ceil(TILE_N)),
                    u64::from(params.m.div_ceil(TILE_M)),
                    depth,
                ),
                metal::MTLSize::new(u64::from(TILE_THREADS), 1, 1),
            );
            enc.end_encoding();
            Ok(())
        }
        Plan::Split { tiled, reduce, partial_elems } => {
            let partial = alloc_partial(ctx, partial_elems);
            let state = pipeline(ctx, "contract_partial_f32")?;
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&state);
            enc.set_buffer(0, Some(a.buffer), 0);
            enc.set_buffer(1, Some(b.buffer), 0);
            enc.set_buffer(2, Some(&partial), 0);
            set_params(enc, 3, &tiled);
            let depth = u64::from(tiled.batch) * u64::from(tiled.splits);
            enc.dispatch_thread_groups(
                metal::MTLSize::new(
                    u64::from(tiled.n.div_ceil(TILE_N)),
                    u64::from(tiled.m.div_ceil(TILE_M)),
                    depth,
                ),
                metal::MTLSize::new(u64::from(TILE_THREADS), 1, 1),
            );
            enc.end_encoding();

            let state = pipeline(ctx, "contract_reduce_f32")?;
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&state);
            enc.set_buffer(0, Some(&partial), 0);
            enc.set_buffer(1, Some(c.buffer), 0);
            set_params(enc, 2, &reduce);
            dispatch_threads(enc, reduce.count);
            enc.end_encoding();
            track(cmd, partial);
            Ok(())
        }
        Plan::General(params) => {
            let state = pipeline(ctx, kernels.general)?;
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&state);
            enc.set_buffer(0, Some(a.buffer), 0);
            enc.set_buffer(1, Some(b.buffer), 0);
            enc.set_buffer(2, Some(c.buffer), 0);
            set_params(enc, 3, &params);
            enc.dispatch_thread_groups(
                metal::MTLSize::new(
                    u64::from(params.n.div_ceil(GENERAL_TILE)),
                    u64::from(params.m.div_ceil(GENERAL_TILE)),
                    u64::from(params.batch),
                ),
                metal::MTLSize::new(32, 1, 1),
            );
            enc.end_encoding();
            Ok(())
        }
    }
}

fn finish(cmd: &metal::CommandBufferRef) -> Result<()> {
    cmd.commit();
    cmd.wait_until_completed();
    if cmd.status() == metal::MTLCommandBufferStatus::Error {
        return Err(LinalgError::ShaderCompile("command buffer failed".into()));
    }
    Ok(())
}

/// Encodes `C = α·A·B + β·C` for labelled operands into `cmd`.
///
/// A label shared by A and B but absent from C is contracted. A label in A and C is an M
/// axis, a label in B and C is an N axis, and a label in all three is a batch axis. Every
/// label appears in two or three operands, once each, with a single extent. `β = 0` does
/// not read C (a NaN already there does not propagate) and `α = 0` does not read A or B.
///
/// # Errors
///
/// Returns [`LinalgError::InvalidArgument`] for a bad label, extent, rank, or view, and
/// [`LinalgError::Overlap`] when C aliases an input it also reads, or writes an element twice.
/// The in-place scale `C = α·C·s` (the other two extents are 1) is allowed.
pub fn encode_contract<T: Element>(
    ctx: &MetalContext,
    cmd: &metal::CommandBufferRef,
    alpha: f32,
    a: Operand<'_, '_, T>,
    b: Operand<'_, '_, T>,
    beta: f32,
    c: Operand<'_, '_, T>,
) -> Result<()> {
    let (av, bv, cv) = (a.0, b.0, c.0);
    let plan = plan_contract(alpha, a, b, beta, c)?;
    encode_plan(ctx, cmd, plan, av, bv, cv)
}

/// Synchronous [`encode_contract`]: commits and waits for completion.
///
/// Host reads of the destination are valid after this returns.
pub fn contract<T: Element>(
    ctx: &MetalContext,
    alpha: f32,
    a: Operand<'_, '_, T>,
    b: Operand<'_, '_, T>,
    beta: f32,
    c: Operand<'_, '_, T>,
) -> Result<()> {
    let cmd = ctx.queue.new_command_buffer();
    encode_contract(ctx, cmd, alpha, a, b, beta, c)?;
    finish(cmd)
}

/// Encodes the batched matrix product `C = α·A·B + β·C` into `cmd`.
///
/// Each view has rank `r ≥ 2` and the same rank. The leading `r − 2` axes are the batch,
/// and the matrices are `A[..., M, K]`, `B[..., K, N]`, `C[..., M, N]`. Strides may describe
/// a transpose or a strided window; nothing is copied. The scalar rules match
/// [`encode_contract`].
///
/// # Errors
///
/// Returns [`LinalgError::InvalidArgument`] when ranks, batch extents, or inner extents
/// disagree, and [`LinalgError::Overlap`] under the same aliasing rules as [`encode_contract`].
pub fn encode_gemm<T: Element>(
    ctx: &MetalContext,
    cmd: &metal::CommandBufferRef,
    alpha: f32,
    a: &TensorView<'_, T>,
    b: &TensorView<'_, T>,
    beta: f32,
    c: &TensorView<'_, T>,
) -> Result<()> {
    let plan = plan_gemm(alpha, a, b, beta, c)?;
    encode_plan(ctx, cmd, plan, a, b, c)
}

/// Synchronous [`encode_gemm`]: commits and waits for completion.
pub fn gemm<T: Element>(
    ctx: &MetalContext,
    alpha: f32,
    a: &TensorView<'_, T>,
    b: &TensorView<'_, T>,
    beta: f32,
    c: &TensorView<'_, T>,
) -> Result<()> {
    let cmd = ctx.queue.new_command_buffer();
    encode_gemm(ctx, cmd, alpha, a, b, beta, c)?;
    finish(cmd)
}
