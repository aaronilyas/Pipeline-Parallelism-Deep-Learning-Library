//! Tests for zero-copy slicing and GPU packing / concatenation.
//!
//! Packing and concatenation move raw bits, so every comparison is on bytes. The GPU
//! results are checked against an inline reference that walks logical indices on the CPU.

use std::ffi::c_void;
use std::mem::size_of;
use std::ops::Range;
use std::time::Instant;

use half::f16;
use linalg::abi::{
    alloc_shared, compute_pipeline, row_major_strides, LinalgError, MetalContext, TensorView,
};
use linalg::ops::slicing::{
    concat, encode_concat, encode_pack, pack, slice, MAX_CONCAT_PARTS, MAX_RANK,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const SENTINEL: u8 = 0xA5;

/// Plain streaming copy used as the bandwidth baseline. Compiled under `slicing_bench`.
const BENCH_COPY: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void bench_copy(device const ulong4* src [[buffer(0)]],
                       device ulong4* dst [[buffer(1)]],
                       constant uint& n [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < n) dst[i] = src[i];
}
"#;

fn metal_context() -> MetalContext {
    MetalContext::new().expect("Metal device")
}

fn rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

/// Sentinel bit patterns for one element width. Payloads, not just quiet NaNs.
fn specials_for(size: usize) -> Vec<Vec<u8>> {
    match size {
        2 => [0x0000u16, 0x8000, 0x7c00, 0xfc00, 0x7e00, 0x7c01, 0x0001, 0x8001, 0x3c00, 0xfbff]
            .into_iter()
            .map(|b| b.to_le_bytes().to_vec())
            .collect(),
        4 => [
            0x0000_0000u32,
            0x8000_0000,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0000,
            0xffc0_0001,
            0x7f80_0001,
            0x0000_0001,
            0x8000_0001,
            0x7f7f_ffff,
        ]
        .into_iter()
        .map(|b| b.to_le_bytes().to_vec())
        .collect(),
        8 => [
            0x0000_0000_0000_0000u64,
            0x8000_0000_0000_0000,
            0x7ff0_0000_0000_0000,
            0xfff0_0000_0000_0000,
            0x7ff8_0000_0000_0001,
            0xfff8_0000_0000_0001,
            0x7ff0_0000_0000_0001,
            0x0000_0000_0000_0001,
            0x8000_0000_0000_0001,
            0x7fef_ffff_ffff_ffff,
        ]
        .into_iter()
        .map(|b| b.to_le_bytes().to_vec())
        .collect(),
        16 => {
            let lanes = specials_for(8);
            let mut out = Vec::new();
            for (i, hi) in lanes.iter().enumerate() {
                let lo = &lanes[(i + 3) % lanes.len()];
                let mut bytes = lo.clone();
                bytes.extend_from_slice(hi);
                out.push(bytes);
            }
            out
        }
        _ => Vec::new(),
    }
}

/// Complex f32 pairs: independent payloads in the real and imaginary words.
fn complex_specials() -> Vec<Vec<u8>> {
    let words = specials_for(4);
    let mut out = Vec::new();
    for (i, re) in words.iter().enumerate() {
        let im = &words[(i * 3 + 1) % words.len()];
        let mut bytes = re.clone();
        bytes.extend_from_slice(im);
        out.push(bytes);
    }
    out
}

fn buffer_bytes(buf: &metal::BufferRef) -> Vec<u8> {
    let n = buf.length() as usize;
    // SAFETY: shared allocation of `n` bytes, no command buffer in flight.
    unsafe { std::slice::from_raw_parts(buf.contents() as *const u8, n).to_vec() }
}

fn fill_sentinel(buf: &metal::BufferRef) {
    let n = buf.length() as usize;
    // SAFETY: shared allocation, exclusively owned by the host here.
    unsafe { std::ptr::write_bytes(buf.contents() as *mut u8, SENTINEL, n) };
}

fn fill_indexed(buf: &metal::BufferRef, seed: u8) {
    let n = buf.length() as usize;
    // SAFETY: shared allocation, exclusively owned by the host here.
    let bytes = unsafe { std::slice::from_raw_parts_mut(buf.contents() as *mut u8, n) };
    for (i, b) in bytes.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(17).wrapping_add(seed);
    }
}

fn fill_random(buf: &metal::BufferRef, rng: &mut ChaCha8Rng) {
    let n = buf.length() as usize / 4;
    // SAFETY: length is a multiple of the 16 KB granule, so of 4 bytes.
    let words = unsafe { std::slice::from_raw_parts_mut(buf.contents() as *mut u32, n) };
    for w in words {
        *w = rng.next_u32();
    }
}

fn store_bytes<T>(view: &TensorView<'_, T>, idx: &[usize], bytes: &[u8]) {
    assert_eq!(bytes.len(), size_of::<T>());
    let at = view.element_index(idx) * size_of::<T>();
    // SAFETY: `view` passed validation in the caller; `bytes` is one element.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            (view.host_ptr as *mut u8).add(at),
            size_of::<T>(),
        );
    }
}

fn bump(idx: &mut [usize], shape: &[usize]) -> bool {
    for d in (0..idx.len()).rev() {
        idx[d] += 1;
        if idx[d] < shape[d] {
            return true;
        }
        idx[d] = 0;
    }
    false
}

/// Bytes of `view` in row-major logical order.
fn logical_bytes<T>(view: &TensorView<'_, T>) -> Vec<u8> {
    let es = size_of::<T>();
    let mut out = Vec::with_capacity(view.len().saturating_mul(es));
    if view.is_empty() {
        return out;
    }
    let mut idx = vec![0; view.rank()];
    loop {
        let at = view.element_index(&idx) * es;
        // SAFETY: validated view; one element lies inside the buffer.
        let elt = unsafe { std::slice::from_raw_parts((view.host_ptr as *const u8).add(at), es) };
        out.extend_from_slice(elt);
        if !bump(&mut idx, &view.shape) {
            break;
        }
    }
    out
}

fn paint<T>(image: &mut [u8], dst: &TensorView<'_, T>, logical: &[u8]) {
    let es = size_of::<T>();
    assert_eq!(logical.len(), dst.len() * es, "logical length");
    if dst.is_empty() {
        return;
    }
    let mut idx = vec![0; dst.rank()];
    let mut n = 0;
    loop {
        let at = dst.element_index(&idx) * es;
        image[at..at + es].copy_from_slice(&logical[n * es..(n + 1) * es]);
        n += 1;
        if !bump(&mut idx, &dst.shape) {
            break;
        }
    }
}

fn assert_bytes_eq(label: &str, got: &[u8], want: &[u8]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
        panic!("{label}: byte {i}: got {got:#04x}, want {want:#04x}", got = got[i], want = want[i]);
    }
}

fn assert_pack<T>(
    label: &str,
    ctx: &MetalContext,
    src: &TensorView<'_, T>,
    dst_buf: &metal::BufferRef,
    dst: &TensorView<'_, T>,
) {
    let want = logical_bytes(src);
    pack(ctx, src, dst).unwrap_or_else(|e| panic!("{label}: {e}"));
    let mut image = vec![SENTINEL; dst_buf.length() as usize];
    paint(&mut image, dst, &want);
    assert_bytes_eq(label, &buffer_bytes(dst_buf), &image);
}

fn reference_concat<T>(parts: &[&TensorView<'_, T>], axis: usize, dst_shape: &[usize]) -> Vec<u8> {
    let es = size_of::<T>();
    let mut out = vec![0u8; dst_shape.iter().product::<usize>() * es];
    if dst_shape.contains(&0) || parts.is_empty() {
        return out;
    }
    let mut prefix = vec![0usize];
    for part in parts {
        prefix.push(prefix.last().copied().unwrap() + part.shape[axis]);
    }
    let mut idx = vec![0; dst_shape.len()];
    let mut n = 0;
    loop {
        let along = idx[axis];
        let k = (0..parts.len()).find(|&k| along < prefix[k + 1]).expect("prefix covers axis");
        let mut local = idx.clone();
        local[axis] = along - prefix[k];
        let at = parts[k].element_index(&local) * es;
        // SAFETY: part views are validated.
        let elt =
            unsafe { std::slice::from_raw_parts((parts[k].host_ptr as *const u8).add(at), es) };
        out[n * es..(n + 1) * es].copy_from_slice(elt);
        n += 1;
        if !bump(&mut idx, dst_shape) {
            break;
        }
    }
    out
}

fn assert_concat<T>(
    label: &str,
    ctx: &MetalContext,
    parts: &[&TensorView<'_, T>],
    axis: usize,
    dst_buf: &metal::BufferRef,
    dst: &TensorView<'_, T>,
) {
    let want = reference_concat(parts, axis, &dst.shape);
    concat(ctx, parts, axis, dst).unwrap_or_else(|e| panic!("{label}: {e}"));
    let mut image = vec![SENTINEL; dst_buf.length() as usize];
    paint(&mut image, dst, &want);
    assert_bytes_eq(label, &buffer_bytes(dst_buf), &image);
}

fn fill_specials<T>(view: &TensorView<'_, T>, pats: &[Vec<u8>]) {
    if view.is_empty() || pats.is_empty() {
        return;
    }
    let mut idx = vec![0; view.rank()];
    let mut n = 0usize;
    loop {
        store_bytes(view, &idx, &pats[n % pats.len()]);
        n += 1;
        if !bump(&mut idx, &view.shape) {
            break;
        }
    }
}

fn invalid(result: Result<(), LinalgError>) -> bool {
    matches!(result, Err(LinalgError::InvalidArgument(_)))
}

// ---------------------------------------------------------------------------------------
// slice
// ---------------------------------------------------------------------------------------

#[test]
fn slice_keeps_pointer_identity_and_applies_the_offset_formula() {
    let ctx = metal_context();
    let buf = alloc_shared::<f32>(&ctx, 4096);
    fill_indexed(&buf, 9);
    let parent = TensorView::<f32>::new(&buf, 4, vec![3, 5, 7], vec![40, 8, 2]);
    let out = slice(&parent, &[1..3, 2..2, 3..7]).unwrap();
    assert_eq!(out.host_ptr, parent.host_ptr);
    assert_eq!(out.buffer.contents(), parent.buffer.contents());
    assert_eq!(out.strides, parent.strides);
    // offset + 1*40 + 2*8 + 3*2, including the empty axis.
    assert_eq!(out.offset, 4 + 40 + 16 + 6);
    assert_eq!(out.shape, vec![2, 0, 4]);
    assert!(out.is_empty() && out.validate().is_ok());

    let again = slice(&parent, &[0..2, 1..4, 0..5]).unwrap();
    assert_eq!(again.element_index(&[1, 2, 3]), parent.element_index(&[1, 3, 3]));
    let nested = slice(&again, &[1..2, 0..2, 1..4]).unwrap();
    assert_eq!(nested.host_ptr, parent.host_ptr);
    // Includes the zero-length axis: 1*40 + 0*8 + 1*2.
    let starts = [1usize, 0, 1];
    let added: usize =
        starts.iter().zip(&again.strides).map(|(start, stride)| start * stride).sum();
    assert_eq!(nested.offset, again.offset + added);
    assert_eq!(nested.shape, vec![1, 2, 3]);

    let scalar = TensorView::<f32>::new(&buf, 6, vec![], vec![]);
    let same = slice(&scalar, &[]).unwrap();
    assert_eq!(same.offset, 6);
    assert!(same.shape.is_empty());
    assert_eq!(same.host_ptr, scalar.host_ptr);
}

#[test]
fn slice_rejects_out_of_bounds_inverted_and_rank_mismatch() {
    let ctx = metal_context();
    let buf = alloc_shared::<u16>(&ctx, 256);
    let v = TensorView::<u16>::row_major(&buf, 0, vec![4, 6]);
    let before = buffer_bytes(&buf);
    for ranges in [
        vec![0..5, 0..6],
        vec![0..4, 0..7],
        vec![4..4, 7..7],
        vec![Range { start: 3, end: 2 }, 0..6],
        vec![0..4, Range { start: 5, end: 4 }],
        std::iter::once(0..4).collect(),
        vec![0..4, 0..6, 0..1],
        vec![],
    ] {
        assert!(invalid(slice(&v, &ranges).map(|_| ())), "ranges {ranges:?} must be rejected");
    }
    assert_eq!(buffer_bytes(&buf), before, "a rejected slice must not write");
    // Empty at both ends is legal, including a start on the extent.
    assert_eq!(slice(&v, &[4..4, 6..6]).unwrap().shape, vec![0, 0]);
    assert_eq!(slice(&v, &[0..0, 0..0]).unwrap().shape, vec![0, 0]);
    assert!(slice(&v, &[4..4, 0..6]).unwrap().validate().is_ok());
}

// ---------------------------------------------------------------------------------------
// pack
// ---------------------------------------------------------------------------------------

fn pack_sweep<T: Copy>(ctx: &MetalContext, label: &str) {
    const INNERS: [usize; 14] = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33];
    const OFFS: [usize; 5] = [0, 1, 2, 3, 4];
    const SLOT: usize = 64;
    let n = INNERS.len() * OFFS.len() * OFFS.len();
    let src_buf = alloc_shared::<T>(ctx, n * SLOT);
    let dst_buf = alloc_shared::<T>(ctx, n * SLOT);
    fill_indexed(&src_buf, 11);
    fill_sentinel(&dst_buf);
    let es = size_of::<T>();
    let mut expect = vec![SENTINEL; dst_buf.length() as usize];
    let mut jobs: Vec<(TensorView<'_, T>, TensorView<'_, T>, String)> = Vec::with_capacity(n);
    let mut slot = 0usize;
    for &inner in &INNERS {
        for &so in &OFFS {
            for &d_off in &OFFS {
                let src_off = slot * SLOT + so;
                let dst_off = slot * SLOT + d_off;
                let src = TensorView::<T>::row_major(&src_buf, src_off, vec![inner]);
                let dst = TensorView::<T>::row_major(&dst_buf, dst_off, vec![inner]);
                paint(&mut expect, &dst, &logical_bytes(&src));
                jobs.push((
                    src,
                    dst,
                    format!("{label} inner {inner} src_off {so} dst_off {d_off}"),
                ));
                slot += 1;
            }
        }
    }
    let cmd = ctx.queue.new_command_buffer();
    for (src, dst, _) in &jobs {
        encode_pack(ctx, cmd, src, dst).unwrap();
    }
    assert_eq!(cmd.status(), metal::MTLCommandBufferStatus::NotEnqueued);
    cmd.commit();
    cmd.wait_until_completed();
    let got = buffer_bytes(&dst_buf);
    if let Some(i) = (0..got.len()).find(|&i| got[i] != expect[i]) {
        let elem = i / es;
        let which = elem / SLOT;
        let name = jobs.get(which).map(|j| j.2.as_str()).unwrap_or("outside slots");
        panic!("{name}: byte {i}: got {:#04x}, want {:#04x}", got[i], expect[i]);
    }
}

#[test]
fn pack_every_offset_mod_4_covers_head_vector_and_tail() {
    let ctx = metal_context();
    pack_sweep::<f16>(&ctx, "f16");
    pack_sweep::<f32>(&ctx, "f32");
    pack_sweep::<f64>(&ctx, "f64");
    pack_sweep::<[f32; 2]>(&ctx, "c32");
    pack_sweep::<[u64; 2]>(&ctx, "u128");
}

fn check_shaped_pack<T: Copy>(
    ctx: &MetalContext,
    label: &str,
    src_offset: usize,
    shape: &[usize],
    src_strides: &[usize],
    dst_offset: usize,
    dst_strides: &[usize],
) {
    let src_span = span(src_offset, shape, src_strides);
    let dst_span = span(dst_offset, shape, dst_strides);
    let src_buf = alloc_shared::<T>(ctx, src_span);
    let dst_buf = alloc_shared::<T>(ctx, dst_span + 8);
    fill_indexed(&src_buf, 21);
    fill_sentinel(&dst_buf);
    let src = TensorView::<T>::new(&src_buf, src_offset, shape.to_vec(), src_strides.to_vec());
    let dst = TensorView::<T>::new(&dst_buf, dst_offset, shape.to_vec(), dst_strides.to_vec());
    assert_pack(&format!("{label} {}", std::any::type_name::<T>()), ctx, &src, &dst_buf, &dst);
}

fn span(offset: usize, shape: &[usize], strides: &[usize]) -> usize {
    if shape.contains(&0) {
        return offset + 1;
    }
    offset + 1 + shape.iter().zip(strides).map(|(&n, &s)| (n - 1).saturating_mul(s)).sum::<usize>()
}

#[test]
fn pack_strided_rows_hit_head_vector_and_tail() {
    let ctx = metal_context();
    // f32: 16-byte alignment is 4 elements. Offset 1..3 forces a scalar head.
    // cols 16 is a multiple of 4 so a row-major destination keeps that alignment on
    // every row; ld 20 is a padded inner-stride-1 source. inner 16 with head 3 is
    // 3 scalar + one 32-byte vector + one 16-byte vector + 1 scalar.
    for off in 0..4 {
        check_shaped_pack::<f32>(
            &ctx,
            &format!("f32 rows off {off}"),
            off,
            &[3, 16],
            &[20, 1],
            off,
            &row_major_strides(&[3, 16]),
        );
    }
    // f64: offset 1 is an 8-byte head. cols 8 → head 1 + ulong4(4) + ulong2(2) + tail 1.
    check_shaped_pack::<f64>(
        &ctx,
        "f64 rows",
        1,
        &[4, 8],
        &[10, 1],
        1,
        &row_major_strides(&[4, 8]),
    );
    // f16: offset 1 is a 2-byte head (7 elements to a 16-byte boundary).
    check_shaped_pack::<f16>(
        &ctx,
        "f16 rows",
        1,
        &[2, 24],
        &[32, 1],
        1,
        &row_major_strides(&[2, 24]),
    );
    // 16-byte elements are always 16-byte aligned; remainder uses ulong2, tail is empty.
    check_shaped_pack::<[u64; 2]>(
        &ctx,
        "u128 rows",
        0,
        &[2, 5],
        &[8, 1],
        0,
        &row_major_strides(&[2, 5]),
    );
    // Rank 3, inner stride 1, outer strides a multiple of the 16-byte quantum.
    check_shaped_pack::<f32>(
        &ctx,
        "f32 rank3",
        0,
        &[3, 4, 8],
        &[80, 16, 1],
        4,
        &row_major_strides(&[3, 4, 8]),
    );
}

#[test]
fn pack_nonunit_inner_stride_broadcast_and_rank_edges() {
    let ctx = metal_context();
    check_shaped_pack::<f32>(&ctx, "gather", 2, &[32], &[3], 5, &[1]);
    check_shaped_pack::<f16>(&ctx, "gather f16", 1, &[17], &[5], 3, &[1]);
    check_shaped_pack::<[f32; 2]>(&ctx, "gather c32", 0, &[9], &[2], 1, &[1]);
    // Source rows all read the same address; the destination is injective.
    check_shaped_pack::<f32>(
        &ctx,
        "broadcast",
        0,
        &[5, 8],
        &[0, 1],
        0,
        &row_major_strides(&[5, 8]),
    );
    check_shaped_pack::<f64>(&ctx, "rank0", 3, &[], &[], 1, &[]);
    let shape = vec![2u8; MAX_RANK];
    let shape: Vec<usize> = shape.into_iter().map(|n| n as usize).collect();
    let strides = row_major_strides(&shape);
    check_shaped_pack::<u16>(&ctx, "rank8", 0, &shape, &strides, 0, &strides);

    let c = metal_context();
    let buf = alloc_shared::<f32>(&c, 64);
    fill_indexed(&buf, 1);
    let dst_buf = alloc_shared::<f32>(&c, 64);
    fill_sentinel(&dst_buf);
    let src = TensorView::<f32>::new(&buf, 0, vec![4, 4], vec![0, 1]);
    // Destination rows alias (outer stride 0). A parallel store would race.
    let dst = TensorView::<f32>::new(&dst_buf, 0, vec![4, 4], vec![0, 1]);
    assert!(invalid(pack(&c, &src, &dst)), "aliasing destination must be rejected");
    assert!(buffer_bytes(&dst_buf).iter().all(|&b| b == SENTINEL));
}

#[test]
fn pack_special_bit_patterns_survive() {
    let ctx = metal_context();
    fn run<T: Copy>(ctx: &MetalContext, pats: &[Vec<u8>], shape: &[usize], src_strides: &[usize]) {
        let src_buf = alloc_shared::<T>(ctx, span(1, shape, src_strides));
        let dst_buf = alloc_shared::<T>(ctx, span(2, shape, &row_major_strides(shape)));
        fill_sentinel(&src_buf);
        fill_sentinel(&dst_buf);
        let src = TensorView::<T>::new(&src_buf, 1, shape.to_vec(), src_strides.to_vec());
        fill_specials(&src, pats);
        let dst = TensorView::<T>::row_major(&dst_buf, 2, shape.to_vec());
        let label = format!("specials {} {shape:?}", std::any::type_name::<T>());
        assert_pack(&label, ctx, &src, &dst_buf, &dst);
    }
    run::<f16>(&ctx, &specials_for(2), &[6, 5], &[8, 1]);
    run::<f32>(&ctx, &specials_for(4), &[6, 5], &[8, 1]);
    run::<f64>(&ctx, &specials_for(8), &[4, 5], &[6, 1]);
    run::<[f32; 2]>(&ctx, &complex_specials(), &[4, 5], &[6, 1]);
    run::<[u64; 2]>(&ctx, &specials_for(16), &[3, 4], &[5, 1]);
    // Also the real f16 type, not only its width: values come from `from_bits`.
    let bits = [0x7e00u16, 0x8000, 0x0001, 0xfc00];
    let src_buf = alloc_shared::<f16>(&ctx, 8);
    let dst_buf = alloc_shared::<f16>(&ctx, 8);
    fill_sentinel(&dst_buf);
    let src = TensorView::<f16>::row_major(&src_buf, 0, vec![4]);
    for (i, &b) in bits.iter().enumerate() {
        store_bytes(&src, &[i], &f16::from_bits(b).to_bits().to_le_bytes());
    }
    let dst = TensorView::<f16>::row_major(&dst_buf, 1, vec![4]);
    assert_pack("f16::from_bits", &ctx, &src, &dst_buf, &dst);
}

// ---------------------------------------------------------------------------------------
// concat and quadrant round trips
// ---------------------------------------------------------------------------------------

fn pack_fresh<T: Copy>(ctx: &MetalContext, view: &TensorView<'_, T>) -> metal::Buffer {
    let buf = alloc_shared::<T>(ctx, view.len());
    let dst = TensorView::<T>::row_major(&buf, 0, view.shape.clone());
    pack(ctx, view, &dst).unwrap_or_else(|e| panic!("pack {:?}: {e}", view.shape));
    buf
}

fn concat_fresh<T: Copy>(
    ctx: &MetalContext,
    parts: &[&TensorView<'_, T>],
    axis: usize,
    shape: Vec<usize>,
) -> metal::Buffer {
    let buf = alloc_shared::<T>(ctx, shape.iter().product());
    let dst = TensorView::<T>::row_major(&buf, 0, shape);
    concat(ctx, parts, axis, &dst).unwrap_or_else(|e| panic!("concat: {e}"));
    buf
}

fn rebuild<T: Copy>(
    ctx: &MetalContext,
    view: &TensorView<'_, T>,
    depth: usize,
    empties: &mut usize,
) -> metal::Buffer {
    if depth == 0 || view.is_empty() || view.rank() < 2 {
        if view.is_empty() {
            *empties += 1;
        }
        return pack_fresh(ctx, view);
    }
    let (m, n) = (view.shape[0], view.shape[1]);
    let (rm, cn) = (m / 2, n / 2);
    let q = [
        slice(view, &[0..rm, 0..cn]).unwrap(),
        slice(view, &[0..rm, cn..n]).unwrap(),
        slice(view, &[rm..m, 0..cn]).unwrap(),
        slice(view, &[rm..m, cn..n]).unwrap(),
    ];
    assert!(q.iter().all(|p| p.host_ptr == view.host_ptr));
    let kids = [
        rebuild(ctx, &q[0], depth - 1, empties),
        rebuild(ctx, &q[1], depth - 1, empties),
        rebuild(ctx, &q[2], depth - 1, empties),
        rebuild(ctx, &q[3], depth - 1, empties),
    ];
    let views = [
        TensorView::<T>::row_major(&kids[0], 0, q[0].shape.clone()),
        TensorView::<T>::row_major(&kids[1], 0, q[1].shape.clone()),
        TensorView::<T>::row_major(&kids[2], 0, q[2].shape.clone()),
        TensorView::<T>::row_major(&kids[3], 0, q[3].shape.clone()),
    ];
    let top = concat_fresh(ctx, &[&views[0], &views[1]], 1, vec![q[0].shape[0], n]);
    let bot = concat_fresh(ctx, &[&views[2], &views[3]], 1, vec![q[2].shape[0], n]);
    let top_v = TensorView::<T>::row_major(&top, 0, vec![q[0].shape[0], n]);
    let bot_v = TensorView::<T>::row_major(&bot, 0, vec![q[2].shape[0], n]);
    concat_fresh(ctx, &[&top_v, &bot_v], 0, view.shape.clone())
}

fn round_trip<T: Copy>(
    ctx: &MetalContext,
    m: usize,
    n: usize,
    ld: Option<usize>,
    depth: usize,
    want_empty: bool,
) {
    let strides = match ld {
        Some(ld) => vec![ld, 1],
        None => row_major_strides(&[m, n]),
    };
    let src_buf = alloc_shared::<T>(ctx, span(0, &[m, n], &strides));
    fill_indexed(&src_buf, 33);
    let src = TensorView::<T>::new(&src_buf, 0, vec![m, n], strides);
    let size = size_of::<T>();
    let pats = if size == 8 && std::any::type_name::<T>().contains("[f32") {
        complex_specials()
    } else {
        specials_for(size)
    };
    fill_specials(&src, &pats);
    let want = logical_bytes(&src);
    let mut empties = 0usize;
    let got = rebuild(ctx, &src, depth, &mut empties);
    let got_v = TensorView::<T>::row_major(&got, 0, vec![m, n]);
    let label =
        format!("round trip {} {m}x{n} ld {ld:?} depth {depth}", std::any::type_name::<T>());
    assert_bytes_eq(&label, &logical_bytes(&got_v), &want);
    if want_empty {
        assert!(empties > 0, "{label} should have produced an empty quadrant");
    }
}

#[test]
fn quadrant_round_trip_127x251_and_empty_blocks() {
    let ctx = metal_context();
    for ld in [None, Some(252usize)] {
        round_trip::<f32>(&ctx, 127, 251, ld, 3, false);
        round_trip::<f16>(&ctx, 127, 251, ld.map(|_| 256), 3, false);
        round_trip::<f64>(&ctx, 127, 251, ld.map(|_| 256), 3, false);
        round_trip::<[f32; 2]>(&ctx, 127, 251, ld.map(|_| 256), 3, false);
    }
    // 5×3 split by integer division reaches a 0-extent quadrant within depth 3.
    round_trip::<f32>(&ctx, 5, 3, Some(8), 3, true);
    round_trip::<f16>(&ctx, 5, 3, None, 4, true);
    round_trip::<f64>(&ctx, 7, 1, None, 3, true);
    round_trip::<[f32; 2]>(&ctx, 3, 5, Some(8), 3, true);
}

#[test]
fn concat_strided_quadrants_directly() {
    let ctx = metal_context();
    fn run<T: Copy>(ctx: &MetalContext, m: usize, n: usize, ld: usize) {
        let src_buf = alloc_shared::<T>(ctx, m * ld + 4);
        fill_indexed(&src_buf, 45);
        let src = TensorView::<T>::new(&src_buf, 2, vec![m, n], vec![ld, 1]);
        let (rm, cn) = (m / 2, n / 2);
        let q = [
            slice(&src, &[0..rm, 0..cn]).unwrap(),
            slice(&src, &[0..rm, cn..n]).unwrap(),
            slice(&src, &[rm..m, 0..cn]).unwrap(),
            slice(&src, &[rm..m, cn..n]).unwrap(),
        ];
        let top = concat_fresh(ctx, &[&q[0], &q[1]], 1, vec![rm, n]);
        let bot = concat_fresh(ctx, &[&q[2], &q[3]], 1, vec![m - rm, n]);
        let top_v = TensorView::<T>::row_major(&top, 0, vec![rm, n]);
        let bot_v = TensorView::<T>::row_major(&bot, 0, vec![m - rm, n]);
        let full = concat_fresh(ctx, &[&top_v, &bot_v], 0, vec![m, n]);
        let full_v = TensorView::<T>::row_major(&full, 0, vec![m, n]);
        assert_bytes_eq(
            &format!("direct {} {m}x{n}", std::any::type_name::<T>()),
            &logical_bytes(&full_v),
            &logical_bytes(&src),
        );
    }
    run::<f32>(&ctx, 127, 251, 260);
    run::<f16>(&ctx, 15, 17, 24);
    run::<f64>(&ctx, 9, 13, 16);
    run::<[f32; 2]>(&ctx, 8, 9, 12);
}

#[test]
fn concat_each_axis_of_rank3_including_empty_blocks() {
    let ctx = metal_context();
    let src_buf = alloc_shared::<f32>(&ctx, 8 * 8 * 8);
    fill_indexed(&src_buf, 70);
    let src = TensorView::<f32>::new(&src_buf, 1, vec![6, 5, 7], vec![40, 8, 1]);
    // Axis 0, with an empty block between the two non-empty slices.
    let a = slice(&src, &[0..2, 0..5, 0..7]).unwrap();
    let gap = slice(&src, &[2..2, 0..5, 0..7]).unwrap();
    let b = slice(&src, &[2..6, 0..5, 0..7]).unwrap();
    assert!(gap.is_empty());
    let shape0 = vec![6, 5, 7];
    let dst0 = alloc_shared::<f32>(&ctx, 6 * 5 * 7 + 3);
    fill_sentinel(&dst0);
    let view0 = TensorView::<f32>::row_major(&dst0, 3, shape0);
    assert_concat("axis0", &ctx, &[&a, &gap, &b], 0, &dst0, &view0);

    // Axis 1: three parts, the middle one empty, and one part is column-strided.
    let wide = alloc_shared::<u16>(&ctx, 256);
    fill_indexed(&wide, 81);
    let w = TensorView::<u16>::new(&wide, 4, vec![3, 10, 2], vec![20, 2, 1]);
    let p0 = slice(&w, &[0..3, 0..1, 0..2]).unwrap();
    let p1 = slice(&w, &[0..3, 1..1, 0..2]).unwrap();
    let p2 = slice(&w, &[0..3, 1..4, 0..2]).unwrap();
    // A second source buffer with a different stride on the free axes.
    // Extents on axis 1: 1 + 0 + 2 + 3 = 6.
    let other = alloc_shared::<u16>(&ctx, 64);
    fill_indexed(&other, 82);
    let extra = TensorView::<u16>::new(&other, 0, vec![3, 2, 2], vec![8, 2, 1]);
    let dst1 = alloc_shared::<u16>(&ctx, 3 * 6 * 2);
    fill_sentinel(&dst1);
    let view1 = TensorView::<u16>::row_major(&dst1, 0, vec![3, 6, 2]);
    assert_concat("axis1", &ctx, &[&p0, &p1, &extra, &p2], 1, &dst1, &view1);

    // Axis 2 of a rank-3 f64 tensor, parts in separate buffers.
    let x = alloc_shared::<f64>(&ctx, 128);
    let y = alloc_shared::<f64>(&ctx, 128);
    fill_indexed(&x, 90);
    fill_indexed(&y, 91);
    let px = TensorView::<f64>::new(&x, 2, vec![2, 2, 3], vec![12, 4, 1]);
    let py = TensorView::<f64>::new(&y, 0, vec![2, 2, 1], vec![2, 1, 1]);
    let dst2 = alloc_shared::<f64>(&ctx, 2 * 2 * 4 + 5);
    fill_sentinel(&dst2);
    let view2 = TensorView::<f64>::row_major(&dst2, 5, vec![2, 2, 4]);
    assert_concat("axis2", &ctx, &[&px, &py], 2, &dst2, &view2);

    // Entirely empty concatenation.
    let e0 = TensorView::<f32>::row_major(&src_buf, 0, vec![0, 4]);
    let e1 = TensorView::<f32>::row_major(&src_buf, 0, vec![0, 4]);
    let edst = alloc_shared::<f32>(&ctx, 16);
    fill_sentinel(&edst);
    let ev = TensorView::<f32>::row_major(&edst, 0, vec![0, 4]);
    assert_concat("empty", &ctx, &[&e0, &e1], 0, &edst, &ev);
    assert!(buffer_bytes(&edst).iter().all(|&b| b == SENTINEL));
}

/// Every part slot, including the last and an empty one in the middle.
#[test]
fn concat_sixteen_parts_with_an_empty_block() {
    let ctx = metal_context();
    let mut bufs = Vec::new();
    for i in 0..MAX_CONCAT_PARTS {
        let len = if i == 7 { 0 } else { 1 };
        let buf = alloc_shared::<f32>(&ctx, len.max(1));
        fill_indexed(&buf, 30 + i as u8);
        bufs.push(buf);
    }
    let views: Vec<TensorView<f32>> = bufs
        .iter()
        .enumerate()
        .map(|(i, buf)| {
            let len = if i == 7 { 0 } else { 1 };
            TensorView::<f32>::row_major(buf, 0, vec![len])
        })
        .collect();
    let refs: Vec<_> = views.iter().collect();
    let dst_buf = alloc_shared::<f32>(&ctx, MAX_CONCAT_PARTS - 1);
    fill_sentinel(&dst_buf);
    let dst = TensorView::<f32>::row_major(&dst_buf, 0, vec![MAX_CONCAT_PARTS - 1]);
    assert_concat("16 parts", &ctx, &refs, 0, &dst_buf, &dst);
}

// ---------------------------------------------------------------------------------------
// encode-only behaviour and rejection
// ---------------------------------------------------------------------------------------

#[test]
fn encode_pack_and_concat_share_one_command_buffer() {
    let ctx = metal_context();
    let src_buf = alloc_shared::<[f32; 2]>(&ctx, 64 * 8);
    fill_indexed(&src_buf, 100);
    let src = TensorView::<[f32; 2]>::new(&src_buf, 3, vec![8, 16], vec![20, 1]);
    let left = slice(&src, &[0..8, 0..7]).unwrap();
    let right = slice(&src, &[0..8, 7..16]).unwrap();
    let mid_buf = alloc_shared::<[f32; 2]>(&ctx, 8 * 16);
    fill_sentinel(&mid_buf);
    let mid = TensorView::<[f32; 2]>::row_major(&mid_buf, 0, vec![8, 16]);
    let dst_buf = alloc_shared::<[f32; 2]>(&ctx, 8 * 16 + 4);
    fill_sentinel(&dst_buf);
    let dst = TensorView::<[f32; 2]>::row_major(&dst_buf, 4, vec![8, 16]);
    let untouched = buffer_bytes(&dst_buf);

    let cmd = ctx.queue.new_command_buffer();
    encode_concat(&ctx, cmd, &[&left, &right], 1, &mid).unwrap();
    encode_pack(&ctx, cmd, &mid, &dst).unwrap();
    assert_eq!(cmd.status(), metal::MTLCommandBufferStatus::NotEnqueued);
    assert_eq!(buffer_bytes(&dst_buf), untouched, "encode must not run the kernel");
    cmd.commit();
    cmd.wait_until_completed();
    assert_eq!(cmd.status(), metal::MTLCommandBufferStatus::Completed);
    assert_bytes_eq("chained", &logical_bytes(&dst), &logical_bytes(&src));
}

#[test]
fn pack_and_concat_reject_overlap_bad_shapes_and_sizes_without_writing() {
    let ctx = metal_context();
    let buf = alloc_shared::<f32>(&ctx, 4096);
    fill_indexed(&buf, 120);
    let before = buffer_bytes(&buf);
    let src = TensorView::<f32>::row_major(&buf, 0, vec![32, 32]);
    let overlap = TensorView::<f32>::row_major(&buf, 16, vec![32, 32]);
    assert!(matches!(pack(&ctx, &src, &overlap), Err(LinalgError::Overlap(_))));
    assert!(matches!(pack(&ctx, &src, &src), Err(LinalgError::Overlap(_))));
    let cmd = ctx.queue.new_command_buffer();
    assert!(matches!(encode_pack(&ctx, cmd, &src, &src), Err(LinalgError::Overlap(_))));
    cmd.commit();
    cmd.wait_until_completed();
    assert_eq!(buffer_bytes(&buf), before);

    let dst_buf = alloc_shared::<f32>(&ctx, 4096);
    fill_sentinel(&dst_buf);
    let dst = TensorView::<f32>::row_major(&dst_buf, 0, vec![32, 32]);
    let wrong = TensorView::<f32>::row_major(&dst_buf, 0, vec![31, 32]);
    assert!(invalid(pack(&ctx, &src, &wrong)));
    let rank9 = vec![1usize; MAX_RANK + 1];
    let big = TensorView::<f32>::new(&buf, 0, rank9.clone(), row_major_strides(&rank9));
    let big_dst = TensorView::<f32>::new(&dst_buf, 0, rank9.clone(), row_major_strides(&rank9));
    assert!(invalid(pack(&ctx, &big, &big_dst)));
    assert!(buffer_bytes(&dst_buf).iter().all(|&b| b == SENTINEL));

    let u8_held = alloc_shared::<u8>(&ctx, 32);
    let u8_src = TensorView::<u8>::row_major(&u8_held, 0, vec![16]);
    let u8_buf = alloc_shared::<u8>(&ctx, 32);
    fill_sentinel(&u8_buf);
    let u8_dst = TensorView::<u8>::row_major(&u8_buf, 0, vec![16]);
    assert!(invalid(pack(&ctx, &u8_src, &u8_dst)));
    assert!(buffer_bytes(&u8_buf).iter().all(|&b| b == SENTINEL));
    let wide_held = alloc_shared::<[u64; 4]>(&ctx, 8);
    let wide_src = TensorView::<[u64; 4]>::row_major(&wide_held, 0, vec![4]);
    let wide_buf = alloc_shared::<[u64; 4]>(&ctx, 8);
    fill_sentinel(&wide_buf);
    let wide_dst = TensorView::<[u64; 4]>::row_major(&wide_buf, 0, vec![4]);
    assert!(invalid(pack(&ctx, &wide_src, &wide_dst)));

    // Disjoint regions of one buffer are not overlap.
    let lo = TensorView::<f32>::row_major(&buf, 0, vec![10]);
    let hi = TensorView::<f32>::row_major(&buf, 10, vec![10]);
    let hi_buf_before = logical_bytes(&lo);
    pack(&ctx, &lo, &hi).unwrap();
    assert_eq!(logical_bytes(&hi), hi_buf_before);

    // Concat rejections.
    let c0 = slice(&src, &[0..10, 0..32]).unwrap();
    let c1 = slice(&src, &[10..32, 0..32]).unwrap();
    assert!(matches!(concat(&ctx, &[&c0, &c1], 0, &src), Err(LinalgError::Overlap(_))));
    assert!(invalid(concat(&ctx, &[], 0, &dst)));
    assert!(invalid(concat(&ctx, &[&c0], 2, &dst)));
    assert!(invalid(concat(&ctx, &[&c0, &c1], 0, &wrong)));
    let one = TensorView::<f32>::row_major(&buf, 0, vec![1]);
    let too_many: Vec<_> = (0..MAX_CONCAT_PARTS + 1).map(|_| one.clone()).collect();
    let refs: Vec<_> = too_many.iter().collect();
    let wide = TensorView::<f32>::row_major(&dst_buf, 0, vec![MAX_CONCAT_PARTS + 1]);
    assert!(invalid(concat(&ctx, &refs, 0, &wide)));
    assert!(buffer_bytes(&dst_buf).iter().all(|&b| b == SENTINEL));

    // Empty pack writes nothing.
    let empty = TensorView::<f32>::row_major(&buf, 0, vec![0, 8]);
    let empty_dst_buf = alloc_shared::<f32>(&ctx, 32);
    fill_sentinel(&empty_dst_buf);
    let empty_dst = TensorView::<f32>::row_major(&empty_dst_buf, 0, vec![0, 8]);
    pack(&ctx, &empty, &empty_dst).unwrap();
    assert!(buffer_bytes(&empty_dst_buf).iter().all(|&b| b == SENTINEL));
}

// ---------------------------------------------------------------------------------------
// Benchmarks (run with --ignored under the exclusive GPU lock)
// ---------------------------------------------------------------------------------------

fn best_gbps(
    ctx: &MetalContext,
    bytes: usize,
    iters: usize,
    reps: usize,
    encode: impl Fn(&metal::CommandBufferRef),
) -> f64 {
    let mut best = f64::MAX;
    for rep in 0..reps + 1 {
        let cmd = ctx.queue.new_command_buffer();
        for _ in 0..iters {
            encode(cmd);
        }
        let start = Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        let secs = start.elapsed().as_secs_f64() / iters as f64;
        if rep > 0 {
            best = best.min(secs);
        }
    }
    2.0 * bytes as f64 / best / 1e9
}

#[test]
#[ignore = "benchmark: run under the exclusive GPU lock"]
fn bench_strided_pack_vs_copy() {
    let ctx = metal_context();
    let (rows, cols, ld) = (2048usize, 8192usize, 8192usize + 128);
    assert!(ld > cols && ld % 4 == 0 && cols % 8 == 0);
    let bytes = rows * cols * size_of::<f32>();
    let src_buf = alloc_shared::<f32>(&ctx, rows * ld);
    fill_random(&src_buf, &mut rng(7));
    let dst_buf = alloc_shared::<f32>(&ctx, rows * cols);
    let src = TensorView::<f32>::new(&src_buf, 0, vec![rows, cols], vec![ld, 1]);
    let dst = TensorView::<f32>::row_major(&dst_buf, 0, vec![rows, cols]);

    let copy_src = alloc_shared::<f32>(&ctx, rows * cols);
    fill_random(&copy_src, &mut rng(8));
    let copy_dst = alloc_shared::<f32>(&ctx, rows * cols);
    let nvec = (bytes / 32) as u32;
    let options = || {
        let opts = metal::CompileOptions::new();
        opts.set_language_version(metal::MTLLanguageVersion::V3_0);
        opts.set_fast_math_enabled(false);
        opts
    };
    let pso = compute_pipeline(&ctx, "slicing_bench", BENCH_COPY, options, "bench_copy").unwrap();
    let copy = |cmd: &metal::CommandBufferRef| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&copy_src), 0);
        enc.set_buffer(1, Some(&copy_dst), 0);
        enc.set_bytes(2, 4, &nvec as *const u32 as *const c_void);
        enc.dispatch_threads(
            metal::MTLSize::new(nvec as u64, 1, 1),
            metal::MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
    };
    let packed = |cmd: &metal::CommandBufferRef| encode_pack(&ctx, cmd, &src, &dst).unwrap();
    let copy_gbps = best_gbps(&ctx, bytes, 8, 4, copy);
    let pack_gbps = best_gbps(&ctx, bytes, 8, 4, packed);
    let ratio = pack_gbps / copy_gbps;
    println!("pack {pack_gbps:8.2} GB/s");
    println!("copy {copy_gbps:8.2} GB/s");
    println!("ratio {ratio:8.3}  ({:.1}% of copy)", 100.0 * ratio);

    let got = logical_bytes(&dst);
    let want = logical_bytes(&src);
    assert_eq!(got.len(), want.len());
    for &i in &[0usize, 1, cols - 1, cols, rows * cols - 1] {
        assert_eq!(got[i * 4..i * 4 + 4], want[i * 4..i * 4 + 4], "element {i}");
    }
    assert!(ratio >= 0.80, "strided pack reaches only {:.1}% of copy", 100.0 * ratio);
}
