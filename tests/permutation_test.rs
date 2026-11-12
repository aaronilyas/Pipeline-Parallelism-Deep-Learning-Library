//! Tests for the permutation primitive: view metadata, materialization against an inline
//! pure-Rust reference, the bitwise adjoint involution, error paths, and benchmarks.
//!
//! The kernels only move bits, so every comparison is on raw 32-bit words.

use std::mem::size_of;
use std::time::Instant;

use linalg::abi::TensorView;
use linalg::abi::{alloc_shared, compute_pipeline, row_major_strides, LinalgError, MetalContext};
use linalg::ops::permutation::{
    adjoint, conjugate, encode_adjoint, encode_conjugate, encode_materialize, materialize,
    permute_view, Element,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const SENTINEL: u32 = 0xDEAD_BEEF;

/// Interesting 32-bit patterns: ±0, ±inf, quiet and signalling NaNs with payloads, subnormals.
const SPECIALS: [u32; 12] = [
    0x0000_0000,
    0x8000_0000,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0000,
    0xffc0_0000,
    0x7fc0_0001,
    0xffc1_2345,
    0x7f80_0001,
    0xff80_0007,
    0x0000_0001,
    0x8000_0001,
];

fn words_per_element<T>() -> usize {
    size_of::<T>() / size_of::<u32>()
}

fn ctx() -> MetalContext {
    MetalContext::new().expect("Metal device")
}

fn rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

fn words_mut(buf: &metal::BufferRef) -> &mut [u32] {
    // SAFETY: shared buffer, no command buffer in flight while tests touch it on the host.
    unsafe { std::slice::from_raw_parts_mut(buf.contents() as *mut u32, buf.length() as usize / 4) }
}

fn fill_random(buf: &metal::BufferRef, rng: &mut ChaCha8Rng) {
    words_mut(buf).iter_mut().for_each(|w| *w = rng.next_u32());
}

fn fill_special(buf: &metal::BufferRef, rng: &mut ChaCha8Rng) {
    for w in words_mut(buf) {
        *w = if rng.next_u32() % 2 == 0 {
            SPECIALS[rng.next_u32() as usize % SPECIALS.len()]
        } else {
            rng.next_u32()
        };
    }
}

fn sentinel_buffer<T>(ctx: &MetalContext, len: usize) -> metal::Buffer {
    let buf = alloc_shared::<T>(ctx, len);
    words_mut(&buf).fill(SENTINEL);
    buf
}

/// One past the highest element index a view with these parameters can reach.
fn span(offset: usize, shape: &[usize], strides: &[usize]) -> usize {
    if shape.contains(&0) {
        return offset;
    }
    offset + 1 + shape.iter().zip(strides).map(|(&n, &s)| (n - 1) * s).sum::<usize>()
}

/// Reference gather: the words of `view` in row-major logical order, optionally conjugated.
fn gather_words<T: Element>(view: &TensorView<'_, T>, conj: bool) -> Vec<u32> {
    let w = words_per_element::<T>();
    let mut out = Vec::with_capacity(view.len() * w);
    if view.is_empty() {
        return out;
    }
    let base = view.host_ptr as *const u32;
    let mut idx = vec![0usize; view.rank()];
    loop {
        let e = view.element_index(&idx);
        for k in 0..w {
            // SAFETY: callers pass views that passed validation.
            let mut word = unsafe { *base.add(e * w + k) };
            if conj && T::COMPLEX && k == 1 {
                word ^= 0x8000_0000;
            }
            out.push(word);
        }
        let mut d = view.rank();
        loop {
            if d == 0 {
                return out;
            }
            d -= 1;
            idx[d] += 1;
            if idx[d] < view.shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
}

fn assert_words_eq(label: &str, got: &[u32], want: &[u32]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    if let Some(i) = (0..got.len()).find(|&i| got[i] != want[i]) {
        panic!("{label}: word {i}: got {:#010x}, want {:#010x}", got[i], want[i]);
    }
}

/// Asserts `dst_buf` holds `expected` at `dst_offset` and the sentinel everywhere else.
fn assert_dst<T>(label: &str, dst_buf: &metal::BufferRef, dst_offset: usize, expected: &[u32]) {
    let w = words_per_element::<T>();
    let mut want = vec![SENTINEL; dst_buf.length() as usize / 4];
    want[dst_offset * w..dst_offset * w + expected.len()].copy_from_slice(expected);
    assert_words_eq(label, words_mut(dst_buf), &want);
}

fn permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![vec![]];
    }
    let mut out = Vec::new();
    for tail in permutations(n - 1) {
        for pos in 0..n {
            let mut p = tail.clone();
            p.insert(pos, n - 1);
            out.push(p);
        }
    }
    out
}

/// Materializes `permute_view(src, perm)` and compares it, element for element, with the
/// reference. `src_strides = None` means row-major.
fn check_materialize<T: Element>(
    ctx: &MetalContext,
    base: &[usize],
    perm: &[usize],
    src_offset: usize,
    src_strides: Option<Vec<usize>>,
    dst_offset: usize,
    seed: u64,
) {
    let strides = src_strides.unwrap_or_else(|| row_major_strides(base));
    let src_buf = alloc_shared::<T>(ctx, span(src_offset, base, &strides));
    fill_random(&src_buf, &mut rng(seed));
    let src = TensorView::<T>::new(&src_buf, src_offset, base.to_vec(), strides);
    let view = permute_view(&src, perm).unwrap();
    let want = gather_words(&view, false);
    let dst_buf = sentinel_buffer::<T>(ctx, dst_offset + view.len() + 17);
    let dst = TensorView::<T>::row_major(&dst_buf, dst_offset, view.shape.clone());
    let label = format!("{} base={base:?} perm={perm:?}", std::any::type_name::<T>());
    materialize(ctx, &view, &dst).unwrap_or_else(|e| panic!("{label}: {e}"));
    assert_dst::<T>(&label, &dst_buf, dst_offset, &want);
}

fn check_all_perms<T: Element>(ctx: &MetalContext, base: &[usize], seed: u64) {
    for (i, perm) in permutations(base.len()).iter().enumerate() {
        check_materialize::<T>(ctx, base, perm, 0, None, 0, seed + i as u64);
    }
}

fn for_each_type(f: impl Fn(&dyn Fn(&MetalContext, &[usize], u64))) {
    f(&|c, b, s| check_all_perms::<f32>(c, b, s));
    f(&|c, b, s| check_all_perms::<u32>(c, b, s));
    f(&|c, b, s| check_all_perms::<[f32; 2]>(c, b, s));
}

// ---------------------------------------------------------------------------------------
// View path
// ---------------------------------------------------------------------------------------

#[test]
fn view_path_keeps_identity_and_permutes_metadata() {
    let ctx = ctx();
    let buf = alloc_shared::<f32>(&ctx, 1024);
    let v = TensorView::<f32>::new(&buf, 5, vec![2, 3, 4], vec![12, 4, 1]);
    let p = permute_view(&v, &[2, 0, 1]).unwrap();
    assert_eq!(p.host_ptr, v.host_ptr);
    assert_eq!(p.buffer.contents(), v.buffer.contents());
    assert_eq!(p.offset, 5);
    assert_eq!(p.shape, vec![4, 2, 3]);
    assert_eq!(p.strides, vec![1, 12, 4]);
    // Axis i of the result is axis perm[i] of the source.
    assert_eq!(p.element_index(&[3, 1, 2]), v.element_index(&[1, 2, 3]));
    assert_eq!(p.len(), v.len());
}

#[test]
fn view_path_identity_and_inverse_round_trip() {
    let ctx = ctx();
    let buf = alloc_shared::<f32>(&ctx, 4096);
    let v = TensorView::<f32>::new(&buf, 3, vec![3, 5, 7, 2], vec![70, 14, 2, 1]);
    let same = permute_view(&v, &[0, 1, 2, 3]).unwrap();
    assert_eq!((same.shape.clone(), same.strides.clone()), (v.shape.clone(), v.strides.clone()));
    let perm = [3, 1, 0, 2];
    let mut inverse = [0; 4];
    for (i, &p) in perm.iter().enumerate() {
        inverse[p] = i;
    }
    let back = permute_view(&permute_view(&v, &perm).unwrap(), &inverse).unwrap();
    assert_eq!((back.shape, back.strides, back.offset), (v.shape, v.strides, v.offset));
}

#[test]
fn view_path_handles_rank_zero_and_one() {
    let ctx = ctx();
    let buf = alloc_shared::<f32>(&ctx, 16);
    let scalar = TensorView::<f32>::new(&buf, 2, vec![], vec![]);
    assert!(permute_view(&scalar, &[]).unwrap().shape.is_empty());
    let vector = TensorView::<f32>::row_major(&buf, 0, vec![9]);
    assert_eq!(permute_view(&vector, &[0]).unwrap().shape, vec![9]);
}

#[test]
fn view_path_rejects_non_permutations() {
    let ctx = ctx();
    let buf = alloc_shared::<f32>(&ctx, 1024);
    let v = TensorView::<f32>::row_major(&buf, 0, vec![2, 3, 4]);
    for bad in [
        &[0usize, 0, 1][..],
        &[0, 1],
        &[0, 1, 2, 3],
        &[0, 1, 3],
        &[3, 1, 0],
        &[usize::MAX, 0, 1],
        &[],
        &[1, 1, 1],
    ] {
        assert!(
            matches!(permute_view(&v, bad), Err(LinalgError::InvalidArgument(_))),
            "perm {bad:?} must be rejected"
        );
    }
}

// ---------------------------------------------------------------------------------------
// Materialization
// ---------------------------------------------------------------------------------------

#[test]
fn materialize_all_perms_rank_1_to_4_prime_extents() {
    let ctx = ctx();
    for_each_type(|check| {
        check(&ctx, &[101], 1);
        check(&ctx, &[37, 41], 10);
        check(&ctx, &[5, 37, 67], 100);
        check(&ctx, &[5, 7, 37, 41], 1000);
    });
}

#[test]
fn materialize_all_perms_rank_5_prime_extents() {
    let ctx = ctx();
    for_each_type(|check| check(&ctx, &[3, 5, 7, 11, 13], 5000));
}

#[test]
fn materialize_all_perms_rank_4_with_unit_and_two_extents() {
    let ctx = ctx();
    for_each_type(|check| {
        check(&ctx, &[1, 2, 3, 37], 7000);
        check(&ctx, &[2, 1, 41, 2], 7100);
    });
}

#[test]
fn materialize_with_source_and_destination_offsets() {
    let ctx = ctx();
    for (n, (so, doff)) in [(0, 0), (1, 3), (13, 7), (33, 1), (4095, 4097)].into_iter().enumerate()
    {
        let seed = 200 + n as u64;
        check_materialize::<f32>(&ctx, &[37, 41], &[1, 0], so, None, doff, seed);
        check_materialize::<[f32; 2]>(&ctx, &[5, 37, 43], &[2, 0, 1], so, None, doff, seed);
        check_materialize::<u32>(&ctx, &[7, 11, 13, 3], &[3, 2, 1, 0], so, None, doff, seed);
    }
}

#[test]
fn materialize_multi_tile_transposes_with_ragged_edges() {
    let ctx = ctx();
    for &(m, n) in &[(32, 32), (33, 31), (64, 64), (100, 130), (129, 65), (31, 257), (1000, 33)] {
        check_materialize::<f32>(&ctx, &[m, n], &[1, 0], 0, None, 0, 300);
        check_materialize::<[f32; 2]>(&ctx, &[m, n], &[1, 0], 0, None, 0, 301);
    }
}

#[test]
fn materialize_batched_transpose_with_many_batches() {
    let ctx = ctx();
    check_materialize::<f32>(&ctx, &[257, 33, 35], &[0, 2, 1], 0, None, 0, 400);
    check_materialize::<[f32; 2]>(&ctx, &[1031, 3, 5], &[0, 2, 1], 0, None, 0, 401);
    check_materialize::<u32>(&ctx, &[3, 4, 5, 6, 7], &[4, 0, 3, 1, 2], 0, None, 0, 402);
}

#[test]
fn materialize_padded_leading_dimension_and_sliced_views() {
    let ctx = ctx();
    // 45 x 40 window inside a 45 x 64 buffer, with and without transposition.
    for perm in [[0, 1], [1, 0]] {
        check_materialize::<f32>(&ctx, &[45, 40], &perm, 5, Some(vec![64, 1]), 3, 500);
        check_materialize::<[f32; 2]>(&ctx, &[45, 40], &perm, 5, Some(vec![64, 1]), 3, 501);
    }
    // Every other column and every third row.
    for perm in [[0, 1], [1, 0]] {
        check_materialize::<u32>(&ctx, &[20, 31], &perm, 2, Some(vec![150, 2]), 0, 502);
        check_materialize::<f32>(&ctx, &[20, 31], &perm, 0, Some(vec![3 * 64, 1]), 0, 503);
    }
    // Column-major source: the view's fastest axis is already unit stride after permuting.
    check_materialize::<f32>(&ctx, &[37, 41], &[1, 0], 0, Some(vec![1, 37]), 0, 504);
    check_materialize::<f32>(&ctx, &[37, 41], &[0, 1], 0, Some(vec![1, 37]), 0, 505);
    // Rank 3 with a padded middle axis.
    check_materialize::<[f32; 2]>(
        &ctx,
        &[4, 9, 33],
        &[2, 0, 1],
        1,
        Some(vec![9 * 40, 40, 1]),
        2,
        506,
    );
}

#[test]
fn materialize_broadcast_zero_strides() {
    let ctx = ctx();
    check_materialize::<f32>(&ctx, &[5, 37], &[1, 0], 0, Some(vec![0, 1]), 0, 600);
    check_materialize::<f32>(&ctx, &[37, 5], &[1, 0], 0, Some(vec![1, 0]), 0, 601);
    check_materialize::<[f32; 2]>(&ctx, &[3, 4, 33], &[2, 1, 0], 0, Some(vec![0, 0, 1]), 0, 602);
}

#[test]
fn materialize_degenerate_extents() {
    let ctx = ctx();
    for &n in &[1usize, 2, 31, 32, 33, 997, 1000] {
        for perm in [[0, 1], [1, 0]] {
            check_materialize::<f32>(&ctx, &[1, n], &perm, 0, None, 0, 700);
            check_materialize::<f32>(&ctx, &[n, 1], &perm, 0, None, 0, 701);
            check_materialize::<[f32; 2]>(&ctx, &[1, n], &perm, 0, None, 0, 702);
            check_materialize::<[f32; 2]>(&ctx, &[n, 1], &perm, 0, None, 0, 703);
        }
    }
    check_materialize::<f32>(&ctx, &[1, 1], &[1, 0], 0, None, 0, 704);
    check_materialize::<f32>(&ctx, &[1, 1, 1], &[2, 0, 1], 0, None, 0, 705);
    // Rank 0: a single element.
    check_materialize::<f32>(&ctx, &[], &[], 3, None, 1, 706);
}

#[test]
fn materialize_zero_sized_extents_succeed_and_write_nothing() {
    let ctx = ctx();
    for shape in [vec![0usize, 5], vec![5, 0], vec![0, 0], vec![3, 0, 4], vec![0]] {
        let perms = permutations(shape.len());
        for perm in &perms {
            check_materialize::<f32>(&ctx, &shape, perm, 0, None, 2, 800);
            check_materialize::<[f32; 2]>(&ctx, &shape, perm, 0, None, 2, 801);
        }
    }
}

#[test]
fn materialize_into_disjoint_region_of_same_buffer() {
    let ctx = ctx();
    let (m, n) = (37, 41);
    let buf = alloc_shared::<f32>(&ctx, 4 * m * n);
    fill_random(&buf, &mut rng(900));
    let src = TensorView::<f32>::row_major(&buf, 0, vec![m, n]);
    let view = permute_view(&src, &[1, 0]).unwrap();
    let want = gather_words(&view, false);
    let dst = TensorView::<f32>::row_major(&buf, 2 * m * n, vec![n, m]);
    materialize(&ctx, &view, &dst).unwrap();
    assert_words_eq("disjoint", &words_mut(&buf)[2 * m * n..3 * m * n], &want);
}

// ---------------------------------------------------------------------------------------
// Adjoint and conjugate
// ---------------------------------------------------------------------------------------

/// Reference adjoint of a row-major `[batch.., m, n]` tensor: swap the last two axes and
/// conjugate complex elements.
fn reference_adjoint<T: Element>(src: &TensorView<'_, T>) -> Vec<u32> {
    let mut perm: Vec<usize> = (0..src.rank()).collect();
    let r = src.rank();
    perm.swap(r - 2, r - 1);
    let swapped = TensorView::<T>::new(
        src.buffer,
        src.offset,
        perm.iter().map(|&p| src.shape[p]).collect(),
        perm.iter().map(|&p| src.strides[p]).collect(),
    );
    gather_words(&swapped, true)
}

fn adjoint_shape(batch: &[usize], m: usize, n: usize) -> (Vec<usize>, Vec<usize>) {
    let mut a = batch.to_vec();
    a.extend([m, n]);
    let mut b = batch.to_vec();
    b.extend([n, m]);
    (a, b)
}

/// A -> B = A^H -> C = B^H on special-value data: checks B against the reference and C
/// against A, bit for bit.
fn check_involution<T: Element>(
    ctx: &MetalContext,
    batch: &[usize],
    m: usize,
    n: usize,
    seed: u64,
) {
    let (a_shape, b_shape) = adjoint_shape(batch, m, n);
    let len: usize = a_shape.iter().product();
    let a_buf = alloc_shared::<T>(ctx, len);
    fill_special(&a_buf, &mut rng(seed));
    let a = TensorView::<T>::row_major(&a_buf, 0, a_shape.clone());
    let a_words = gather_words(&a, false);
    let b_buf = sentinel_buffer::<T>(ctx, len);
    let b = TensorView::<T>::row_major(&b_buf, 0, b_shape);
    let c_buf = sentinel_buffer::<T>(ctx, len);
    let c = TensorView::<T>::row_major(&c_buf, 0, a_shape);
    let label = format!("{} batch={batch:?} {m}x{n}", std::any::type_name::<T>());
    adjoint(ctx, &a, &b).unwrap_or_else(|e| panic!("{label}: {e}"));
    assert_dst::<T>(&format!("{label} B"), &b_buf, 0, &reference_adjoint(&a));
    adjoint(ctx, &b, &c).unwrap_or_else(|e| panic!("{label}: {e}"));
    assert_dst::<T>(&format!("{label} C"), &c_buf, 0, &a_words);
}

#[test]
fn adjoint_twice_is_bitwise_identity_with_special_values() {
    let ctx = ctx();
    let shapes: [(&[usize], usize, usize); 10] = [
        (&[], 1, 1),
        (&[], 1, 7),
        (&[], 7, 1),
        (&[], 32, 32),
        (&[], 31, 33),
        (&[], 67, 101),
        (&[], 129, 65),
        (&[3], 5, 37),
        (&[2, 3], 41, 7),
        (&[7], 33, 1),
    ];
    for (i, &(batch, m, n)) in shapes.iter().enumerate() {
        check_involution::<f32>(&ctx, batch, m, n, 1000 + i as u64);
        check_involution::<[f32; 2]>(&ctx, batch, m, n, 1100 + i as u64);
    }
}

#[test]
fn adjoint_of_complex_flips_only_the_imaginary_sign_bit() {
    let ctx = ctx();
    let (m, n) = (2, 3);
    let words = [
        [0x0000_0000u32, 0x0000_0000], // +0 + 0i
        [0x8000_0000, 0x8000_0000],    // -0 - 0i
        [0x7fc0_0001, 0x7fc0_0001],    // NaN payload
        [0x7f80_0000, 0xff80_0000],    // +inf - inf i
        [0x3f80_0000, 0x0000_0001],    // 1 + subnormal i
        [0xffc1_2345, 0x7f80_0001],    // negative NaN, signalling NaN
    ];
    let a_buf = alloc_shared::<[f32; 2]>(&ctx, m * n);
    for (k, w) in words.iter().enumerate() {
        words_mut(&a_buf)[2 * k..2 * k + 2].copy_from_slice(w);
    }
    let b_buf = sentinel_buffer::<[f32; 2]>(&ctx, m * n);
    let a = TensorView::<[f32; 2]>::row_major(&a_buf, 0, vec![m, n]);
    let b = TensorView::<[f32; 2]>::row_major(&b_buf, 0, vec![n, m]);
    adjoint(&ctx, &a, &b).unwrap();
    for i in 0..m {
        for j in 0..n {
            let [re, im] = words[i * n + j];
            let at = 2 * (j * m + i);
            assert_eq!(words_mut(&b_buf)[at], re, "re of ({i},{j})");
            assert_eq!(words_mut(&b_buf)[at + 1], im ^ 0x8000_0000, "im of ({i},{j})");
        }
    }
}

#[test]
fn adjoint_of_real_is_plain_transpose() {
    let ctx = ctx();
    for &(m, n) in &[(1usize, 1usize), (3, 5), (37, 41), (64, 33)] {
        let a_buf = alloc_shared::<f32>(&ctx, m * n);
        fill_special(&a_buf, &mut rng(1200));
        let a = TensorView::<f32>::row_major(&a_buf, 0, vec![m, n]);
        let b_buf = sentinel_buffer::<f32>(&ctx, m * n);
        let b = TensorView::<f32>::row_major(&b_buf, 0, vec![n, m]);
        adjoint(&ctx, &a, &b).unwrap();
        let t = permute_view(&a, &[1, 0]).unwrap();
        assert_dst::<f32>("real adjoint", &b_buf, 0, &gather_words(&t, false));
    }
}

#[test]
fn adjoint_batched_strided_source_with_offsets() {
    let ctx = ctx();
    let (batch, m, n, ld) = (3, 37, 41, 48);
    let strides = vec![m * ld, ld, 1];
    let src_offset = 11;
    let a_buf = alloc_shared::<[f32; 2]>(&ctx, span(src_offset, &[batch, m, n], &strides));
    fill_special(&a_buf, &mut rng(1300));
    let a = TensorView::<[f32; 2]>::new(&a_buf, src_offset, vec![batch, m, n], strides);
    let b_buf = sentinel_buffer::<[f32; 2]>(&ctx, 5 + batch * m * n + 9);
    let b = TensorView::<[f32; 2]>::row_major(&b_buf, 5, vec![batch, n, m]);
    adjoint(&ctx, &a, &b).unwrap();
    assert_dst::<[f32; 2]>("strided adjoint", &b_buf, 5, &reference_adjoint(&a));
}

#[test]
fn adjoint_of_a_permuted_view_composes_with_permutation() {
    let ctx = ctx();
    let a_buf = alloc_shared::<[f32; 2]>(&ctx, 5 * 7 * 11);
    fill_special(&a_buf, &mut rng(1400));
    let a = TensorView::<[f32; 2]>::row_major(&a_buf, 0, vec![5, 7, 11]);
    let p = permute_view(&a, &[2, 0, 1]).unwrap(); // [11, 5, 7]
    let b_buf = sentinel_buffer::<[f32; 2]>(&ctx, 5 * 7 * 11);
    let b = TensorView::<[f32; 2]>::row_major(&b_buf, 0, vec![11, 7, 5]);
    adjoint(&ctx, &p, &b).unwrap();
    assert_dst::<[f32; 2]>("adjoint of permuted view", &b_buf, 0, &reference_adjoint(&p));
}

#[test]
fn adjoint_zero_sized_and_degenerate_shapes() {
    let ctx = ctx();
    for (batch, m, n) in
        [(vec![], 0usize, 5usize), (vec![], 5, 0), (vec![2], 0, 0), (vec![0], 3, 4)]
    {
        let (a_shape, b_shape) = adjoint_shape(&batch, m, n);
        let a_buf = alloc_shared::<[f32; 2]>(&ctx, 64);
        let b_buf = sentinel_buffer::<[f32; 2]>(&ctx, 64);
        let a = TensorView::<[f32; 2]>::row_major(&a_buf, 0, a_shape);
        let b = TensorView::<[f32; 2]>::row_major(&b_buf, 0, b_shape);
        adjoint(&ctx, &a, &b).unwrap();
        assert!(words_mut(&b_buf).iter().all(|&w| w == SENTINEL));
    }
}

#[test]
fn conjugate_flips_imaginary_sign_and_is_an_involution() {
    let ctx = ctx();
    for shape in [vec![1usize], vec![97], vec![7, 13], vec![3, 5, 37]] {
        let len: usize = shape.iter().product();
        let a_buf = alloc_shared::<[f32; 2]>(&ctx, len);
        fill_special(&a_buf, &mut rng(1500));
        let a = TensorView::<[f32; 2]>::row_major(&a_buf, 0, shape.clone());
        let b_buf = sentinel_buffer::<[f32; 2]>(&ctx, len);
        let b = TensorView::<[f32; 2]>::row_major(&b_buf, 0, shape.clone());
        let c_buf = sentinel_buffer::<[f32; 2]>(&ctx, len);
        let c = TensorView::<[f32; 2]>::row_major(&c_buf, 0, shape);
        conjugate(&ctx, &a, &b).unwrap();
        assert_dst::<[f32; 2]>("conj", &b_buf, 0, &gather_words(&a, true));
        conjugate(&ctx, &b, &c).unwrap();
        assert_dst::<[f32; 2]>("conj conj", &c_buf, 0, &gather_words(&a, false));
    }
}

#[test]
fn conjugate_of_real_is_a_copy() {
    let ctx = ctx();
    let a_buf = alloc_shared::<f32>(&ctx, 1000);
    fill_special(&a_buf, &mut rng(1600));
    let a = TensorView::<f32>::row_major(&a_buf, 0, vec![10, 100]);
    let b_buf = sentinel_buffer::<f32>(&ctx, 1000);
    let b = TensorView::<f32>::row_major(&b_buf, 0, vec![10, 100]);
    conjugate(&ctx, &a, &b).unwrap();
    assert_dst::<f32>("real conj", &b_buf, 0, &gather_words(&a, false));
}

// ---------------------------------------------------------------------------------------
// Encode-only behaviour
// ---------------------------------------------------------------------------------------

#[test]
fn encode_only_shares_one_command_buffer_and_never_commits() {
    let ctx = ctx();
    let (m, n) = (67, 101);
    let a_buf = alloc_shared::<[f32; 2]>(&ctx, m * n);
    fill_special(&a_buf, &mut rng(1700));
    let a = TensorView::<[f32; 2]>::row_major(&a_buf, 0, vec![m, n]);
    let b_buf = sentinel_buffer::<[f32; 2]>(&ctx, m * n);
    let b = TensorView::<[f32; 2]>::row_major(&b_buf, 0, vec![n, m]);
    let c_buf = sentinel_buffer::<[f32; 2]>(&ctx, m * n);
    let c = TensorView::<[f32; 2]>::row_major(&c_buf, 0, vec![m, n]);
    let d_buf = sentinel_buffer::<[f32; 2]>(&ctx, m * n);
    let d = TensorView::<[f32; 2]>::row_major(&d_buf, 0, vec![m, n]);
    let a_words = gather_words(&a, false);

    let cmd = ctx.queue.new_command_buffer();
    encode_adjoint(&ctx, cmd, &a, &b).unwrap();
    encode_adjoint(&ctx, cmd, &b, &c).unwrap();
    encode_conjugate(&ctx, cmd, &c, &d).unwrap();
    assert_eq!(cmd.status(), metal::MTLCommandBufferStatus::NotEnqueued);
    cmd.commit();
    cmd.wait_until_completed();
    assert_eq!(cmd.status(), metal::MTLCommandBufferStatus::Completed);
    assert_dst::<[f32; 2]>("chained C", &c_buf, 0, &a_words);
    assert_dst::<[f32; 2]>("chained D", &d_buf, 0, &gather_words(&c, true));
}

#[test]
fn encode_materialize_orders_after_earlier_work_in_the_same_command_buffer() {
    let ctx = ctx();
    let (m, n) = (45, 40);
    let a_buf = alloc_shared::<f32>(&ctx, m * n);
    fill_random(&a_buf, &mut rng(1800));
    let a = TensorView::<f32>::row_major(&a_buf, 0, vec![m, n]);
    let t = permute_view(&a, &[1, 0]).unwrap();
    let b_buf = sentinel_buffer::<f32>(&ctx, m * n);
    let b = TensorView::<f32>::row_major(&b_buf, 0, vec![n, m]);
    let c_buf = sentinel_buffer::<f32>(&ctx, m * n);
    let c = TensorView::<f32>::row_major(&c_buf, 0, vec![m, n]);
    let cmd = ctx.queue.new_command_buffer();
    encode_materialize(&ctx, cmd, &t, &b).unwrap();
    let bt = permute_view(&b, &[1, 0]).unwrap();
    encode_materialize(&ctx, cmd, &bt, &c).unwrap();
    cmd.commit();
    cmd.wait_until_completed();
    assert_dst::<f32>("round trip", &c_buf, 0, &gather_words(&a, false));
}

// ---------------------------------------------------------------------------------------
// Error paths
// ---------------------------------------------------------------------------------------

fn assert_untouched(buf: &metal::BufferRef) {
    assert!(words_mut(buf).iter().all(|&w| w == SENTINEL), "destination was modified");
}

#[test]
fn rejects_aliased_destination() {
    let ctx = ctx();
    let buf = alloc_shared::<[f32; 2]>(&ctx, 64 * 64);
    fill_random(&buf, &mut rng(2000));
    let before = words_mut(&buf).to_vec();
    let a = TensorView::<[f32; 2]>::row_major(&buf, 0, vec![32, 32]);
    // Same view in place, and a partially overlapping destination.
    let shifted = TensorView::<[f32; 2]>::row_major(&buf, 100, vec![32, 32]);
    for dst in [&a, &shifted] {
        assert!(matches!(adjoint(&ctx, &a, dst), Err(LinalgError::Overlap(_))));
        assert!(matches!(conjugate(&ctx, &a, dst), Err(LinalgError::Overlap(_))));
        assert!(matches!(materialize(&ctx, &a, dst), Err(LinalgError::Overlap(_))));
    }
    let cmd = ctx.queue.new_command_buffer();
    assert!(matches!(encode_adjoint(&ctx, cmd, &a, &a), Err(LinalgError::Overlap(_))));
    assert_eq!(words_mut(&buf), &before[..], "an aliased call must not launch");
}

#[test]
fn rejects_invalid_metadata_without_launching() {
    let ctx = ctx();
    let src_buf = alloc_shared::<f32>(&ctx, 1024);
    fill_random(&src_buf, &mut rng(2100));
    let dst_buf = sentinel_buffer::<f32>(&ctx, 1024);
    let src = TensorView::<f32>::row_major(&src_buf, 0, vec![16, 32]);
    let invalid = |r: linalg::abi::Result<()>| matches!(r, Err(LinalgError::InvalidArgument(_)));

    // Destination shape mismatch (materialize wants the same shape, adjoint the swapped one).
    let wrong = TensorView::<f32>::row_major(&dst_buf, 0, vec![16, 32]);
    assert!(invalid(adjoint(&ctx, &src, &wrong)));
    let swapped = TensorView::<f32>::row_major(&dst_buf, 0, vec![32, 16]);
    assert!(invalid(materialize(&ctx, &src, &swapped)));
    assert!(invalid(conjugate(&ctx, &src, &swapped)));
    // Destination not contiguous row-major.
    let strided = TensorView::<f32>::new(&dst_buf, 0, vec![32, 16], vec![1, 32]);
    assert!(invalid(adjoint(&ctx, &src, &strided)));
    let padded = TensorView::<f32>::new(&dst_buf, 0, vec![32, 16], vec![20, 1]);
    assert!(invalid(adjoint(&ctx, &src, &padded)));
    // Destination or source outside its buffer (allocations round up to whole granules).
    let capacity = dst_buf.length() as usize / size_of::<f32>();
    let too_far = TensorView::<f32>::row_major(&dst_buf, capacity - 100, vec![32, 16]);
    assert!(invalid(adjoint(&ctx, &src, &too_far)));
    let src_too_far = TensorView::<f32>::row_major(&src_buf, capacity - 100, vec![16, 32]);
    let ok_dst = TensorView::<f32>::row_major(&dst_buf, 0, vec![32, 16]);
    assert!(invalid(adjoint(&ctx, &src_too_far, &ok_dst)));
    // Rank / stride-count mismatches and rank < 2 for the adjoint.
    let bad_rank = TensorView::<f32>::new(&src_buf, 0, vec![16, 32], vec![32]);
    assert!(invalid(adjoint(&ctx, &bad_rank, &ok_dst)));
    let vector = TensorView::<f32>::row_major(&src_buf, 0, vec![16]);
    let vector_dst = TensorView::<f32>::row_major(&dst_buf, 0, vec![16]);
    assert!(invalid(adjoint(&ctx, &vector, &vector_dst)));
    // Source and destination of a mis-sized extent.
    let longer = TensorView::<f32>::row_major(&dst_buf, 0, vec![32, 17]);
    assert!(invalid(adjoint(&ctx, &src, &longer)));
    assert_untouched(&dst_buf);
}

// ---------------------------------------------------------------------------------------
// Benchmarks (run with --ignored under the exclusive GPU lock)
// ---------------------------------------------------------------------------------------

const BENCH_COPY: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void bench_copy(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                       constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) dst[i] = src[i];
}
"#;

/// Best-of-`reps` GB/s for `iters` back-to-back launches per command buffer, counting one
/// read and one write of `bytes`.
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
            best = best.min(secs); // rep 0 is a warm-up
        }
    }
    2.0 * bytes as f64 / best / 1e9
}

#[test]
#[ignore = "benchmark: run under the exclusive GPU lock"]
fn bench_transpose_8191x4099_vs_copy() {
    let ctx = ctx();
    let (m, n) = (8191usize, 4099usize);
    let bytes = m * n * size_of::<f32>();
    let src_buf = alloc_shared::<f32>(&ctx, m * n);
    fill_random(&src_buf, &mut rng(3000));
    let dst_buf = sentinel_buffer::<f32>(&ctx, m * n);
    let src = TensorView::<f32>::row_major(&src_buf, 0, vec![m, n]);
    let dst = TensorView::<f32>::row_major(&dst_buf, 0, vec![n, m]);
    let t = permute_view(&src, &[1, 0]).unwrap();

    let pso = compute_pipeline(
        &ctx,
        "permutation_bench",
        BENCH_COPY,
        metal::CompileOptions::new,
        "bench_copy",
    )
    .unwrap();
    let count = (m * n) as u32;
    let copy_dst = sentinel_buffer::<f32>(&ctx, m * n);
    let copy = |cmd: &metal::CommandBufferRef| {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&src_buf), 0);
        enc.set_buffer(1, Some(&copy_dst), 0);
        enc.set_bytes(2, 4, &count as *const u32 as *const std::ffi::c_void);
        enc.dispatch_threads(
            metal::MTLSize::new(m as u64 * n as u64, 1, 1),
            metal::MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
    };
    let transpose =
        |cmd: &metal::CommandBufferRef| encode_materialize(&ctx, cmd, &t, &dst).unwrap();

    let copy_gbps = best_gbps(&ctx, bytes, 10, 5, copy);
    let transpose_gbps = best_gbps(&ctx, bytes, 10, 5, transpose);
    let ratio = transpose_gbps / copy_gbps;
    println!("copy      {copy_gbps:8.2} GB/s");
    println!("transpose {transpose_gbps:8.2} GB/s  ({:.1}% of copy)", 100.0 * ratio);

    // Spot-check the result of the last run.
    let w = words_mut(&dst_buf);
    let s = words_mut(&src_buf);
    for &(i, j) in &[(0usize, 0usize), (m - 1, n - 1), (m - 1, 0), (0, n - 1), (4100, 2049)] {
        assert_eq!(w[j * m + i], s[i * n + j], "element ({i},{j})");
    }
    assert!(ratio >= 0.70, "transpose reaches only {:.1}% of plain copy", 100.0 * ratio);
}
