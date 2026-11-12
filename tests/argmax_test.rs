//! Correctness tests and benchmarks for the argmax primitive.
//!
//! Every expectation comes from an inline pure-Rust reference. Real input is ranked by
//! `|x|` (NaN above everything, lowest index on ties); complex input uses the same `f32`
//! formulas as the kernel and is additionally checked against an `f64` magnitude within
//! 1 ULP of the true maximum.

// The `objc` selector macros probe a `cargo-clippy` cfg that rustc does not know about.
#![allow(unexpected_cfgs)]

use linalg::abi::{alloc_shared, LinalgError, MetalContext, TensorView};
use linalg::ops::argmax::{
    argmax, argmax_complex, encode_argmax, encode_argmax_complex, Magnitude, MAX_GROUPS,
    SINGLE_PASS_MAX_LEN, THREADGROUP_SIZE,
};
use rand::{Rng, RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

const SENTINEL: u32 = 0xDEAD_BEEF;
const TG: usize = THREADGROUP_SIZE;

// ---------------------------------------------------------------------------------------
// References
// ---------------------------------------------------------------------------------------

/// Reference ordering: does candidate `(ma, ia)` beat `(mb, ib)`?
fn beats(ma: f32, ia: usize, mb: f32, ib: usize) -> bool {
    match (ma.is_nan(), mb.is_nan()) {
        (true, false) => true,
        (false, true) => false,
        (true, true) => ia < ib,
        (false, false) => ma > mb || (ma == mb && ia < ib),
    }
}

/// Index of the best element of a magnitude sequence under the reference ordering.
fn ref_best(mags: impl Iterator<Item = f32>) -> usize {
    let mut best: Option<(f32, usize)> = None;
    for (i, m) in mags.enumerate() {
        if best.is_none_or(|(bm, bi)| beats(m, i, bm, bi)) {
            best = Some((m, i));
        }
    }
    best.expect("reference needs a non-empty input").1
}

fn ref_argmax(x: &[f32]) -> usize {
    ref_best(x.iter().map(|v| v.abs()))
}

/// A complex part as the kernel sees it: parts below the smallest normal count as zero,
/// because Apple GPUs may flush subnormals and the contract must not depend on that.
fn part(v: f32) -> f32 {
    let a = v.abs();
    if a < f32::MIN_POSITIVE {
        0.0
    } else {
        a
    }
}

/// The `f32` complex magnitude formulas shared with the kernel.
fn mag32(z: [f32; 2], kind: Magnitude) -> f32 {
    let (re, im) = (part(z[0]), part(z[1]));
    if re.is_nan() || im.is_nan() {
        return f32::NAN;
    }
    match kind {
        Magnitude::L1 => re + im,
        Magnitude::Hypot => {
            let (hi, lo) = (re.max(im), re.min(im));
            if hi == 0.0 {
                0.0
            } else if hi.is_infinite() {
                f32::INFINITY
            } else {
                let r = lo / hi;
                hi * r.mul_add(r, 1.0).sqrt()
            }
        }
    }
}

fn ref_argmax_complex(x: &[[f32; 2]], kind: Magnitude) -> usize {
    ref_best(x.iter().map(|&z| mag32(z, kind)))
}

/// `f64` magnitude (NaN if either part is NaN).
fn mag64(z: [f32; 2], kind: Magnitude) -> f64 {
    let (re, im) = (f64::from(z[0]).abs(), f64::from(z[1]).abs());
    match kind {
        Magnitude::L1 => re + im,
        Magnitude::Hypot => (re * re + im * im).sqrt(),
    }
}

/// Asserts the selected element's magnitude is within 1 ULP (f32) of the f64 true maximum.
fn assert_within_one_ulp(x: &[[f32; 2]], kind: Magnitude, got: usize) {
    let true_max = x.iter().map(|&z| mag64(z, kind)).fold(f64::NEG_INFINITY, |a, m| {
        if a.is_nan() || m.is_nan() {
            f64::NAN
        } else {
            a.max(m)
        }
    });
    let picked = mag64(x[got], kind);
    if true_max.is_nan() {
        assert!(picked.is_nan(), "a NaN element exists but index {got} is not NaN");
        return;
    }
    let (want, have) = (true_max as f32, picked as f32);
    let distance = i64::from(want.to_bits()).abs_diff(i64::from(have.to_bits()));
    // Parts below the smallest normal are treated as zero, which can shift a magnitude by
    // less than one smallest-normal; that slack is the only allowance beyond 1 ULP.
    let flushed = (f64::from(want) - f64::from(have)).abs() <= f64::from(f32::MIN_POSITIVE);
    assert!(
        distance <= 1 || flushed,
        "magnitude {have:e} is {distance} ULP from true max {want:e}"
    );
}

// ---------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------

fn ctx() -> MetalContext {
    MetalContext::new().expect("Metal device")
}

fn upload<T: Copy>(ctx: &MetalContext, data: &[T]) -> metal::Buffer {
    let buf = alloc_shared::<T>(ctx, data.len());
    // SAFETY: the buffer holds at least `data.len()` elements and is not yet in flight.
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), buf.contents() as *mut T, data.len()) };
    buf
}

fn slot(ctx: &MetalContext) -> metal::Buffer {
    let buf = alloc_shared::<u32>(ctx, 1);
    read_slot_write(&buf, SENTINEL);
    buf
}

fn read_slot_write(buf: &metal::BufferRef, value: u32) {
    // SAFETY: shared buffer with at least one `u32`, no command buffer in flight.
    unsafe { *(buf.contents() as *mut u32) = value };
}

fn read_slot(buf: &metal::BufferRef) -> u32 {
    // SAFETY: as above; callers read after completion.
    unsafe { *(buf.contents() as *const u32) }
}

fn out_view(buf: &metal::BufferRef) -> TensorView<'_, u32> {
    TensorView::<u32>::row_major(buf, 0, vec![1])
}

fn vector<T>(buf: &metal::BufferRef, offset: usize, n: usize, stride: usize) -> TensorView<'_, T> {
    TensorView::<T>::new(buf, offset, vec![n], vec![stride])
}

/// Runs the real argmax on a contiguous slice and returns the index.
fn gpu_argmax(ctx: &MetalContext, x: &[f32]) -> usize {
    let xb = upload(ctx, x);
    let ob = slot(ctx);
    argmax(ctx, &vector::<f32>(&xb, 0, x.len(), 1), &out_view(&ob)).expect("argmax");
    read_slot(&ob) as usize
}

fn gpu_argmax_complex(ctx: &MetalContext, x: &[[f32; 2]], kind: Magnitude) -> usize {
    let xb = upload(ctx, x);
    let ob = slot(ctx);
    argmax_complex(ctx, &vector::<[f32; 2]>(&xb, 0, x.len(), 1), kind, &out_view(&ob))
        .expect("argmax_complex");
    read_slot(&ob) as usize
}

fn check(ctx: &MetalContext, x: &[f32], what: &str) {
    let got = gpu_argmax(ctx, x);
    let want = ref_argmax(x);
    assert_eq!(got, want, "{what}: n={}", x.len());
}

fn check_complex(ctx: &MetalContext, x: &[[f32; 2]], kind: Magnitude, what: &str) {
    let got = gpu_argmax_complex(ctx, x, kind);
    let want = ref_argmax_complex(x, kind);
    assert_eq!(got, want, "{what} {kind:?}: n={}", x.len());
    assert_within_one_ulp(x, kind, got);
}

fn rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

fn random_f32(r: &mut impl Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| r.random_range(-1000.0f32..1000.0)).collect()
}

fn random_c32(r: &mut impl Rng, n: usize) -> Vec<[f32; 2]> {
    (0..n)
        .map(|_| [r.random_range(-1000.0f32..1000.0), r.random_range(-1000.0f32..1000.0)])
        .collect()
}

/// Lengths that straddle every tier boundary.
fn boundary_sizes() -> Vec<usize> {
    let mut sizes = vec![1, 2, 3, 31, 32, 33, 63, 64, 65, 255, 256, 257, 1023, 1024, 1025];
    for base in [TG, 2 * TG, SINGLE_PASS_MAX_LEN, 4 * SINGLE_PASS_MAX_LEN, MAX_GROUPS * TG] {
        sizes.extend([base - 1, base, base + 1]);
    }
    sizes.extend([2 * MAX_GROUPS * TG - 1, 2 * MAX_GROUPS * TG, 3 * MAX_GROUPS * TG + 1]);
    sizes.sort_unstable();
    sizes.dedup();
    sizes
}

// ---------------------------------------------------------------------------------------
// Real input
// ---------------------------------------------------------------------------------------

#[test]
fn small_known_vectors() {
    let c = ctx();
    assert_eq!(gpu_argmax(&c, &[3.0]), 0);
    assert_eq!(gpu_argmax(&c, &[1.0, -5.0, 4.0]), 1);
    assert_eq!(gpu_argmax(&c, &[-1.0, 2.0, -2.0, 2.0]), 1);
    assert_eq!(gpu_argmax(&c, &[0.0, 0.0, 1e-45]), 2);
    assert_eq!(gpu_argmax(&c, &[f32::MAX, -f32::MAX]), 0);
    assert_eq!(gpu_argmax(&c, &[1.0, f32::NEG_INFINITY, f32::INFINITY]), 1);
}

#[test]
fn negative_extrema_beat_positive_values() {
    let c = ctx();
    for n in boundary_sizes() {
        let mut x = vec![4.0f32; n];
        let k = n * 2 / 3;
        x[k] = -5.0;
        check(&c, &x, "negative extremum");
        assert_eq!(gpu_argmax(&c, &x), k);
        x[k] = -3.0;
        assert_eq!(gpu_argmax(&c, &x), 0, "n={n}: -3 must not beat 4");
    }
}

#[test]
fn tie_storm_all_equal_returns_zero() {
    let c = ctx();
    for n in boundary_sizes() {
        assert_eq!(gpu_argmax(&c, &vec![7.0f32; n]), 0, "n={n} all +7");
        assert_eq!(gpu_argmax(&c, &vec![-7.0f32; n]), 0, "n={n} all -7");
        let alt: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 7.0 } else { -7.0 }).collect();
        assert_eq!(gpu_argmax(&c, &alt), 0, "n={n} alternating sign");
        let alt_odd_first: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { -7.0 } else { 7.0 }).collect();
        assert_eq!(gpu_argmax(&c, &alt_odd_first), 0, "n={n} alternating sign, negative first");
    }
}

#[test]
fn tie_storm_planted_maxima_pick_lowest() {
    let c = ctx();
    let mut r = rng(11);
    for n in boundary_sizes().into_iter().filter(|&n| n >= 2) {
        for _ in 0..6 {
            let mut x: Vec<f32> = (0..n).map(|_| r.random_range(-10.0f32..10.0)).collect();
            let copies = r.random_range(2..=8usize.min(n));
            let mut spots: Vec<usize> = (0..copies).map(|_| r.random_range(0..n)).collect();
            spots.sort_unstable();
            spots.dedup();
            for (j, &p) in spots.iter().enumerate() {
                x[p] = if j % 2 == 0 { 50.0 } else { -50.0 };
            }
            assert_eq!(gpu_argmax(&c, &x), spots[0], "n={n} spots={spots:?}");
        }
    }
}

#[test]
fn maximum_at_every_structural_position() {
    let c = ctx();
    let mut r = rng(12);
    for n in boundary_sizes() {
        let mut positions = vec![0, n / 2, n - 1, n.saturating_sub(2)];
        positions.extend([TG - 1, TG, TG + 1, 2 * TG - 1, 2 * TG, SINGLE_PASS_MAX_LEN - 1]);
        positions.extend([SINGLE_PASS_MAX_LEN, MAX_GROUPS * TG - 1, MAX_GROUPS * TG]);
        for p in positions.into_iter().filter(|&p| p < n) {
            let mut x: Vec<f32> = (0..n).map(|_| r.random_range(-1.0f32..1.0)).collect();
            x[p] = if p % 2 == 0 { 9.5 } else { -9.5 };
            assert_eq!(gpu_argmax(&c, &x), p, "n={n} planted at {p}");
        }
    }
}

#[test]
fn random_inputs_match_reference_across_sizes() {
    let c = ctx();
    let mut r = rng(13);
    for n in boundary_sizes() {
        let x = random_f32(&mut r, n);
        check(&c, &x, "random");
    }
}

#[test]
fn ten_million_elements() {
    let c = ctx();
    let mut r = rng(14);
    let n = 10_000_000;
    let mut x = random_f32(&mut r, n);
    check(&c, &x, "random 1e7");
    x.iter_mut().for_each(|v| *v /= 1000.0);
    x[n - 1] = 10.0;
    x[n / 2] = -10.0;
    x[777] = -10.0;
    assert_eq!(gpu_argmax(&c, &x), 777, "tie storm at 1e7");
}

#[test]
fn signed_zero_ties_to_lowest_index() {
    let c = ctx();
    assert_eq!(gpu_argmax(&c, &[-0.0, 0.0]), 0);
    assert_eq!(gpu_argmax(&c, &[0.0, -0.0]), 0);
    for n in boundary_sizes() {
        let x: Vec<f32> = (0..n).map(|i| if i % 3 == 0 { -0.0 } else { 0.0 }).collect();
        assert_eq!(gpu_argmax(&c, &x), 0, "n={n} mixed zeros");
        let mut y = x.clone();
        y[n - 1] = f32::from_bits(1); // smallest subnormal beats ±0
        assert_eq!(gpu_argmax(&c, &y), n - 1, "n={n} subnormal beats zero");
    }
}

#[test]
fn nan_first_middle_last_and_ranks_above_infinity() {
    let c = ctx();
    for n in boundary_sizes() {
        for nan in [f32::NAN, -f32::NAN, f32::from_bits(0x7fc0_1234), f32::from_bits(0xffa0_0001)] {
            for p in [0, n / 2, n - 1] {
                let mut x = vec![1.0f32; n];
                x[n / 3] = f32::INFINITY;
                x[p] = nan;
                assert_eq!(gpu_argmax(&c, &x), p, "n={n} nan at {p}");
            }
        }
    }
}

#[test]
fn lowest_index_nan_wins() {
    let c = ctx();
    let mut r = rng(15);
    for n in boundary_sizes().into_iter().filter(|&n| n >= 3) {
        for _ in 0..5 {
            let mut x = random_f32(&mut r, n);
            let mut spots: Vec<usize> =
                (0..r.random_range(2..=6usize)).map(|_| r.random_range(0..n)).collect();
            spots.sort_unstable();
            spots.dedup();
            for &p in &spots {
                x[p] = if p % 2 == 0 { f32::NAN } else { -f32::NAN };
            }
            x[n - 1] = if spots.contains(&(n - 1)) { x[n - 1] } else { f32::INFINITY };
            assert_eq!(gpu_argmax(&c, &x), spots[0], "n={n} nan spots={spots:?}");
        }
    }
}

#[test]
fn infinities_tie_by_index() {
    let c = ctx();
    for n in boundary_sizes().into_iter().filter(|&n| n >= 4) {
        let mut x = vec![2.0f32; n];
        x[n - 1] = f32::NEG_INFINITY;
        x[n - 2] = f32::INFINITY;
        assert_eq!(gpu_argmax(&c, &x), n - 2, "n={n}");
        x[n / 2] = f32::NEG_INFINITY;
        assert_eq!(gpu_argmax(&c, &x), n / 2, "n={n} earlier -inf");
    }
}

#[test]
fn results_are_bitwise_deterministic() {
    let c = ctx();
    let mut r = rng(16);
    let x: Vec<f32> = (0..300_000).map(|_| f32::from(r.random_range(-3i8..=3))).collect();
    let first = gpu_argmax(&c, &x);
    assert_eq!(first, ref_argmax(&x));
    for _ in 0..25 {
        assert_eq!(gpu_argmax(&c, &x), first);
    }
}

// ---------------------------------------------------------------------------------------
// Strided views
// ---------------------------------------------------------------------------------------

#[test]
fn row_and_column_scans_of_a_1000_square_matrix_with_offsets() {
    let c = ctx();
    let mut r = rng(17);
    let (rows, cols, base) = (1000usize, 1000usize, 37usize);
    let mut m = vec![0.0f32; base + rows * cols + 5];
    for v in &mut m[base..base + rows * cols] {
        *v = r.random_range(-100.0f32..100.0);
    }
    let mb = upload(&c, &m);
    let at = |i: usize, j: usize| m[base + i * cols + j];
    for i in (0..rows).step_by(7).chain([rows - 1]) {
        let ob = slot(&c);
        let row = vector::<f32>(&mb, base + i * cols, cols, 1);
        argmax(&c, &row, &out_view(&ob)).unwrap();
        let want = ref_best((0..cols).map(|j| at(i, j).abs()));
        assert_eq!(read_slot(&ob) as usize, want, "row {i}");
    }
    for j in (0..cols).step_by(7).chain([cols - 1]) {
        let ob = slot(&c);
        let col = vector::<f32>(&mb, base + j, rows, cols);
        argmax(&c, &col, &out_view(&ob)).unwrap();
        let want = ref_best((0..rows).map(|i| at(i, j).abs()));
        assert_eq!(read_slot(&ob) as usize, want, "column {j}");
    }
    // Trailing sub-column starting below the diagonal, as an LU pivot search would.
    let ob = slot(&c);
    let k = 313;
    let tail = vector::<f32>(&mb, base + k * cols + k, rows - k, cols);
    argmax(&c, &tail, &out_view(&ob)).unwrap();
    assert_eq!(read_slot(&ob) as usize, ref_best((k..rows).map(|i| at(i, k).abs())));
}

#[test]
fn complex_row_and_column_scans_with_offsets() {
    let c = ctx();
    let mut r = rng(18);
    let (n, base) = (1000usize, 11usize);
    let mut m = vec![[0.0f32; 2]; base + n * n];
    for z in &mut m[base..] {
        *z = [r.random_range(-50.0f32..50.0), r.random_range(-50.0f32..50.0)];
    }
    let mb = upload(&c, &m);
    for kind in [Magnitude::L1, Magnitude::Hypot] {
        for line in [0usize, 1, 499, 998, 999] {
            let ob = slot(&c);
            argmax_complex(
                &c,
                &vector::<[f32; 2]>(&mb, base + line * n, n, 1),
                kind,
                &out_view(&ob),
            )
            .unwrap();
            let row: Vec<[f32; 2]> = (0..n).map(|j| m[base + line * n + j]).collect();
            assert_eq!(read_slot(&ob) as usize, ref_argmax_complex(&row, kind), "row {line}");
            assert_within_one_ulp(&row, kind, read_slot(&ob) as usize);

            argmax_complex(&c, &vector::<[f32; 2]>(&mb, base + line, n, n), kind, &out_view(&ob))
                .unwrap();
            let col: Vec<[f32; 2]> = (0..n).map(|i| m[base + i * n + line]).collect();
            assert_eq!(read_slot(&ob) as usize, ref_argmax_complex(&col, kind), "col {line}");
            assert_within_one_ulp(&col, kind, read_slot(&ob) as usize);
        }
    }
}

#[test]
fn large_strides_and_stride_zero() {
    let c = ctx();
    let mut r = rng(19);
    let n = 50_000usize;
    for stride in [2usize, 3, 17, 33] {
        let len = (n - 1) * stride + 1 + 4;
        let buf: Vec<f32> = random_f32(&mut r, len);
        let bb = upload(&c, &buf);
        let ob = slot(&c);
        argmax(&c, &vector::<f32>(&bb, 4, n, stride), &out_view(&ob)).unwrap();
        let want = ref_best((0..n).map(|i| buf[4 + i * stride].abs()));
        assert_eq!(read_slot(&ob) as usize, want, "stride {stride}");
    }
    let bb = upload(&c, &[-3.0f32, 9.0]);
    let ob = slot(&c);
    argmax(&c, &vector::<f32>(&bb, 1, 100_000, 0), &out_view(&ob)).unwrap();
    assert_eq!(read_slot(&ob), 0, "stride 0 is one value repeated");
}

#[test]
fn rank_zero_and_length_one_outputs_are_accepted() {
    let c = ctx();
    let xb = upload(&c, &[1.0f32, -4.0, 2.0]);
    let ob = slot(&c);
    let scalar = TensorView::<u32>::new(&ob, 0, vec![], vec![]);
    argmax(&c, &vector::<f32>(&xb, 0, 3, 1), &scalar).unwrap();
    assert_eq!(read_slot(&ob), 1);
    // Output slot at a nonzero element offset inside a larger buffer.
    let wide = upload(&c, &[SENTINEL; 8]);
    let at5 = TensorView::<u32>::row_major(&wide, 5, vec![1]);
    argmax(&c, &vector::<f32>(&xb, 0, 3, 1), &at5).unwrap();
    // SAFETY: completed; 8 elements were uploaded.
    let words = unsafe { std::slice::from_raw_parts(wide.contents() as *const u32, 8) };
    assert_eq!(words[5], 1);
    assert!(words.iter().enumerate().all(|(i, &w)| i == 5 || w == SENTINEL), "stray write");
}

// ---------------------------------------------------------------------------------------
// Complex input
// ---------------------------------------------------------------------------------------

#[test]
fn complex_random_matches_reference_in_both_magnitudes() {
    let c = ctx();
    let mut r = rng(20);
    for kind in [Magnitude::L1, Magnitude::Hypot] {
        for n in boundary_sizes() {
            let x = random_c32(&mut r, n);
            check_complex(&c, &x, kind, "random");
        }
    }
}

#[test]
fn complex_ten_million_elements() {
    let c = ctx();
    let mut r = rng(21);
    let x = random_c32(&mut r, 10_000_000);
    for kind in [Magnitude::L1, Magnitude::Hypot] {
        check_complex(&c, &x, kind, "random 1e7");
    }
}

#[test]
fn complex_magnitude_kinds_disagree_where_they_should() {
    let c = ctx();
    // |3|+|3| = 6 beats |5|+|0| = 5 in L1, but hypot(3,3) = 4.24 loses to hypot(5,0) = 5.
    let x = [[5.0f32, 0.0], [-3.0, 3.0]];
    assert_eq!(gpu_argmax_complex(&c, &x, Magnitude::L1), 1);
    assert_eq!(gpu_argmax_complex(&c, &x, Magnitude::Hypot), 0);
}

#[test]
fn complex_ties_nans_and_signed_zero() {
    let c = ctx();
    for kind in [Magnitude::L1, Magnitude::Hypot] {
        for n in boundary_sizes() {
            let ties = vec![[3.0f32, -4.0]; n];
            check_complex(&c, &ties, kind, "all equal");
            assert_eq!(gpu_argmax_complex(&c, &ties, kind), 0);
            let zeros: Vec<[f32; 2]> = (0..n)
                .map(|i| match i % 4 {
                    0 => [-0.0, 0.0],
                    1 => [0.0, -0.0],
                    2 => [0.0, 0.0],
                    _ => [-0.0, -0.0],
                })
                .collect();
            assert_eq!(gpu_argmax_complex(&c, &zeros, kind), 0, "n={n} zeros");
            // Swapped parts have equal magnitude: the lower index wins.
            let swapped: Vec<[f32; 2]> =
                (0..n).map(|i| if i % 2 == 0 { [1.0, 6.0] } else { [-6.0, -1.0] }).collect();
            assert_eq!(gpu_argmax_complex(&c, &swapped, kind), 0, "n={n} swapped");
            for p in [0, n / 2, n - 1] {
                for bad in [
                    [f32::NAN, 1.0],
                    [1.0, f32::NAN],
                    [f32::NAN, f32::NAN],
                    [f32::INFINITY, f32::NAN],
                ] {
                    let mut x = vec![[1.0f32, 1.0]; n];
                    x[n / 3] = [f32::INFINITY, 0.0];
                    x[p] = bad;
                    assert_eq!(gpu_argmax_complex(&c, &x, kind), p, "{kind:?} n={n} nan at {p}");
                }
            }
            if n >= 3 {
                let mut x = vec![[1.0f32, 1.0]; n];
                x[n - 1] = [0.0, f32::NAN];
                x[n - 2] = [f32::NAN, 0.0];
                assert_eq!(gpu_argmax_complex(&c, &x, kind), n - 2, "lowest NaN wins");
            }
        }
    }
}

#[test]
fn complex_magnitudes_near_f32_max_do_not_overflow_in_hypot() {
    let c = ctx();
    let m = f32::MAX;
    for n in [2usize, 33, TG + 1, SINGLE_PASS_MAX_LEN + 1, 5 * SINGLE_PASS_MAX_LEN + 3] {
        // Naive sqrt(re² + im²) overflows for all of these; max-scaling must not.
        let mut x = vec![[0.6 * m, 0.6 * m]; n];
        x[n - 1] = [0.9 * m, 0.3 * m];
        check_complex(&c, &x, Magnitude::Hypot, "near MAX hypot");
        assert_eq!(gpu_argmax_complex(&c, &x, Magnitude::Hypot), n - 1, "n={n} hypot");
        // L1 stays finite while the sum fits.
        let mut y = vec![[0.3 * m, 0.3 * m]; n];
        y[n / 2] = [0.5 * m, 0.4 * m];
        check_complex(&c, &y, Magnitude::L1, "near MAX l1");
        assert_eq!(gpu_argmax_complex(&c, &y, Magnitude::L1), n / 2, "n={n} l1");
    }
}

#[test]
fn complex_l1_overflow_saturates_and_ties_to_lowest_index() {
    let c = ctx();
    let m = f32::MAX;
    let mut x = vec![[0.9 * m, 0.9 * m]; 4000];
    x[10] = [m, m];
    assert_eq!(gpu_argmax_complex(&c, &x, Magnitude::L1), 0);
    check_complex(&c, &x, Magnitude::L1, "saturated l1");
}

#[test]
fn complex_subnormal_and_mixed_scale_parts() {
    let c = ctx();
    let tiny = f32::from_bits(3);
    let x = vec![[tiny, 0.0], [0.0, 0.0], [0.0, tiny], [f32::from_bits(4), 0.0], [0.0, 0.0]];
    let mut y = x.clone();
    y[3] = [f32::MIN_POSITIVE, 0.0];
    for kind in [Magnitude::L1, Magnitude::Hypot] {
        // Subnormal parts count as zero, so every element ties at 0 until a normal one appears.
        check_complex(&c, &x, kind, "subnormals");
        assert_eq!(gpu_argmax_complex(&c, &x, kind), 0, "{kind:?} subnormals are zero");
        check_complex(&c, &y, kind, "smallest normal");
        assert_eq!(gpu_argmax_complex(&c, &y, kind), 3, "{kind:?} smallest normal beats zero");
    }
    let mut r = rng(22);
    let wide: Vec<[f32; 2]> = (0..20_000)
        .map(|_| {
            let e = |r: &mut ChaCha8Rng| 2f32.powi(r.random_range(-60..60));
            [e(&mut r) * r.random_range(-1.0f32..1.0), e(&mut r) * r.random_range(-1.0f32..1.0)]
        })
        .collect();
    for kind in [Magnitude::L1, Magnitude::Hypot] {
        check_complex(&c, &wide, kind, "mixed scales");
    }
}

// ---------------------------------------------------------------------------------------
// Invalid metadata
// ---------------------------------------------------------------------------------------

fn assert_invalid(result: linalg::abi::Result<()>, ob: &metal::BufferRef, what: &str) {
    assert!(matches!(result, Err(LinalgError::InvalidArgument(_))), "{what}: got {result:?}");
    assert_eq!(read_slot(ob), SENTINEL, "{what}: output must stay untouched");
}

#[test]
fn empty_and_oversized_inputs_are_errors() {
    let c = ctx();
    let xb = upload(&c, &[1.0f32; 16]);
    let ob = slot(&c);
    assert_invalid(argmax(&c, &vector::<f32>(&xb, 0, 0, 1), &out_view(&ob)), &ob, "n = 0");
    assert_invalid(
        argmax(&c, &vector::<f32>(&xb, 16, 0, 1), &out_view(&ob)),
        &ob,
        "n = 0 at end of buffer",
    );
    // Stride 0 keeps the view in bounds, so only the length limit can reject it.
    let too_long = u32::MAX as usize + 1;
    assert_invalid(
        argmax(&c, &vector::<f32>(&xb, 0, too_long, 0), &out_view(&ob)),
        &ob,
        "n > u32::MAX",
    );
    let zb = upload(&c, &[[1.0f32, 1.0]; 4]);
    assert_invalid(
        argmax_complex(&c, &vector::<[f32; 2]>(&zb, 0, 0, 1), Magnitude::L1, &out_view(&ob)),
        &ob,
        "complex n = 0",
    );
    assert_invalid(
        argmax_complex(
            &c,
            &vector::<[f32; 2]>(&zb, 0, too_long, 0),
            Magnitude::Hypot,
            &out_view(&ob),
        ),
        &ob,
        "complex n > u32::MAX",
    );
}

#[test]
fn malformed_views_are_errors_and_never_launch() {
    let c = ctx();
    let xb = upload(&c, &[1.0f32; 64]);
    let ob = slot(&c);
    let cap = xb.length() as usize / 4;
    assert_invalid(argmax(&c, &vector::<f32>(&xb, 0, cap + 1, 1), &out_view(&ob)), &ob, "past end");
    assert_invalid(
        argmax(&c, &vector::<f32>(&xb, cap, 1, 1), &out_view(&ob)),
        &ob,
        "offset at end",
    );
    assert_invalid(
        argmax(&c, &vector::<f32>(&xb, 0, 1 << 40, usize::MAX), &out_view(&ob)),
        &ob,
        "overflow",
    );
    let rank2 = TensorView::<f32>::row_major(&xb, 0, vec![4, 4]);
    assert_invalid(argmax(&c, &rank2, &out_view(&ob)), &ob, "rank 2 input");
    let mismatched = TensorView::<f32>::new(&xb, 0, vec![8], vec![1, 1]);
    assert_invalid(argmax(&c, &mismatched, &out_view(&ob)), &ob, "shape/stride rank mismatch");

    let x = vector::<f32>(&xb, 0, 8, 1);
    let wide_out = alloc_shared::<u32>(&c, 4);
    read_slot_write(&wide_out, SENTINEL);
    let two = TensorView::<u32>::row_major(&wide_out, 0, vec![2]);
    assert_invalid(argmax(&c, &x, &two), &wide_out, "output of two slots");
    let empty_out = TensorView::<u32>::new(&wide_out, 0, vec![0], vec![1]);
    assert_invalid(argmax(&c, &x, &empty_out), &wide_out, "empty output");
    let past = TensorView::<u32>::row_major(&wide_out, wide_out.length() as usize / 4, vec![1]);
    assert_invalid(argmax(&c, &x, &past), &wide_out, "output past end");
}

#[test]
fn output_overlapping_input_is_rejected() {
    let c = ctx();
    let both = alloc_shared::<f32>(&c, 64);
    read_slot_write(&both, SENTINEL);
    // SAFETY: fresh buffer, 64 elements.
    let words = unsafe { std::slice::from_raw_parts_mut(both.contents() as *mut f32, 64) };
    words[4..].fill(1.0);
    let x = vector::<f32>(&both, 4, 20, 1);
    let alias = TensorView::<u32>::row_major(&both, 10, vec![1]);
    let result = argmax(&c, &x, &alias);
    assert!(matches!(result, Err(LinalgError::Overlap(_))), "got {result:?}");
    let disjoint = TensorView::<u32>::row_major(&both, 30, vec![1]);
    argmax(&c, &x, &disjoint).unwrap();
}

#[test]
fn failed_encode_leaves_command_buffer_usable() {
    let c = ctx();
    let xb = upload(&c, &[2.0f32, -8.0, 1.0]);
    let ob = slot(&c);
    let cmd = c.queue.new_command_buffer();
    let bad = encode_argmax(&c, cmd, &vector::<f32>(&xb, 0, 0, 1), &out_view(&ob));
    assert!(matches!(bad, Err(LinalgError::InvalidArgument(_))));
    encode_argmax(&c, cmd, &vector::<f32>(&xb, 0, 3, 1), &out_view(&ob)).unwrap();
    cmd.commit();
    cmd.wait_until_completed();
    assert_eq!(read_slot(&ob), 1);
}

// ---------------------------------------------------------------------------------------
// Encode-only API, shared scratch, threads
// ---------------------------------------------------------------------------------------

#[test]
fn many_two_pass_calls_share_one_command_buffer() {
    let c = ctx();
    let mut r = rng(23);
    let calls = 64usize;
    let sizes = [SINGLE_PASS_MAX_LEN + 1, 100_000, 3 * MAX_GROUPS * TG + 7, 20];
    let inputs: Vec<Vec<f32>> =
        (0..calls).map(|i| random_f32(&mut r, sizes[i % sizes.len()])).collect();
    let buffers: Vec<metal::Buffer> = inputs.iter().map(|x| upload(&c, x)).collect();
    let outs = alloc_shared::<u32>(&c, calls);
    read_slot_write(&outs, SENTINEL);
    let cmd = c.queue.new_command_buffer();
    for (i, (x, b)) in inputs.iter().zip(&buffers).enumerate() {
        let out = TensorView::<u32>::row_major(&outs, i, vec![1]);
        encode_argmax(&c, cmd, &vector::<f32>(b, 0, x.len(), 1), &out).unwrap();
    }
    let zin = upload(&c, &random_c32(&mut r, 50_000));
    let zout = slot(&c);
    encode_argmax_complex(
        &c,
        cmd,
        &vector::<[f32; 2]>(&zin, 0, 50_000, 1),
        Magnitude::Hypot,
        &out_view(&zout),
    )
    .unwrap();
    cmd.commit();
    cmd.wait_until_completed();
    // SAFETY: completed; `calls` outputs were allocated.
    let got = unsafe { std::slice::from_raw_parts(outs.contents() as *const u32, calls) };
    for (i, x) in inputs.iter().enumerate() {
        assert_eq!(got[i] as usize, ref_argmax(x), "call {i} n={}", x.len());
    }
    // SAFETY: completed; 50_000 complex elements were uploaded.
    let z = unsafe { std::slice::from_raw_parts(zin.contents() as *const [f32; 2], 50_000) };
    assert_eq!(read_slot(&zout) as usize, ref_argmax_complex(z, Magnitude::Hypot));
}

#[test]
fn a_matrix_of_scans_in_one_command_buffer() {
    let c = ctx();
    let mut r = rng(24);
    let n = 1000usize;
    let m = random_f32(&mut r, n * n);
    let mb = upload(&c, &m);
    let outs = alloc_shared::<u32>(&c, 2 * n);
    let cmd = c.queue.new_command_buffer();
    for i in 0..n {
        let row_out = TensorView::<u32>::row_major(&outs, i, vec![1]);
        encode_argmax(&c, cmd, &vector::<f32>(&mb, i * n, n, 1), &row_out).unwrap();
        let col_out = TensorView::<u32>::row_major(&outs, n + i, vec![1]);
        encode_argmax(&c, cmd, &vector::<f32>(&mb, i, n, n), &col_out).unwrap();
    }
    cmd.commit();
    cmd.wait_until_completed();
    // SAFETY: completed; `2n` outputs were allocated.
    let got = unsafe { std::slice::from_raw_parts(outs.contents() as *const u32, 2 * n) };
    for i in 0..n {
        assert_eq!(got[i] as usize, ref_best((0..n).map(|j| m[i * n + j].abs())), "row {i}");
        assert_eq!(got[n + i] as usize, ref_best((0..n).map(|j| m[j * n + i].abs())), "col {i}");
    }
}

#[test]
fn output_feeds_a_later_dispatch_in_the_same_command_buffer() {
    // The result must be visible to GPU work encoded after it in the same command buffer.
    const READ_BACK: &str = r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void read_back(device const float* x [[buffer(0)]],
                              device const uint* k [[buffer(1)]],
                              device float* y [[buffer(2)]]) { y[0] = x[k[0]]; }
    "#;
    let c = ctx();
    let mut r = rng(25);
    let x = random_f32(&mut r, 200_000);
    let xb = upload(&c, &x);
    let ob = slot(&c);
    let yb = alloc_shared::<f32>(&c, 1);
    let pso = linalg::abi::compute_pipeline(
        &c,
        "argmax_test",
        READ_BACK,
        metal::CompileOptions::new,
        "read_back",
    )
    .unwrap();
    let cmd = c.queue.new_command_buffer();
    encode_argmax(&c, cmd, &vector::<f32>(&xb, 0, x.len(), 1), &out_view(&ob)).unwrap();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pso);
    enc.set_buffer(0, Some(&xb), 0);
    enc.set_buffer(1, Some(&ob), 0);
    enc.set_buffer(2, Some(&yb), 0);
    enc.dispatch_threads(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(1, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    let want = x[ref_argmax(&x)];
    // SAFETY: completed.
    assert_eq!(unsafe { *(yb.contents() as *const f32) }, want);
}

#[test]
fn concurrent_threads_encode_independently() {
    let c = ctx();
    std::thread::scope(|s| {
        for t in 0..6u64 {
            let c = &c;
            s.spawn(move || {
                let mut r = rng(100 + t);
                for round in 0..12 {
                    let n = [5_000usize, 70_000, 400_000][round % 3];
                    let x = random_f32(&mut r, n);
                    let xb = upload(c, &x);
                    let outs = alloc_shared::<u32>(c, 4);
                    let cmd = c.queue.new_command_buffer();
                    for k in 0..4 {
                        let out = TensorView::<u32>::row_major(&outs, k, vec![1]);
                        encode_argmax(c, cmd, &vector::<f32>(&xb, 0, n, 1), &out).unwrap();
                    }
                    cmd.commit();
                    cmd.wait_until_completed();
                    // SAFETY: completed; four outputs.
                    let got =
                        unsafe { std::slice::from_raw_parts(outs.contents() as *const u32, 4) };
                    assert!(
                        got.iter().all(|&g| g as usize == ref_argmax(&x)),
                        "thread {t} round {round}"
                    );
                }
            });
        }
    });
}

// ---------------------------------------------------------------------------------------
// Fuzz against the reference with a small value alphabet (ties, zeros, NaN, infinities)
// ---------------------------------------------------------------------------------------

#[test]
fn seeded_fuzz_with_adversarial_alphabet() {
    let c = ctx();
    let mut r = rng(26);
    let alphabet = [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        2.0,
        -2.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        f32::from_bits(1),
    ];
    for round in 0..160 {
        let n = match round % 4 {
            0 => r.random_range(1..70usize),
            1 => r.random_range(1..3 * TG),
            2 => r.random_range(SINGLE_PASS_MAX_LEN - 50..SINGLE_PASS_MAX_LEN + 50),
            _ => r.random_range(1..400_000usize),
        };
        let nan_odds = if round % 5 == 0 { 40 } else { 0 };
        let x: Vec<f32> = (0..n)
            .map(|_| {
                if nan_odds != 0 && r.random_range(0..nan_odds * 100) == 0 {
                    f32::NAN
                } else {
                    alphabet[r.random_range(0..alphabet.len())]
                }
            })
            .collect();
        check(&c, &x, "fuzz");
    }
}

// ---------------------------------------------------------------------------------------
// Benchmarks (run with `--ignored --nocapture --test-threads=1` under the exclusive GPU lock)
// ---------------------------------------------------------------------------------------

/// Pure streaming-read baseline: each thread folds a strided slice with `max` and writes one
/// partial, so it measures what a bandwidth-bound reduction can reach on this GPU.
const BANDWIDTH_BASELINE: &str = r#"
    #include <metal_stdlib>
    using namespace metal;
    kernel void read_max(device const float4* x [[buffer(0)]],
                         device float* partial [[buffer(1)]],
                         constant uint& n4 [[buffer(2)]],
                         uint gid [[thread_position_in_grid]],
                         uint threads [[threads_per_grid]]) {
        float4 acc = float4(0.0f);
        for (uint i = gid; i < n4; i += threads) acc = max(acc, abs(x[i]));
        partial[gid] = max(max(acc.x, acc.y), max(acc.z, acc.w));
    }
"#;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// Wall-clock microseconds per synchronous call (encode + commit + wait), median of `iters`.
fn time_sync_us(iters: usize, mut call: impl FnMut()) -> f64 {
    for _ in 0..5 {
        call();
    }
    median(
        (0..iters)
            .map(|_| {
                let t = std::time::Instant::now();
                call();
                t.elapsed().as_secs_f64() * 1e6
            })
            .collect(),
    )
}

/// GPU-side microseconds of a command buffer (`GPUEndTime - GPUStartTime`).
fn gpu_us(cmd: &metal::CommandBufferRef) -> f64 {
    use metal::objc::{msg_send, sel, sel_impl};
    // SAFETY: `GPUStartTime`/`GPUEndTime` are CFTimeInterval (f64) getters on a completed buffer.
    let (start, end): (f64, f64) =
        unsafe { (msg_send![cmd, GPUStartTime], msg_send![cmd, GPUEndTime]) };
    (end - start) * 1e6
}

#[test]
#[ignore = "benchmark: run under the exclusive GPU lock"]
fn bench_latency_and_bandwidth() {
    let c = ctx();
    let mut r = rng(30);
    println!("\nargmax f32 (sync = encode+commit+wait wall time, gpu = GPUEnd-GPUStart), median");
    println!(
        "{:>10} {:>12} {:>12} {:>12} {:>10}",
        "N", "sync us", "gpu us", "base gpu us", "GB/s(gpu)"
    );
    let pso = linalg::abi::compute_pipeline(
        &c,
        "argmax_bench",
        BANDWIDTH_BASELINE,
        metal::CompileOptions::new,
        "read_max",
    )
    .unwrap();
    for n in [64usize, 1024, 100_000, 10_000_000] {
        let x = random_f32(&mut r, n);
        let xb = upload(&c, &x);
        let ob = slot(&c);
        let view = vector::<f32>(&xb, 0, n, 1);
        let sync = time_sync_us(300, || argmax(&c, &view, &out_view(&ob)).unwrap());
        let gpu = median(
            (0..200)
                .map(|_| {
                    let cmd = c.queue.new_command_buffer();
                    encode_argmax(&c, cmd, &view, &out_view(&ob)).unwrap();
                    cmd.commit();
                    cmd.wait_until_completed();
                    gpu_us(cmd)
                })
                .collect(),
        );
        // Baseline: same bytes, 1024 groups x 256 threads, no tie logic and no second pass.
        let n4 = (n / 4) as u32;
        let threads = (n4 as usize).clamp(1, 1024 * 256);
        let partials = alloc_shared::<f32>(&c, threads);
        let base = median(
            (0..200)
                .map(|_| {
                    let cmd = c.queue.new_command_buffer();
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&pso);
                    enc.set_buffer(0, Some(&xb), 0);
                    enc.set_buffer(1, Some(&partials), 0);
                    enc.set_bytes(2, 4, (&n4 as *const u32).cast());
                    enc.dispatch_threads(
                        metal::MTLSize::new(threads as u64, 1, 1),
                        metal::MTLSize::new(256.min(threads) as u64, 1, 1),
                    );
                    enc.end_encoding();
                    cmd.commit();
                    cmd.wait_until_completed();
                    gpu_us(cmd)
                })
                .collect(),
        );
        let gbs = (n * 4) as f64 / (gpu * 1e-6) / 1e9;
        println!("{n:>10} {sync:>12.1} {gpu:>12.1} {base:>12.1} {gbs:>10.1}");
    }
}

#[test]
#[ignore = "benchmark: run under the exclusive GPU lock"]
fn bench_complex_and_strided_1e7() {
    let c = ctx();
    let mut r = rng(31);
    let n = 10_000_000usize;
    let z = random_c32(&mut r, n);
    let zb = upload(&c, &z);
    let ob = slot(&c);
    println!("\ncomplex n=1e7 ({} MB): gpu us / GB/s", n * 8 / 1_000_000);
    for kind in [Magnitude::L1, Magnitude::Hypot] {
        let view = vector::<[f32; 2]>(&zb, 0, n, 1);
        let gpu = median(
            (0..100)
                .map(|_| {
                    let cmd = c.queue.new_command_buffer();
                    encode_argmax_complex(&c, cmd, &view, kind, &out_view(&ob)).unwrap();
                    cmd.commit();
                    cmd.wait_until_completed();
                    gpu_us(cmd)
                })
                .collect(),
        );
        println!("{kind:?}: {gpu:.1} us, {:.1} GB/s", (n * 8) as f64 / (gpu * 1e-6) / 1e9);
    }
    let x = random_f32(&mut r, n);
    let xb = upload(&c, &x);
    for stride in [1usize, 1000] {
        let m = n / stride;
        let view = vector::<f32>(&xb, 0, m, stride);
        let gpu = median(
            (0..100)
                .map(|_| {
                    let cmd = c.queue.new_command_buffer();
                    encode_argmax(&c, cmd, &view, &out_view(&ob)).unwrap();
                    cmd.commit();
                    cmd.wait_until_completed();
                    gpu_us(cmd)
                })
                .collect(),
        );
        println!(
            "real stride {stride:>4} n={m}: {gpu:.1} us (useful {:.1} GB/s)",
            (m * 4) as f64 / (gpu * 1e-6) / 1e9
        );
    }
}
