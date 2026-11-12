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

/// Sign-stripped bits of `+inf`, of the smallest normal, and the fraction-field mask.
const INF_BITS: u32 = 0x7f80_0000;
const MIN_NORMAL_BITS: u32 = 0x0080_0000;
const FRACTION_MASK: u32 = 0x007f_ffff;

/// Mirror of `kHypotMaxSteps` in `shaders/argmax.metal`.
const HYPOT_MAX_STEPS: u32 = 6;

/// Mirror of `hypot_round_exact` in `shaders/argmax.metal`: steps the candidate `est` (float
/// bits in `[1, 4)`) one ULP at a time until exact 64-bit midpoint tests accept it.
fn hypot_round_exact(mut est: u32, sig_a: u32, sig_b: u32, gap: u32) -> u32 {
    let x = u64::from(sig_a) << (gap + 1);
    let y = u64::from(sig_b) << 1;
    let sum = x.wrapping_mul(x).wrapping_add(y.wrapping_mul(y));
    for _ in 0..HYPOT_MAX_STEPS {
        let q = MIN_NORMAL_BITS | (est & FRACTION_MASK);
        let s = (est >> 23) - 126 + gap;
        let big = u64::from(q) << s;
        let r4 = (sum.wrapping_sub(big.wrapping_mul(big)) << 2) as i64;
        let odd = q & 1 == 1;
        let above = r4 - (big << (s + 2)) as i64 - (1i64 << (2 * s));
        let s_below = if q == MIN_NORMAL_BITS { s - 1 } else { s };
        let below = r4 + (big << (s_below + 2)) as i64 - (1i64 << (2 * s_below));
        if above > 0 || (above == 0 && odd) {
            est += 1;
        } else if below < 0 || (below == 0 && odd) {
            est -= 1;
        } else {
            break;
        }
    }
    est
}

/// Bits of the correctly rounded `f32` modulus of two non-NaN magnitudes given by their
/// sign-stripped bits. This mirrors `hypot_key` in `shaders/argmax.metal` step for step:
/// a float estimate from exactly scaled normal operands, a 32-bit integer certificate that
/// accepts the estimate or one of its neighbours, and exact 64-bit midpoint tests (ties to
/// even) for whatever the certificate cannot decide.
fn hypot_bits(re: u32, im: u32) -> u32 {
    let (a, b) = (re.max(im), re.min(im));
    if a == INF_BITS {
        return INF_BITS;
    }
    if b == 0 {
        return a;
    }
    if a < MIN_NORMAL_BITS {
        // Both subnormal: the nearest integer to sqrt(a² + b²) in units of 2^-149.
        let sum = u64::from(a) * u64::from(a) + u64::from(b) * u64::from(b);
        let mut q = (sum as f32).sqrt() as u64;
        for _ in 0..HYPOT_MAX_STEPS {
            if sum > q * q + q {
                q += 1;
            } else if sum <= q * q - q {
                q -= 1;
            } else {
                break;
            }
        }
        return q as u32;
    }
    let exp_a = a >> 23;
    let gap = exp_a - (b >> 23).max(1);
    if gap > 12 {
        return a;
    }
    let sig_a = MIN_NORMAL_BITS | (a & FRACTION_MASK);
    let sig_b = if b >= MIN_NORMAL_BITS { MIN_NORMAL_BITS | (b & FRACTION_MASK) } else { b };

    let fa = f32::from_bits(0x3f80_0000 | (a & FRACTION_MASK));
    let fb = sig_b as f32 * f32::from_bits((104 - gap) << 23);
    let mut est = (fa * fa + fb * fb).sqrt().to_bits().clamp(0x3f80_0000, 0x407f_ffff);

    let q = MIN_NORMAL_BITS | (est & FRACTION_MASK);
    let e2 = 2 * ((est >> 23) - 127);
    let lo_term = ((u64::from(sig_b) * u64::from(sig_b)) >> (2 * gap)) as u32;
    let r = sig_a.wrapping_mul(sig_a).wrapping_add(lo_term).wrapping_sub(q.wrapping_mul(q) << e2)
        as i32;
    let w = (q << e2) as i32;
    let pow2 = q == MIN_NORMAL_BITS;
    if r >= (if pow2 { 0 } else { 2 - w }) && r < w {
        // est is already correctly rounded.
    } else if r >= w + 2 && r < 3 * w {
        est += 1;
    } else if !pow2 && r < -w && r >= (if q == MIN_NORMAL_BITS + 1 { -2 * w } else { 10 - 3 * w }) {
        est -= 1;
    } else {
        est = hypot_round_exact(est, sig_a, sig_b, gap);
    }
    ((est - 0x3f80_0000) + (exp_a << 23)).min(INF_BITS)
}

/// The `f32` complex magnitude formulas shared with the kernel. Subnormals participate;
/// the host arithmetic is IEEE, and the kernel matches it without flush-to-zero. Hypot is
/// the correctly rounded modulus (see [`hypot_bits`]).
fn mag32(z: [f32; 2], kind: Magnitude) -> f32 {
    let (re, im) = (z[0].abs(), z[1].abs());
    if re.is_nan() || im.is_nan() {
        return f32::NAN;
    }
    match kind {
        Magnitude::L1 => re + im,
        Magnitude::Hypot => f32::from_bits(hypot_bits(re.to_bits(), im.to_bits())),
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
    assert!(distance <= 1, "magnitude {have:e} is {distance} ULP from true max {want:e}");
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
fn complex_subnormal_parts_keep_their_magnitude() {
    let c = ctx();
    let t = f32::from_bits(1);
    // The failing adversarial case: |t| + |t| beats a single subnormal part, and a true
    // zero must not win.
    let mut pair = vec![[0.0f32, 0.0]; 6];
    pair[3] = [0.0, -t];
    pair[5] = [t, t];
    check_complex(&c, &pair, Magnitude::L1, "one-bit subnormal l1");
    assert_eq!(gpu_argmax_complex(&c, &pair, Magnitude::L1), 5);
    check_complex(&c, &pair, Magnitude::Hypot, "one-bit subnormal hypot");

    // Two largest subnormals sum to the normal 0x00fffffe, above the smallest normal.
    let s = f32::from_bits(0x007f_ffff);
    let summed = vec![[f32::MIN_POSITIVE, 0.0], [s, s]];
    check_complex(&c, &summed, Magnitude::L1, "subnormal sum beats min normal");
    assert_eq!(gpu_argmax_complex(&c, &summed, Magnitude::L1), 1);
    check_complex(&c, &summed, Magnitude::Hypot, "subnormal hypot beats min normal");

    let tiny = f32::from_bits(3);
    let small = vec![[tiny, 0.0], [0.0, 0.0], [0.0, tiny], [f32::from_bits(4), 0.0], [0.0, 0.0]];
    for kind in [Magnitude::L1, Magnitude::Hypot] {
        check_complex(&c, &small, kind, "largest tiny subnormal");
        assert_eq!(gpu_argmax_complex(&c, &small, kind), 3, "{kind:?}");
    }

    // Half-ulp boundary: biased exp 24 has ulp 2^-126, and a subnormal of 2^22 units
    // is exactly half an ulp (tie stays), while one unit more rounds up.
    let n = f32::from_bits(24 << 23);
    let tie = vec![[n, 0.0], [n, f32::from_bits(1 << 22)]];
    let up = vec![[n, 0.0], [n, f32::from_bits((1 << 22) + 1)]];
    check_complex(&c, &tie, Magnitude::L1, "half ulp tie");
    assert_eq!(gpu_argmax_complex(&c, &tie, Magnitude::L1), 0);
    check_complex(&c, &up, Magnitude::L1, "half ulp rounds up");
    assert_eq!(gpu_argmax_complex(&c, &up, Magnitude::L1), 1);

    let mut fields = Vec::new();
    for bits in [1u32, 2, 3, 0x10, 0x7f, 0x80, 0xff, 0x400000, 0x7fffff] {
        let p = f32::from_bits(bits);
        fields.push([p, 0.0]);
        fields.push([0.0, -p]);
        fields.push([p, p]);
        fields.push([-p, f32::from_bits(bits / 2)]);
    }
    for exp in [1u32, 2, 8, 12, 13, 24, 25, 127] {
        let n = f32::from_bits(exp << 23);
        fields.push([n, f32::from_bits(1)]);
        fields.push([n, f32::from_bits(0x7f_ffff)]);
        fields.push([n * 0.75, f32::from_bits(0x40_0000)]);
    }
    for (i, z) in fields.iter().copied().enumerate() {
        let against_zero = vec![[0.0, 0.0], z];
        check_complex(&c, &against_zero, Magnitude::L1, &format!("l1 field {i}"));
        check_complex(&c, &against_zero, Magnitude::Hypot, &format!("hypot field {i}"));
    }

    let mut r = rng(22);
    let wide: Vec<[f32; 2]> = (0..20_000)
        .map(|_| {
            let e = |r: &mut ChaCha8Rng| 2f32.powi(r.random_range(-149..60));
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
        "{:>10} {:>10} {:>10} {:>14} {:>13} {:>10}",
        "N", "sync us", "gpu us", "base sync us", "base gpu us", "GB/s(gpu)"
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
        // Baseline: same bytes, up to 1024 groups x 256 threads, no tie logic and no second pass.
        let n4 = (n / 4) as u32;
        let threads = (n4 as usize).clamp(1, 1024 * 256);
        let partials = alloc_shared::<f32>(&c, threads);
        let run_base = || {
            let started = std::time::Instant::now();
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
            (started.elapsed().as_secs_f64() * 1e6, gpu_us(cmd))
        };
        for _ in 0..5 {
            run_base();
        }
        let (base_wall, base): (Vec<f64>, Vec<f64>) = (0..200).map(|_| run_base()).unzip();
        let (base_sync, base) = (median(base_wall), median(base));
        let gbs = (n * 4) as f64 / (gpu * 1e-6) / 1e9;
        println!("{n:>10} {sync:>10.1} {gpu:>10.1} {base_sync:>14.1} {base:>13.1} {gbs:>10.1}");
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

// --- Audit regression test (ARG3) --------------------------------------------------------

/// ARG3: with Hypot, the picked element's f64 modulus must be within 1 ULP (as f32) of the
/// f64 maximum, also for near-ties where the f32 formula's rounding decides the order.
#[test]
fn audit_hypot_pick_is_within_one_ulp_of_f64_max_on_near_ties() {
    let ctx = ctx();
    let mut rng = ChaCha8Rng::seed_from_u64(6);
    let pairs = 4096usize;
    let m64 = |z: [f32; 2]| f64::from(z[0]).hypot(f64::from(z[1]));
    let mut bad = Vec::new();
    for batch in 0..40 {
        let scale = [1.0f64, 1e-30, 1e30, 3e38 / 2.0, 1e-38][batch % 5];
        let mut data = Vec::with_capacity(2 * pairs);
        for _ in 0..pairs {
            let rad = scale * rng.random_range(0.5f64..1.0);
            let t1 = rng.random_range(0.0f64..std::f64::consts::FRAC_PI_2);
            let t2 = rng.random_range(0.0f64..std::f64::consts::FRAC_PI_2);
            let k = 1.0 + f64::from(rng.random_range(-3i32..=3)) * 2f64.powi(-24);
            data.push([(rad * t1.cos()) as f32, (rad * t1.sin()) as f32]);
            data.push([(rad * k * t2.cos()) as f32, (rad * k * t2.sin()) as f32]);
        }
        let xb = upload(&ctx, &data);
        let outs = alloc_shared::<u32>(&ctx, pairs);
        let cmd = ctx.queue.new_command_buffer();
        for i in 0..pairs {
            let x = TensorView::<[f32; 2]>::new(&xb, 2 * i, vec![2], vec![1]);
            let out = TensorView::<u32>::row_major(&outs, i, vec![1]);
            encode_argmax_complex(&ctx, cmd, &x, Magnitude::Hypot, &out).unwrap();
        }
        cmd.commit();
        cmd.wait_until_completed();
        // SAFETY: `outs` holds `pairs` u32s and the command buffer has completed.
        let got = unsafe { std::slice::from_raw_parts(outs.contents() as *const u32, pairs) };
        for i in 0..pairs {
            let (z, w) = (data[2 * i], data[2 * i + 1]);
            let want =
                ref_best([mag32(z, Magnitude::Hypot), mag32(w, Magnitude::Hypot)].into_iter());
            let tmax = m64(z).max(m64(w)) as f32;
            let picked = m64([z, w][got[i] as usize]) as f32;
            let ulps = (i64::from(tmax.to_bits()) - i64::from(picked.to_bits())).unsigned_abs();
            if ulps > 1 || got[i] as usize != want {
                bad.push(format!(
                    "z={z:?} w={w:?}: got {} (reference {want}), {ulps} ULP below the f64 max",
                    got[i]
                ));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "{} near-tie violations, first: {:#?}",
        bad.len(),
        &bad[..bad.len().min(5)]
    );
}

// --- Hypot key: exactness of the reference and bit equality with the kernel -------------

/// Floor of the square root of `n`.
fn isqrt(n: u128) -> u128 {
    let mut r = (n as f64).sqrt() as u128;
    while r * r > n {
        r -= 1;
    }
    while (r + 1) * (r + 1) <= n {
        r += 1;
    }
    r
}

/// Exact oracle, independent of [`hypot_bits`]: bits of `sqrt(re² + im²)` rounded to nearest
/// (ties to even) from the exact 128-bit integer sum of squares. Inputs are the bits of
/// finite non-negative values.
fn hypot_bits_exact(re: u32, im: u32) -> u32 {
    let (a, b) = (re.max(im), re.min(im));
    let parts = |v: u32| {
        if v >= MIN_NORMAL_BITS {
            (u128::from(MIN_NORMAL_BITS | (v & FRACTION_MASK)), v >> 23)
        } else {
            (u128::from(v), 1)
        }
    };
    let ((sig_a, exp_a), (sig_b, exp_b)) = (parts(a), parts(b));
    let gap = exp_a - exp_b;
    if gap > 30 {
        // lo/hi < 2^-29: the modulus exceeds `hi` by far less than half an ULP.
        return a;
    }
    // Units of 2^(exp_b - 150): the sum of squares is an integer below 2^110.
    let sum = (sig_a << gap) * (sig_a << gap) + sig_b * sig_b;
    let root = isqrt(sum);
    // The result's ULP is 2^drop of these units (never finer than the units themselves).
    let drop = (128 - root.leading_zeros()).saturating_sub(24);
    let mut sig = if drop == 0 {
        root + u128::from(sum > root * root + root)
    } else {
        let floor = root >> drop;
        let midpoint = (floor << drop) + (1 << (drop - 1));
        match sum.cmp(&(midpoint * midpoint)) {
            std::cmp::Ordering::Less => floor,
            std::cmp::Ordering::Greater => floor + 1,
            std::cmp::Ordering::Equal => floor + (floor & 1),
        }
    };
    let mut exp = exp_b + drop;
    if sig == 1 << 24 {
        sig >>= 1;
        exp += 1;
    }
    if sig < 1 << 23 {
        assert_eq!(exp, 1, "a subnormal result has the subnormal exponent");
        sig as u32
    } else if exp >= 255 {
        INF_BITS
    } else {
        (exp << 23) | (sig as u32 & FRACTION_MASK)
    }
}

/// Finite complex elements (as sign-stripped bits) that stress the Hypot key: uniform bit
/// patterns, small exponent gaps, subnormal and mixed parts, sparse and saturated
/// significands (ties and binade edges), and parts near `f32::MAX`.
fn hypot_stress_bits(r: &mut ChaCha8Rng, rounds: usize) -> Vec<[u32; 2]> {
    let mut out = Vec::with_capacity(6 * rounds);
    let sparse = |r: &mut ChaCha8Rng| {
        let mut m = 0u32;
        for _ in 0..r.random_range(0..4) {
            m |= 1 << r.random_range(0..23);
        }
        if r.random_bool(0.5) {
            m ^ FRACTION_MASK
        } else {
            m
        }
    };
    for _ in 0..rounds {
        out.push([r.random_range(0..INF_BITS), r.random_range(0..INF_BITS)]);
        let exp_a = r.random_range(1..255u32);
        let exp_b = exp_a.saturating_sub(r.random_range(0..15));
        out.push([
            (exp_a << 23) | r.random_range(0..=FRACTION_MASK),
            (exp_b << 23) | r.random_range(0..=FRACTION_MASK),
        ]);
        let (wa, wb) = (r.random_range(1..26u32), r.random_range(1..26u32));
        out.push([r.random_range(0..1u32 << wa), r.random_range(0..1u32 << wb)]);
        let exp_a = r.random_range(1..255u32);
        let exp_b = exp_a.saturating_sub(r.random_range(0..14));
        out.push([(exp_a << 23) | sparse(r), (exp_b << 23) | sparse(r)]);
        out.push([
            (254 << 23) | r.random_range(0..=FRACTION_MASK),
            (r.random_range(252..255u32) << 23) | r.random_range(0..=FRACTION_MASK),
        ]);
        // Nearly equal parts just under an exponent boundary of the modulus.
        let near = (r.random_range(1..255u32) << 23) | (0x3504f3 + r.random_range(0..64u32) - 32);
        out.push([near, near + r.random_range(0..64u32) - 32]);
    }
    out
}

/// Primitive Pythagorean triples `p² + q² = c²` with `c` odd in `[2^24, 2^25)`: `c` needs 25
/// bits, so the exact modulus of `(p, q)` lies half way between two `f32` values. Each is
/// returned at several power-of-two scales, from subnormal results to results near overflow.
fn hypot_exact_ties(limit: usize) -> Vec<[u32; 2]> {
    fn gcd(a: u64, b: u64) -> u64 {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }
    let mut out = Vec::new();
    'search: for m in 2900u64..5793 {
        for k in (1 + (m & 1)..m).step_by(2) {
            let (c, p, q) = (m * m + k * k, m * m - k * k, 2 * m * k);
            let exact = |v: u64| (v as f32) as u64 == v;
            if (1 << 24..1 << 25).contains(&c) && gcd(m, k) == 1 && exact(p) && exact(q) {
                for scale in [-150i32, -140, -100, -24, 0, 50, 102] {
                    let s = 2f32.powi(scale);
                    out.push([(p as f32 * s).to_bits(), (q as f32 * s).to_bits()]);
                }
                if out.len() >= limit {
                    break 'search;
                }
            }
        }
    }
    out
}

/// Hand-picked extremes: zeros, the smallest and largest subnormals, the smallest normal,
/// `f32::MAX`, and infinities in every pairing, plus the overflow boundary of the modulus.
fn hypot_extremes() -> Vec<[u32; 2]> {
    let max = f32::MAX.to_bits();
    let edge = [0, 1, 2, 3, 0x7f_ffff, MIN_NORMAL_BITS, MIN_NORMAL_BITS + 1, 0x3f80_0000];
    let top = [max - 1, max, INF_BITS];
    let mut out = Vec::new();
    for &a in edge.iter().chain(&top) {
        for &b in edge.iter().chain(&top) {
            out.push([a, b]);
        }
    }
    // sqrt(MAX² + b²) crosses the last midpoint below 2^128 as b grows through 2^116.
    for b in (0x7980_0000u32 - 40)..(0x7980_0000 + 40) {
        out.push([max, b]);
        out.push([b, max - 1]);
    }
    out
}

/// Asserts that the kernel's Hypot key of every element equals the reference key bit for
/// bit, and returns the number of elements checked.
///
/// The key is observed through the public API: with `k` the reference modulus, the purely
/// real element `(k, 0)` has key `k` exactly, so the two-element scan `[(k, 0), z]` returns 0
/// only if `key(z) <= k`, and `[z, (k, 0)]` returns 0 only if `key(z) >= k`.
fn assert_hypot_keys_match(ctx: &MetalContext, cases: &[[u32; 2]], what: &str) -> usize {
    let mut r = rng(41);
    for chunk in cases.chunks(8192) {
        let mut data = Vec::with_capacity(3 * chunk.len());
        for &[re, im] in chunk {
            // Signs must not matter: flip them at random.
            let z = [
                f32::from_bits(re | (u32::from(r.random_bool(0.5)) << 31)),
                f32::from_bits(im | (u32::from(r.random_bool(0.5)) << 31)),
            ];
            let k = [mag32(z, Magnitude::Hypot), 0.0];
            data.extend([k, z, k]);
        }
        let xb = upload(ctx, &data);
        let outs = alloc_shared::<u32>(ctx, 2 * chunk.len());
        let cmd = ctx.queue.new_command_buffer();
        for i in 0..2 * chunk.len() {
            let x = vector::<[f32; 2]>(&xb, 3 * (i / 2) + i % 2, 2, 1);
            let out = TensorView::<u32>::row_major(&outs, i, vec![1]);
            encode_argmax_complex(ctx, cmd, &x, Magnitude::Hypot, &out).unwrap();
        }
        cmd.commit();
        cmd.wait_until_completed();
        // SAFETY: `outs` holds `2 * chunk.len()` u32s and the command buffer has completed.
        let got =
            unsafe { std::slice::from_raw_parts(outs.contents() as *const u32, 2 * chunk.len()) };
        for (i, &[re, im]) in chunk.iter().enumerate() {
            assert!(
                got[2 * i] == 0 && got[2 * i + 1] == 0,
                "{what}: kernel key of ({re:#010x}, {im:#010x}) is {} the reference {:#010x}",
                if got[2 * i] != 0 { "above" } else { "below" },
                hypot_bits(re, im),
            );
        }
    }
    cases.len()
}

/// The reference Hypot key is the correctly rounded modulus: it equals the exact integer
/// oracle on adversarial inputs, exact ties included, and is symmetric in its parts.
#[test]
fn hypot_reference_is_correctly_rounded() {
    let mut r = rng(40);
    let mut cases = hypot_stress_bits(&mut r, 300_000);
    cases.extend(hypot_exact_ties(200_000));
    cases.extend(hypot_extremes().into_iter().filter(|z| z[0] < INF_BITS && z[1] < INF_BITS));
    for &[re, im] in &cases {
        let want = hypot_bits_exact(re, im);
        assert_eq!(hypot_bits(re, im), want, "hypot_bits({re:#010x}, {im:#010x})");
        assert_eq!(hypot_bits(im, re), want, "hypot_bits({im:#010x}, {re:#010x})");
        // Within half an f32 ULP of the f64 modulus (which carries its own f64 rounding).
        if want < f32::MAX.to_bits() {
            let key = f64::from(f32::from_bits(want));
            let ulp = f64::from(f32::from_bits(want + 1)) - key;
            let modulus = f64::from(f32::from_bits(re)).hypot(f64::from(f32::from_bits(im)));
            let err = (modulus - key).abs() / ulp;
            assert!(err <= 0.5 + 1e-7, "({re:#010x}, {im:#010x}): {err} ULP from the modulus");
        }
    }
    // The extremes pin the special values directly.
    assert_eq!(hypot_bits(INF_BITS, 0), INF_BITS);
    assert_eq!(hypot_bits(1, INF_BITS), INF_BITS);
    assert_eq!(hypot_bits(0, 0), 0);
    assert_eq!(hypot_bits(1, 1), 1, "sqrt(2) units of 2^-149 rounds to 1");
    assert_eq!(hypot_bits(2, 2), 3, "sqrt(8) units of 2^-149 rounds to 3");
    let max = f32::MAX.to_bits();
    assert_eq!(hypot_bits(max, max), INF_BITS, "MAX * sqrt(2) overflows");
    assert_eq!(hypot_bits(max, 0x7900_0000), max, "stays finite below the last midpoint");
    assert_eq!(mag32([3.0, -4.0], Magnitude::Hypot), 5.0);
}

/// The kernel's Hypot key equals the reference key bit for bit: subnormal parts (which the
/// GPU would flush in float arithmetic), exact ties, overflow, and infinities included.
#[test]
fn hypot_key_matches_reference_bit_for_bit() {
    let c = ctx();
    let mut r = rng(42);
    assert_hypot_keys_match(&c, &hypot_extremes(), "extremes");
    assert_hypot_keys_match(&c, &hypot_exact_ties(20_000), "exact ties");
    assert_hypot_keys_match(&c, &hypot_stress_bits(&mut r, 20_000), "stress");
}

/// Large version of [`hypot_key_matches_reference_bit_for_bit`], with the reference checked
/// against the exact oracle on the same inputs.
#[test]
#[ignore = "exhaustive: run under the exclusive GPU lock"]
fn sweep_hypot_key_bits() {
    let c = ctx();
    let mut r = rng(43);
    let started = std::time::Instant::now();
    let mut checked = 0usize;
    let ties = hypot_exact_ties(usize::MAX);
    println!("\nexact-tie inputs: {}", ties.len());
    for cases in std::iter::once(ties).chain((0..40).map(|_| hypot_stress_bits(&mut r, 50_000))) {
        for &[re, im] in &cases {
            assert_eq!(hypot_bits(re, im), hypot_bits_exact(re, im), "({re:#010x}, {im:#010x})");
        }
        checked += assert_hypot_keys_match(&c, &cases, "sweep");
    }
    println!(
        "hypot key: {checked} inputs, kernel == reference == exact oracle, 0 mismatches, {:.1} s",
        started.elapsed().as_secs_f64()
    );
}
