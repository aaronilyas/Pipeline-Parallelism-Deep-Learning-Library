//! Contraction tests.
//!
//! Every GPU result is gated componentwise against an inline pure-Rust reference `C_ref`
//! that accumulates in f64 and is not rounded to the output type:
//!
//! `|C_gpu − C_ref| ≤ (K+2)·u·(|α||A||B| + |β||C₀|) + u_out·|C_ref| + η/2`
//!
//! with `u = 2⁻²⁴`, `u_out` the unit roundoff of the output type (`2⁻²⁴` for f32, `2⁻¹¹`
//! for f16) and `η` its smallest subnormal. `η/2` is the only allowance beyond the spec
//! formula: it is the rounding error a correctly rounded output has below the normal
//! range, where `u_out·|C_ref|` no longer covers half a unit in the last place.
//!
//! Each case also logs `|C_gpu − C_ref| / (√K·u)` as `err/(√K·u)`; run with `--nocapture`
//! to see it. The distance is to the unrounded f64 reference, so an f16 result includes
//! the rounding from the f32 accumulator down to f16 and the printed value is nonzero.
//!
//! The `bench_*` and `sweep_*` tests are `#[ignore]`d: run them under the exclusive GPU
//! lock with `-- --ignored --nocapture --test-threads=1`.

// MPS is reached through `msg_send!`, whose expansion references a cfg this crate does not own.
#![allow(unexpected_cfgs)]

use std::collections::HashMap;
use std::marker::PhantomData;
use std::time::Instant;

use half::f16;
use linalg::abi::{alloc_shared, compute_pipeline, LinalgError, MetalContext, TensorView};
use linalg::ops::contraction::{contract, encode_contract, encode_gemm, gemm, Element};
use metal::foreign_types::{ForeignType, ForeignTypeRef};
use metal::objc::runtime::{Object, NO};
use metal::objc::{class, msg_send, sel, sel_impl};
use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// Unit roundoff of the f32 accumulators, `2⁻²⁴`.
const U: f64 = 1.0 / 16_777_216.0;

/// Spare elements allocated past every view so stray writes are observable.
const TAIL: usize = 13;

/// Element types under test, with the conversions the f64 reference needs.
trait Scalar: Element + PartialEq + std::fmt::Debug + Send + Sync {
    const NAME: &'static str;
    /// Unit roundoff of the type.
    const U_OUT: f64;
    /// Half the smallest positive subnormal: the rounding error of a correctly rounded
    /// output below the normal range.
    const HALF_TINY: f64;
    /// Correctly rounded conversion (round to nearest even).
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
    fn bits(self) -> u32;
}

impl Scalar for f32 {
    const NAME: &'static str = "f32";
    const U_OUT: f64 = U;
    const HALF_TINY: f64 = 7.006_492_321_624_085e-46;
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn bits(self) -> u32 {
        self.to_bits()
    }
}

impl Scalar for f16 {
    const NAME: &'static str = "f16";
    const U_OUT: f64 = 1.0 / 2048.0;
    const HALF_TINY: f64 = U / 2.0;
    fn from_f64(x: f64) -> Self {
        f16::from_f64(x)
    }
    fn to_f64(self) -> f64 {
        f16::to_f64(self)
    }
    fn bits(self) -> u32 {
        u32::from(self.to_bits())
    }
}

fn context() -> MetalContext {
    MetalContext::new().expect("a Metal device is required")
}

/// Offset, shape, and strides of a view, all in elements.
#[derive(Clone, Debug)]
struct Layout {
    offset: usize,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Layout {
    /// Contiguous row-major layout at offset 0.
    fn dense(shape: &[usize]) -> Self {
        let order: Vec<usize> = (0..shape.len()).collect();
        Self::with(shape, &order, &vec![1; shape.len()], 0)
    }

    /// Layout whose memory nests the axes in `order` (outermost first), spacing axis `d`
    /// by `steps[d]` elements of its natural pitch, starting at `offset`.
    fn with(shape: &[usize], order: &[usize], steps: &[usize], offset: usize) -> Self {
        let mut strides = vec![0; shape.len()];
        let mut pitch = 1;
        for &axis in order.iter().rev() {
            strides[axis] = pitch * steps[axis];
            pitch *= shape[axis].max(1) * steps[axis];
        }
        Self { offset, shape: shape.to_vec(), strides }
    }

    /// Row-major matrix layout stored transposed (column-major memory).
    fn transposed(rows: usize, cols: usize) -> Self {
        Self::with(&[rows, cols], &[1, 0], &[1, 1], 0)
    }

    /// Number of buffer elements the view spans, counted from the buffer base.
    fn span(&self) -> usize {
        if self.shape.contains(&0) {
            return self.offset;
        }
        self.offset
            + self.shape.iter().zip(&self.strides).map(|(n, s)| (n - 1) * s).sum::<usize>()
            + 1
    }
}

/// A shared buffer holding one view plus `TAIL` spare elements.
struct Tensor<T> {
    buffer: metal::Buffer,
    cap: usize,
    layout: Layout,
    marker: PhantomData<T>,
}

impl<T: Scalar> Tensor<T> {
    fn new(ctx: &MetalContext, layout: Layout) -> Self {
        let cap = layout.span() + TAIL;
        Self { buffer: alloc_shared::<T>(ctx, cap), cap, layout, marker: PhantomData }
    }

    fn random(ctx: &MetalContext, layout: Layout, rng: &mut ChaCha8Rng) -> Self {
        let mut tensor = Self::new(ctx, layout);
        fill_random(tensor.raw_mut(), rng);
        tensor
    }

    fn raw(&self) -> &[T] {
        // SAFETY: the buffer holds at least `cap` elements and no command buffer is in flight.
        unsafe { std::slice::from_raw_parts(self.buffer.contents() as *const T, self.cap) }
    }

    fn raw_mut(&mut self) -> &mut [T] {
        // SAFETY: as in `raw`, and `&mut self` makes the access exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.buffer.contents() as *mut T, self.cap) }
    }

    fn view(&self) -> TensorView<'_, T> {
        view_of(&self.buffer, &self.layout)
    }
}

fn view_of<'a, T>(buffer: &'a metal::BufferRef, layout: &Layout) -> TensorView<'a, T> {
    TensorView::new(buffer, layout.offset, layout.shape.clone(), layout.strides.clone())
}

/// Fills `dst` with values uniform in `[-1, 1]`, rounded to `T`.
fn fill_random<T: Scalar>(dst: &mut [T], rng: &mut ChaCha8Rng) {
    for x in dst {
        *x = T::from_f64(rng.random_range(-1.0f64..=1.0));
    }
}

fn to_f64s<T: Scalar>(raw: &[T]) -> Vec<f64> {
    raw.iter().map(|x| x.to_f64()).collect()
}

/// One operand of the reference: raw buffer contents, layout, and axis labels.
#[derive(Clone, Copy)]
struct Operand64<'a> {
    data: &'a [f64],
    layout: &'a Layout,
    labels: &'a str,
}

/// Reference result for every logical element of C, in row-major order of C's labels.
struct Expected {
    /// `α·ΣAB + β·C₀` accumulated in f64.
    exact: Vec<f64>,
    /// `|α|·Σ|A||B| + |β|·|C₀|`, the magnitude the error bound scales with.
    magnitude: Vec<f64>,
    /// Buffer element index of each logical element of C.
    index: Vec<usize>,
    /// Number of contracted index combinations (the K of the bound).
    k: usize,
}

/// Element offsets (relative to the view offset) of `labels`' axes as the indices of the
/// `over` labels run in row-major order; labels absent from the operand contribute nothing.
fn offset_table(op: Operand64<'_>, over: &[char], extent: &HashMap<char, usize>) -> Vec<usize> {
    let strides: Vec<usize> = over
        .iter()
        .map(|l| op.labels.chars().position(|x| x == *l).map_or(0, |p| op.layout.strides[p]))
        .collect();
    let total: usize = over.iter().map(|l| extent[l]).product();
    let mut idx = vec![0usize; over.len()];
    let mut table = Vec::with_capacity(total);
    for _ in 0..total {
        table.push(idx.iter().zip(&strides).map(|(i, s)| i * s).sum());
        for d in (0..over.len()).rev() {
            idx[d] += 1;
            if idx[d] < extent[&over[d]] {
                break;
            }
            idx[d] = 0;
        }
    }
    table
}

/// Pure-Rust reference: f64 accumulation over the contracted labels, BLAS scalar rules
/// (`α = 0` ignores A and B, `β = 0` ignores C₀).
fn reference(
    alpha: f32,
    a: Operand64<'_>,
    b: Operand64<'_>,
    beta: f32,
    c: Operand64<'_>,
) -> Expected {
    let mut extent = HashMap::new();
    for op in [a, b, c] {
        for (label, &n) in op.labels.chars().zip(&op.layout.shape) {
            assert_eq!(*extent.entry(label).or_insert(n), n, "test bug: extent mismatch");
        }
    }
    let free: Vec<char> = c.labels.chars().collect();
    let summed: Vec<char> =
        a.labels.chars().filter(|l| b.labels.contains(*l) && !c.labels.contains(*l)).collect();
    let (a_free, a_sum) = (offset_table(a, &free, &extent), offset_table(a, &summed, &extent));
    let (b_free, b_sum) = (offset_table(b, &free, &extent), offset_table(b, &summed, &extent));
    let index: Vec<usize> =
        offset_table(c, &free, &extent).into_iter().map(|o| c.layout.offset + o).collect();
    let (alpha, beta) = (f64::from(alpha), f64::from(beta));
    let (exact, magnitude) = (0..index.len())
        .into_par_iter()
        .map(|i| {
            let (mut sum, mut abs) = (0.0f64, 0.0f64);
            if alpha != 0.0 {
                let (a0, b0) = (a.layout.offset + a_free[i], b.layout.offset + b_free[i]);
                for (ka, kb) in a_sum.iter().zip(&b_sum) {
                    let product = a.data[a0 + ka] * b.data[b0 + kb];
                    sum += product;
                    abs += product.abs();
                }
                sum *= alpha;
                abs *= alpha.abs();
            }
            if beta != 0.0 {
                sum += beta * c.data[index[i]];
                abs += (beta * c.data[index[i]]).abs();
            }
            (sum, abs)
        })
        .unzip();
    Expected { exact, magnitude, index, k: a_sum.len() }
}

/// Error statistics of one checked result.
struct Stats {
    /// Largest accumulation error over `√K·u` (see the module documentation).
    normalized: f64,
    /// Worst `|C_gpu − C_ref| / tolerance`; at most 1 when the gate holds.
    gate: f64,
}

/// Applies the accuracy gate to every element of C against the unrounded f64 reference,
/// logs the error statistics, and panics on the first violation.
fn check<T: Scalar>(name: &str, got: &[T], expected: &Expected) -> Stats {
    let k = expected.k as f64;
    let (mut max_err, mut gate, mut max_ulps) = (0.0f64, 0.0f64, 0.0f64);
    for (i, &at) in expected.index.iter().enumerate() {
        let (value, exact) = (got[at].to_f64(), expected.exact[i]);
        let rounded = T::from_f64(exact).to_f64();
        if !rounded.is_finite() {
            assert!(
                value == rounded || (value.is_nan() && rounded.is_nan()),
                "{name} [{}]: element {i} is {value}, expected {rounded}",
                T::NAME
            );
            continue;
        }
        let err = (value - exact).abs();
        let tol = (k + 2.0) * U * expected.magnitude[i] + T::U_OUT * exact.abs() + T::HALF_TINY;
        assert!(
            err <= tol,
            "{name} [{}]: element {i} is {value}, f64 reference {exact} (rounds to {rounded}); \
             error {err:e} exceeds tolerance {tol:e} with K = {k}",
            T::NAME
        );
        max_err = max_err.max(err);
        gate = gate.max(err / tol);
        max_ulps = max_ulps.max(err / (2.0 * T::U_OUT * exact.abs()).max(2.0 * T::HALF_TINY));
    }
    // Distance to the unrounded reference, not to the rounded output. f16 therefore logs
    // its output rounding instead of a structural zero.
    let normalized = max_err / (k.max(1.0).sqrt() * U);
    println!(
        "[contraction] {name} [{}] K={k}: max_err={max_err:.3e} err/(sqrtK*u)={normalized:.2} \
         max_ulps={max_ulps:.2} gate={gate:.3}",
        T::NAME
    );
    Stats { normalized, gate }
}

/// Asserts that every buffer element outside `written` kept its exact bit pattern.
fn assert_untouched<T: Scalar>(name: &str, before: &[T], after: &[T], written: &[usize]) {
    let mut is_written = vec![false; before.len()];
    for &i in written {
        is_written[i] = true;
    }
    for (i, (x, y)) in before.iter().zip(after).enumerate() {
        assert!(
            is_written[i] || x.bits() == y.bits(),
            "{name} [{}]: element {i} outside the destination view changed from {x:?} to {y:?}",
            T::NAME
        );
    }
}

/// One labelled contraction `C = α·A·B + β·C` on freshly allocated random tensors.
struct Case<'s> {
    name: String,
    a: (Layout, &'s str),
    b: (Layout, &'s str),
    c: (Layout, &'s str),
    alpha: f32,
    beta: f32,
    /// Route through `gemm` (labels must then be batched-matrix shaped) instead of `contract`.
    via_gemm: bool,
}

impl<'s> Case<'s> {
    fn new(
        name: impl Into<String>,
        a: (Layout, &'s str),
        b: (Layout, &'s str),
        c: (Layout, &'s str),
    ) -> Self {
        Self { name: name.into(), a, b, c, alpha: 1.0, beta: 0.0, via_gemm: false }
    }

    /// Plain matrix product with dense row-major operands, routed through `gemm`.
    fn matmul(m: usize, n: usize, k: usize) -> Case<'static> {
        let mut case = Case::new(
            format!("gemm {m}x{n}x{k}"),
            (Layout::dense(&[m, k]), "mk"),
            (Layout::dense(&[k, n]), "kn"),
            (Layout::dense(&[m, n]), "mn"),
        );
        case.via_gemm = true;
        case
    }

    fn scalars(mut self, alpha: f32, beta: f32) -> Self {
        self.alpha = alpha;
        self.beta = beta;
        self.name = format!("{} alpha={alpha} beta={beta}", self.name);
        self
    }

    /// Runs the case and gates the result; with `β = 0` the destination buffer starts
    /// filled with NaN, so any read of C would poison the output.
    fn run<T: Scalar>(&self, ctx: &MetalContext, seed: u64) -> Stats {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let a = Tensor::<T>::random(ctx, self.a.0.clone(), &mut rng);
        let b = Tensor::<T>::random(ctx, self.b.0.clone(), &mut rng);
        let mut c = Tensor::<T>::random(ctx, self.c.0.clone(), &mut rng);
        if self.beta == 0.0 {
            c.raw_mut().fill(T::from_f64(f64::NAN));
        }
        let (a0, b0, c0) = (a.raw().to_vec(), b.raw().to_vec(), c.raw().to_vec());
        let expected = reference(
            self.alpha,
            Operand64 { data: &to_f64s(&a0), layout: &a.layout, labels: self.a.1 },
            Operand64 { data: &to_f64s(&b0), layout: &b.layout, labels: self.b.1 },
            self.beta,
            Operand64 { data: &to_f64s(&c0), layout: &c.layout, labels: self.c.1 },
        );
        let result = if self.via_gemm {
            gemm(ctx, self.alpha, &a.view(), &b.view(), self.beta, &c.view())
        } else {
            let (av, bv, cv) = (a.view(), b.view(), c.view());
            contract(ctx, self.alpha, (&av, self.a.1), (&bv, self.b.1), self.beta, (&cv, self.c.1))
        };
        assert_eq!(result, Ok(()), "{} [{}]", self.name, T::NAME);
        assert_untouched(&self.name, &a0, a.raw(), &[]);
        assert_untouched(&self.name, &b0, b.raw(), &[]);
        assert_untouched(&self.name, &c0, c.raw(), &expected.index);
        check(&self.name, c.raw(), &expected)
    }
}

/// Square products, at and off the tile grid.
fn square<T: Scalar>(ctx: &MetalContext) {
    for (seed, n) in [1usize, 8, 64, 96, 100, 256].into_iter().enumerate() {
        Case::matmul(n, n, n).run::<T>(ctx, seed as u64);
    }
    Case::matmul(192, 192, 192).scalars(-1.25, 0.75).run::<T>(ctx, 7);
}

/// Each of M, N, and K equal to 1, alone and combined.
fn unit_extents<T: Scalar>(ctx: &MetalContext) {
    let shapes = [
        (1, 37, 29),
        (41, 1, 29),
        (41, 37, 1),
        (1, 1, 1),
        (1, 1, 500),
        (1, 300, 1),
        (300, 1, 1),
        (1, 64, 64),
        (64, 1, 64),
        (64, 64, 1),
    ];
    for (seed, (m, n, k)) in shapes.into_iter().enumerate() {
        Case::matmul(m, n, k).run::<T>(ctx, 100 + seed as u64);
        Case::matmul(m, n, k).scalars(1.5, -0.5).run::<T>(ctx, 200 + seed as u64);
    }
}

/// `K = 0` leaves `C = β·C`; with `β = 0` that is exact zero even where C held NaN.
fn zero_k<T: Scalar>(ctx: &MetalContext) {
    for (seed, beta) in [0.0f32, 1.0, -0.5].into_iter().enumerate() {
        Case::matmul(17, 9, 0).scalars(1.0, beta).run::<T>(ctx, 300 + seed as u64);
        let batched = Case::new(
            "batched K=0",
            (Layout::dense(&[3, 5, 0, 2]), "bmkl"),
            (Layout::dense(&[3, 0, 2, 4]), "bkln"),
            (Layout::dense(&[3, 5, 4]), "bmn"),
        );
        batched.scalars(2.0, beta).run::<T>(ctx, 310 + seed as u64);
    }
}

/// Non-dyadic `β` with `K = 0`. `half(β * f32(C))` rounds twice: `β = -0.999` and
/// f16 `0x0DDC` store `0x8DDA`, one ulp off `0x8DDB`. The scale kernel must round once.
#[test]
fn scale_f16_nondyadic_beta() {
    let ctx = context();
    let (m, n) = (4usize, 3usize);
    let mut rng = ChaCha8Rng::seed_from_u64(1800);
    let a = Tensor::<f16>::random(&ctx, Layout::dense(&[m, 0]), &mut rng);
    let b = Tensor::<f16>::random(&ctx, Layout::dense(&[0, n]), &mut rng);
    let mut c = Tensor::<f16>::random(&ctx, Layout::dense(&[m, n]), &mut rng);
    c.raw_mut()[0] = f16::from_bits(0x0DDC);
    c.raw_mut()[1] = f16::from_f64(f64::INFINITY);
    c.raw_mut()[2] = f16::from_f64(f64::NAN);
    let (a0, b0, c0) = (a.raw().to_vec(), b.raw().to_vec(), c.raw().to_vec());
    let beta = -0.999f32;
    let expected = reference(
        1.0,
        Operand64 { data: &to_f64s(&a0), layout: &a.layout, labels: "mk" },
        Operand64 { data: &to_f64s(&b0), layout: &b.layout, labels: "kn" },
        beta,
        Operand64 { data: &to_f64s(&c0), layout: &c.layout, labels: "mn" },
    );
    assert_eq!(gemm(&ctx, 1.0, &a.view(), &b.view(), beta, &c.view()), Ok(()));
    assert_eq!(c.raw()[0].to_bits(), 0x8DDB, "β·f16 0x0DDC must round once to 0x8DDB");
    assert_untouched("nondyadic beta", &a0, a.raw(), &[]);
    assert_untouched("nondyadic beta", &b0, b.raw(), &[]);
    assert_untouched("nondyadic beta", &c0, c.raw(), &expected.index);
    check("nondyadic beta K=0", c.raw(), &expected);
}

/// `M = 0`, `N = 0`, or an empty batch is a no-op that leaves every buffer untouched.
fn empty_output<T: Scalar>(ctx: &MetalContext) {
    for (seed, (m, n, k)) in [(0, 5, 3), (5, 0, 3), (0, 0, 0), (0, 4, 0)].into_iter().enumerate() {
        Case::matmul(m, n, k).scalars(1.0, -0.5).run::<T>(ctx, 400 + seed as u64);
    }
    let empty_batch = Case::new(
        "empty batch",
        (Layout::dense(&[0, 4, 3]), "bmk"),
        (Layout::dense(&[0, 3, 5]), "bkn"),
        (Layout::dense(&[0, 4, 5]), "bmn"),
    );
    empty_batch.run::<T>(ctx, 410);
}

/// Prime extents, so every tile dimension has a ragged edge.
fn prime_sizes<T: Scalar>(ctx: &MetalContext) {
    Case::matmul(127, 251, 63).run::<T>(ctx, 500);
    Case::matmul(127, 251, 63).scalars(-0.75, 1.0).run::<T>(ctx, 501);
    Case::matmul(251, 127, 67).scalars(0.5, -0.5).run::<T>(ctx, 502);
}

/// Every combination of transposed (column-major) A, B, and C.
fn transposed_views<T: Scalar>(ctx: &MetalContext) {
    let (m, n, k) = (45, 70, 33);
    let pick = |t: bool, rows, cols| {
        if t {
            Layout::transposed(rows, cols)
        } else {
            Layout::dense(&[rows, cols])
        }
    };
    for mask in 0..8u64 {
        let (ta, tb, tc) = (mask & 1 != 0, mask & 2 != 0, mask & 4 != 0);
        let case = Case::new(
            format!("transposed a={ta} b={tb} c={tc}"),
            (pick(ta, m, k), "mk"),
            (pick(tb, k, n), "kn"),
            (pick(tc, m, n), "mn"),
        );
        case.scalars(1.0, if mask % 2 == 0 { 0.0 } else { -0.5 }).run::<T>(ctx, 600 + mask);
    }
    let mut via_gemm = Case::matmul(m, n, k);
    via_gemm.a.0 = Layout::transposed(m, k);
    via_gemm.c.0 = Layout::transposed(m, n);
    via_gemm.run::<T>(ctx, 610);
}

/// Stepped views with non-zero offsets; the gaps between elements must stay untouched.
fn strided_views<T: Scalar>(ctx: &MetalContext) {
    let (m, n, k) = (37, 41, 19);
    let layouts = [
        (
            Layout::with(&[m, k], &[0, 1], &[2, 3], 5),
            Layout::with(&[k, n], &[0, 1], &[3, 1], 11),
            Layout::with(&[m, n], &[0, 1], &[2, 2], 3),
        ),
        (
            Layout::with(&[m, k], &[1, 0], &[1, 2], 1),
            Layout::with(&[k, n], &[1, 0], &[2, 2], 9),
            Layout::with(&[m, n], &[1, 0], &[3, 1], 17),
        ),
        (
            Layout::with(&[m, k], &[0, 1], &[1, 1], 64),
            Layout::with(&[k, n], &[0, 1], &[1, 1], 1),
            Layout::with(&[m, n], &[0, 1], &[1, 1], 77),
        ),
    ];
    for (seed, (a, b, c)) in layouts.into_iter().enumerate() {
        for (alpha, beta) in [(1.0, 0.0), (-2.0, 1.0), (0.5, -0.5)] {
            let case = Case::new(
                format!("strided #{seed}"),
                (a.clone(), "mk"),
                (b.clone(), "kn"),
                (c.clone(), "mn"),
            );
            case.scalars(alpha, beta).run::<T>(ctx, 700 + seed as u64);
        }
    }
    let mut via_gemm = Case::matmul(70, 70, 70);
    via_gemm.a.0 = Layout::with(&[70, 70], &[0, 1], &[1, 2], 3);
    via_gemm.b.0 = Layout::with(&[70, 70], &[1, 0], &[2, 1], 6);
    via_gemm.c.0 = Layout::with(&[70, 70], &[0, 1], &[3, 1], 9);
    via_gemm.scalars(1.0, -0.5).run::<T>(ctx, 710);
}

/// Batched GEMM through `gemm`, and batch axes placed arbitrarily through `contract`.
fn batched<T: Scalar>(ctx: &MetalContext) {
    for (seed, batch) in [vec![3usize], vec![2, 3], vec![1, 1, 5], vec![70]].into_iter().enumerate()
    {
        let (m, n, k) = (33, 20, 17);
        let labels = &"xyz"[..batch.len()];
        let shape = |rows, cols| [batch.clone(), vec![rows, cols]].concat();
        let (la, lb, lc) = (format!("{labels}mk"), format!("{labels}kn"), format!("{labels}mn"));
        let mut case = Case::new(
            format!("batched gemm {batch:?}"),
            (Layout::dense(&shape(m, k)), &la),
            (Layout::dense(&shape(k, n)), &lb),
            (Layout::dense(&shape(m, n)), &lc),
        );
        case.via_gemm = true;
        case.scalars(1.0, if seed % 2 == 0 { 0.0 } else { -0.5 }).run::<T>(ctx, 800 + seed as u64);
    }
    let interleaved = Case::new(
        "batch axis in the middle",
        (Layout::with(&[19, 6, 23], &[1, 0, 2], &[1, 1, 1], 2), "mbk"),
        (Layout::with(&[23, 31, 6], &[0, 2, 1], &[1, 1, 2], 0), "knb"),
        (Layout::with(&[31, 6, 19], &[1, 2, 0], &[1, 1, 1], 4), "nbm"),
    );
    interleaved.scalars(0.5, -0.5).run::<T>(ctx, 810);
    let many_small = Case::new(
        "many small matrices",
        (Layout::dense(&[500, 4, 3]), "bmk"),
        (Layout::dense(&[500, 3, 5]), "bkn"),
        (Layout::dense(&[500, 4, 5]), "bmn"),
    );
    many_small.run::<T>(ctx, 811);
    let shared_b = Case::new(
        "one B for every A",
        (Layout::dense(&[7, 21, 40]), "bmk"),
        (Layout::dense(&[40, 13]), "kn"),
        (Layout::dense(&[7, 21, 13]), "bmn"),
    );
    shared_b.scalars(1.0, 1.0).run::<T>(ctx, 812);
}

/// Rank-4 operands contracted over two shared axes, with collapsible and scrambled layouts.
fn rank4_two_axes<T: Scalar>(ctx: &MetalContext) {
    let (i, j, k, l, m, n) = (5, 7, 6, 9, 4, 11);
    let dense = Case::new(
        "ijkl,klmn->ijmn dense",
        (Layout::dense(&[i, j, k, l]), "ijkl"),
        (Layout::dense(&[k, l, m, n]), "klmn"),
        (Layout::dense(&[i, j, m, n]), "ijmn"),
    );
    dense.run::<T>(ctx, 900);
    let scrambled = Case::new(
        "ijkl,lkmn->imjn scrambled",
        (Layout::with(&[i, j, k, l], &[2, 0, 3, 1], &[1, 1, 1, 1], 3), "ijkl"),
        (Layout::with(&[l, k, m, n], &[3, 0, 2, 1], &[1, 2, 1, 1], 0), "lkmn"),
        (Layout::with(&[i, m, j, n], &[1, 3, 0, 2], &[1, 1, 2, 1], 8), "imjn"),
    );
    scrambled.scalars(-1.5, -0.5).run::<T>(ctx, 901);
    let (i, j, k, l, m, n) = (9, 10, 12, 11, 8, 13);
    let larger = Case::new(
        "ijkl,mlkn->njim larger",
        (Layout::dense(&[i, j, k, l]), "ijkl"),
        (Layout::dense(&[m, l, k, n]), "mlkn"),
        (Layout::dense(&[n, j, i, m]), "njim"),
    );
    larger.scalars(1.0, 1.0).run::<T>(ctx, 902);
}

/// A group of more than four axes is fine as long as the host can merge it down.
fn collapsible_axes<T: Scalar>(ctx: &MetalContext) {
    let case = Case::new(
        "six collapsible M axes",
        (Layout::dense(&[2, 3, 2, 3, 2, 3, 5]), "abcdefk"),
        (Layout::dense(&[5, 7]), "kn"),
        (Layout::dense(&[2, 3, 2, 3, 2, 3, 7]), "abcdefn"),
    );
    case.scalars(1.0, -0.5).run::<T>(ctx, 1000);
    let unit_axes = Case::new(
        "unit axes everywhere",
        (Layout::with(&[1, 6, 1, 1, 9, 1], &[5, 3, 1, 0, 2, 4], &[1, 2, 1, 3, 1, 1], 1), "pmqrks"),
        (Layout::with(&[1, 9, 1, 1, 8, 1], &[0, 4, 2, 1, 3, 5], &[2, 1, 1, 1, 1, 4], 0), "skrtnq"),
        (Layout::with(&[1, 1, 6, 8, 1, 1], &[3, 2, 0, 1, 4, 5], &[1, 1, 1, 1, 1, 1], 2), "tpmnsq"),
    );
    unit_axes.run::<T>(ctx, 1001);
}

/// `β ∈ {0, 1, −0.5}` against several `α`, on the tiled, `K = 1`, and Split-K paths.
fn beta_variants<T: Scalar>(ctx: &MetalContext) {
    let shapes = [(70, 50, 40), (33, 65, 1), (5, 6, 20_000)];
    let mut seed = 1100;
    for (m, n, k) in shapes {
        for beta in [0.0f32, 1.0, -0.5] {
            for alpha in [1.0f32, -2.5] {
                Case::matmul(m, n, k).scalars(alpha, beta).run::<T>(ctx, seed);
                seed += 1;
            }
        }
    }
}

/// Small `M·N` with a long `K`: the Split-K path, including ragged slices.
fn split_k<T: Scalar>(ctx: &MetalContext) {
    let shapes =
        [(1, 1, 1_000_003), (8, 8, 65_536), (3, 5, 200_001), (40, 24, 30_000), (1, 70, 9_973)];
    for (seed, (m, n, k)) in shapes.into_iter().enumerate() {
        Case::matmul(m, n, k).run::<T>(ctx, 1200 + seed as u64);
        Case::matmul(m, n, k).scalars(0.5, -0.5).run::<T>(ctx, 1210 + seed as u64);
    }
    let batched = Case::new(
        "batched long K",
        (Layout::with(&[3, 2, 50_000], &[2, 0, 1], &[1, 1, 1], 1), "bmk"),
        (Layout::dense(&[3, 50_000, 4]), "bkn"),
        (Layout::dense(&[3, 2, 4]), "bmn"),
    );
    batched.scalars(-1.0, 1.0).run::<T>(ctx, 1220);
}

/// The same inputs give bit-identical outputs on every run, on every path.
fn deterministic<T: Scalar>(ctx: &MetalContext) {
    for (seed, (m, n, k)) in [(4, 4, 300_000), (130, 70, 90), (9, 1, 1)].into_iter().enumerate() {
        let mut rng = ChaCha8Rng::seed_from_u64(1300 + seed as u64);
        let a = Tensor::<T>::random(ctx, Layout::dense(&[m, k]), &mut rng);
        let b = Tensor::<T>::random(ctx, Layout::dense(&[k, n]), &mut rng);
        let mut runs = Vec::new();
        for _ in 0..3 {
            let c = Tensor::<T>::new(ctx, Layout::dense(&[m, n]));
            gemm(ctx, 1.0, &a.view(), &b.view(), 0.0, &c.view()).unwrap();
            runs.push(c.raw().iter().map(|x| x.bits()).collect::<Vec<u32>>());
        }
        assert!(
            runs[0] == runs[1] && runs[1] == runs[2],
            "{m}x{n}x{k} [{}] is not deterministic",
            T::NAME
        );
    }
}

/// NaN and infinity reach exactly the outputs that depend on them, including through
/// ragged boundary tiles.
fn non_finite_inputs<T: Scalar>(ctx: &MetalContext) {
    for (seed, (m, n, k)) in [(70, 45, 50), (3, 2, 40_000), (9, 11, 1)].into_iter().enumerate() {
        let mut rng = ChaCha8Rng::seed_from_u64(1400 + seed as u64);
        let mut a = Tensor::<T>::random(ctx, Layout::dense(&[m, k]), &mut rng);
        let mut b = Tensor::<T>::random(ctx, Layout::dense(&[k, n]), &mut rng);
        let c = Tensor::<T>::new(ctx, Layout::dense(&[m, n]));
        let (nan_row, inf_col) = (m - 1, n - 2);
        a.raw_mut()[nan_row * k + (k - 1)] = T::from_f64(f64::NAN);
        b.raw_mut()[(k / 2) * n + inf_col] = T::from_f64(f64::INFINITY);
        let c0 = c.raw().to_vec();
        let expected = reference(
            1.0,
            Operand64 { data: &to_f64s(a.raw()), layout: &a.layout, labels: "mk" },
            Operand64 { data: &to_f64s(b.raw()), layout: &b.layout, labels: "kn" },
            0.0,
            Operand64 { data: &to_f64s(&c0), layout: &c.layout, labels: "mn" },
        );
        gemm(ctx, 1.0, &a.view(), &b.view(), 0.0, &c.view()).unwrap();
        let name = format!("non-finite {m}x{n}x{k}");
        check(&name, c.raw(), &expected);
        for row in 0..m {
            for col in 0..n {
                let value = c.raw()[row * n + col].to_f64();
                let poisoned = row == nan_row || col == inf_col;
                assert_eq!(
                    value.is_finite(),
                    !poisoned,
                    "{name}: element ({row}, {col}) is {value}"
                );
            }
        }
    }
}

/// `α = 0` follows BLAS: A and B are not referenced, so their NaNs do not reach C.
fn alpha_zero<T: Scalar>(ctx: &MetalContext) {
    for (seed, (m, n, k)) in [(20, 30, 40), (2, 2, 50_000), (7, 5, 1)].into_iter().enumerate() {
        for beta in [0.0f32, 1.0, -0.5] {
            let mut rng = ChaCha8Rng::seed_from_u64(1500 + seed as u64);
            let mut a = Tensor::<T>::new(ctx, Layout::dense(&[m, k]));
            let mut b = Tensor::<T>::new(ctx, Layout::dense(&[k, n]));
            a.raw_mut().fill(T::from_f64(f64::NAN));
            b.raw_mut().fill(T::from_f64(f64::NAN));
            let c = Tensor::<T>::random(ctx, Layout::dense(&[m, n]), &mut rng);
            let c0 = c.raw().to_vec();
            gemm(ctx, 0.0, &a.view(), &b.view(), beta, &c.view()).unwrap();
            for (i, (before, after)) in c0.iter().zip(c.raw()).enumerate().take(m * n) {
                let want = T::from_f64(f64::from(beta) * before.to_f64());
                assert_eq!(after.to_f64(), want.to_f64(), "alpha=0 beta={beta} element {i}");
            }
        }
    }
}

/// The LU trailing update `C −= L·U`: three element-disjoint views of one matrix whose
/// byte ranges interleave.
fn lu_trailing_update<T: Scalar>(ctx: &MetalContext) {
    for (seed, (n, done, panel)) in
        [(96usize, 0usize, 32usize), (131, 17, 40), (64, 5, 1)].into_iter().enumerate()
    {
        let mut rng = ChaCha8Rng::seed_from_u64(1600 + seed as u64);
        let matrix = Tensor::<T>::random(ctx, Layout::dense(&[n, n]), &mut rng);
        let split = done + panel;
        let rest = n - split;
        let l = Layout { offset: split * n + done, shape: vec![rest, panel], strides: vec![n, 1] };
        let u = Layout { offset: done * n + split, shape: vec![panel, rest], strides: vec![n, 1] };
        let c = Layout { offset: split * n + split, shape: vec![rest, rest], strides: vec![n, 1] };
        let before = matrix.raw().to_vec();
        let data = to_f64s(&before);
        let expected = reference(
            -1.0,
            Operand64 { data: &data, layout: &l, labels: "mk" },
            Operand64 { data: &data, layout: &u, labels: "kn" },
            1.0,
            Operand64 { data: &data, layout: &c, labels: "mn" },
        );
        let (lv, uv, cv) = (
            view_of::<T>(&matrix.buffer, &l),
            view_of(&matrix.buffer, &u),
            view_of(&matrix.buffer, &c),
        );
        assert_eq!(gemm(ctx, -1.0, &lv, &uv, 1.0, &cv), Ok(()));
        let name = format!("LU trailing update n={n} panel={panel}");
        assert_untouched(&name, &before, matrix.raw(), &expected.index);
        check(&name, matrix.raw(), &expected);
    }
}

/// In-place scaling: C may be the very view A (or B) when every other extent is 1, which
/// is how an LU column is scaled by its pivot reciprocal.
fn in_place_scale<T: Scalar>(ctx: &MetalContext) {
    let (n, col, top) = (50usize, 7usize, 8usize);
    let mut rng = ChaCha8Rng::seed_from_u64(1700);
    let matrix = Tensor::<T>::random(ctx, Layout::dense(&[n, n]), &mut rng);
    let pivot = Tensor::<T>::random(ctx, Layout::dense(&[1, 1]), &mut rng);
    let column = Layout { offset: top * n + col, shape: vec![n - top, 1], strides: vec![n, 1] };
    let before = matrix.raw().to_vec();
    let data = to_f64s(&before);
    let scale = Operand64 { data: &to_f64s(pivot.raw()), layout: &pivot.layout, labels: "kn" };
    let column64 = |labels| Operand64 { data: &data, layout: &column, labels };
    let expected = reference(1.0, column64("mk"), scale, 0.0, column64("mn"));
    let view = view_of::<T>(&matrix.buffer, &column);
    assert_eq!(gemm(ctx, 1.0, &view, &pivot.view(), 0.0, &view), Ok(()));
    assert_untouched("in-place column scale", &before, matrix.raw(), &expected.index);
    check("in-place column scale", matrix.raw(), &expected);

    let row = Layout { offset: 3 * n + 2, shape: vec![1, n - 5], strides: vec![n, 1] };
    let before = matrix.raw().to_vec();
    let data = to_f64s(&before);
    let row64 = |labels| Operand64 { data: &data, layout: &row, labels };
    let scale = Operand64 { data: &to_f64s(pivot.raw()), layout: &pivot.layout, labels: "mk" };
    let expected = reference(-0.5, scale, row64("kn"), 1.0, row64("mn"));
    let view = view_of::<T>(&matrix.buffer, &row);
    assert_eq!(gemm(ctx, -0.5, &pivot.view(), &view, 1.0, &view), Ok(()));
    assert_untouched("in-place row scale", &before, matrix.raw(), &expected.index);
    check("in-place row scale", matrix.raw(), &expected);
}

/// C overlapping an input (other than the in-place scale) or itself is rejected with
/// `LinalgError::Overlap`, and nothing is written.
fn overlap_rejected<T: Scalar>(ctx: &MetalContext) {
    let n = 24usize;
    let mut rng = ChaCha8Rng::seed_from_u64(1800);
    let matrix = Tensor::<T>::random(ctx, Layout::dense(&[2 * n, 2 * n]), &mut rng);
    let other = Tensor::<T>::random(ctx, Layout::dense(&[n, n]), &mut rng);
    let before = matrix.raw().to_vec();
    let block = |row: usize, col: usize, rows: usize, cols: usize| Layout {
        offset: row * 2 * n + col,
        shape: vec![rows, cols],
        strides: vec![2 * n, 1],
    };
    let reject =
        |a: &TensorView<'_, T>, b: &TensorView<'_, T>, c: &TensorView<'_, T>, what: &str| {
            let result = gemm(ctx, 1.0, a, b, 0.0, c);
            assert!(matches!(result, Err(LinalgError::Overlap(_))), "{what}: got {result:?}");
        };
    let whole = view_of::<T>(&matrix.buffer, &block(0, 0, n, n));
    let shifted = view_of::<T>(&matrix.buffer, &block(n - 1, n - 1, n, n));
    let apart = view_of::<T>(&matrix.buffer, &block(n, n, n, n));
    reject(&whole, &other.view(), &whole, "C is A with K > 1");
    reject(&other.view(), &whole, &whole, "C is B with K > 1");
    reject(&whole, &other.view(), &shifted, "C shares one element with A");
    reject(&other.view(), &shifted, &whole, "C shares one element with B");
    assert_eq!(gemm(ctx, 1.0, &whole, &whole, 0.0, &apart), Ok(()), "A may alias B");

    let column = view_of::<T>(&matrix.buffer, &block(0, 3, n, 1));
    let lower = view_of::<T>(&matrix.buffer, &block(1, 3, n, 1));
    let one = view_of::<T>(&other.buffer, &Layout::dense(&[1, 1]));
    reject(&column, &one, &lower, "C is A shifted by one row");

    let before_other = other.raw().to_vec();
    let folded = TensorView::<T>::new(&other.buffer, 0, vec![4, 4], vec![0, 1]);
    let source = view_of::<T>(&matrix.buffer, &block(0, 0, 4, 4));
    reject(&source, &source, &folded, "C repeats its rows");
    let diagonal = TensorView::<T>::new(&other.buffer, 0, vec![4, 4], vec![1, 1]);
    reject(&source, &source, &diagonal, "C rows overlap each other");
    assert_untouched("rejected overlaps", &before_other, other.raw(), &[]);
    let apart_written: Vec<usize> = (0..n * n).map(|i| (n + i / n) * 2 * n + n + i % n).collect();
    assert_untouched("rejected overlaps", &before, matrix.raw(), &apart_written);
}

/// Invalid metadata returns `InvalidArgument` without panicking or writing.
fn invalid_metadata<T: Scalar>(ctx: &MetalContext) {
    let mut rng = ChaCha8Rng::seed_from_u64(1900);
    let a = Tensor::<T>::random(ctx, Layout::dense(&[4, 6]), &mut rng);
    let b = Tensor::<T>::random(ctx, Layout::dense(&[6, 5]), &mut rng);
    let c = Tensor::<T>::random(ctx, Layout::dense(&[4, 5]), &mut rng);
    let c0 = c.raw().to_vec();
    let (av, bv, cv) = (a.view(), b.view(), c.view());
    let invalid = |result: linalg::abi::Result<()>, what: &str| {
        assert!(matches!(result, Err(LinalgError::InvalidArgument(_))), "{what}: got {result:?}");
    };
    let labelled = |la, lb, lc| contract(ctx, 1.0, (&av, la), (&bv, lb), 0.0, (&cv, lc));
    assert_eq!(labelled("mk", "kn", "mn"), Ok(()));
    let c1 = c.raw().to_vec();
    invalid(labelled("mk", "kn", "m"), "too few labels for C");
    invalid(labelled("mkx", "kn", "mn"), "too many labels for A");
    invalid(labelled("mk", "jn", "mn"), "labels j and k appear once");
    invalid(labelled("mk", "kn", "mx"), "label x only in C");
    invalid(labelled("kk", "kn", "mn"), "repeated label in A");
    invalid(labelled("mk", "kn", "nm"), "extent mismatch through swapped C labels");
    invalid(labelled("mk", "nk", "mn"), "extent mismatch between A and B");

    let short = TensorView::<T>::row_major(&a.buffer, 0, vec![4, 7]);
    invalid(gemm(ctx, 1.0, &short, &bv, 0.0, &cv), "inner extents differ");
    let vector = TensorView::<T>::row_major(&a.buffer, 0, vec![6]);
    invalid(gemm(ctx, 1.0, &vector, &bv, 0.0, &cv), "rank below 2");
    let stacked = TensorView::<T>::row_major(&a.buffer, 0, vec![1, 4, 6]);
    invalid(gemm(ctx, 1.0, &stacked, &bv, 0.0, &cv), "ranks differ");
    let capacity = a.buffer.length() as usize / std::mem::size_of::<T>();
    let beyond = TensorView::<T>::row_major(&a.buffer, capacity - 23, vec![4, 6]);
    invalid(gemm(ctx, 1.0, &beyond, &bv, 0.0, &cv), "A reaches past its buffer");
    let wrapped = TensorView::<T>::new(&c.buffer, 0, vec![4, 5], vec![usize::MAX / 2, 1]);
    invalid(gemm(ctx, 1.0, &av, &bv, 0.0, &wrapped), "C strides overflow");
    let ragged = TensorView::<T>::new(&c.buffer, 0, vec![4, 5], vec![5]);
    invalid(gemm(ctx, 1.0, &av, &bv, 0.0, &ragged), "shape and strides disagree");

    // Five M axes whose strides differ between A and C cannot merge below five.
    let extents = [2usize, 2, 2, 2, 2, 3];
    let wide_a = Tensor::<T>::random(ctx, Layout::dense(&extents), &mut rng);
    let wide_b = Tensor::<T>::random(ctx, Layout::dense(&[3, 2]), &mut rng);
    let wide_c = Tensor::<T>::random(
        ctx,
        Layout::with(&[2, 2, 2, 2, 2, 2], &[1, 3, 0, 4, 2, 5], &[1; 6], 0),
        &mut rng,
    );
    let (wa, wb, wc) = (wide_a.view(), wide_b.view(), wide_c.view());
    let wide = contract(ctx, 1.0, (&wa, "abcdek"), (&wb, "kn"), 0.0, (&wc, "abcden"));
    invalid(wide, "five unmergeable M axes");

    let cmd = ctx.queue.new_command_buffer();
    invalid(encode_gemm(ctx, cmd, 1.0, &short, &bv, 0.0, &cv), "encode with bad extents");
    invalid(
        encode_contract(ctx, cmd, 1.0, (&av, "mk"), (&bv, "kn"), 0.0, (&cv, "m")),
        "encode with bad labels",
    );
    cmd.commit();
    cmd.wait_until_completed();
    assert_untouched("invalid metadata", &c1, c.raw(), &[]);
    assert_ne!(c0, c1, "the valid call must have written C");
}

/// Several encodes share one command buffer; later ones read what earlier ones wrote, and
/// nothing is read on the host until the buffer completes.
fn shared_command_buffer<T: Scalar>(ctx: &MetalContext) {
    let (m, k, n, p) = (37, 29, 43, 3);
    let mut rng = ChaCha8Rng::seed_from_u64(2000);
    let a = Tensor::<T>::random(ctx, Layout::dense(&[m, k]), &mut rng);
    let b = Tensor::<T>::random(ctx, Layout::dense(&[k, n]), &mut rng);
    let d = Tensor::<T>::random(ctx, Layout::transposed(n, p), &mut rng);
    let long = Tensor::<T>::random(ctx, Layout::dense(&[2, 40_000]), &mut rng);
    let ab = Tensor::<T>::random(ctx, Layout::dense(&[m, n]), &mut rng);
    let abd = Tensor::<T>::random(ctx, Layout::dense(&[m, p]), &mut rng);
    let gram = Tensor::<T>::random(ctx, Layout::dense(&[2, 2]), &mut rng);
    let (ab0, abd0, gram0) = (to_f64s(ab.raw()), to_f64s(abd.raw()), to_f64s(gram.raw()));

    let cmd = ctx.queue.new_command_buffer();
    let (av, bv, dv, lv) = (a.view(), b.view(), d.view(), long.view());
    let (abv, abdv, gv) = (ab.view(), abd.view(), gram.view());
    encode_gemm(ctx, cmd, 1.0, &av, &bv, 0.0, &abv).unwrap();
    encode_contract(ctx, cmd, 0.5, (&abv, "mn"), (&dv, "np"), -0.5, (&abdv, "mp")).unwrap();
    encode_contract(ctx, cmd, 1.0, (&lv, "ik"), (&lv, "jk"), 1.0, (&gv, "ij")).unwrap();
    assert_eq!(to_f64s(ab.raw()), ab0, "encoding must not run or write anything");
    cmd.commit();
    cmd.wait_until_completed();

    let op = |data, layout, labels| Operand64 { data, layout, labels };
    let (a64, b64, d64, l64) =
        (to_f64s(a.raw()), to_f64s(b.raw()), to_f64s(d.raw()), to_f64s(long.raw()));
    let ab1 = to_f64s(ab.raw());
    let first = reference(
        1.0,
        op(&a64, &a.layout, "mk"),
        op(&b64, &b.layout, "kn"),
        0.0,
        op(&ab0, &ab.layout, "mn"),
    );
    check("shared buffer: A·B", ab.raw(), &first);
    let second = reference(
        0.5,
        op(&ab1, &ab.layout, "mn"),
        op(&d64, &d.layout, "np"),
        -0.5,
        op(&abd0, &abd.layout, "mp"),
    );
    check("shared buffer: (A·B)·D", abd.raw(), &second);
    let third = reference(
        1.0,
        op(&l64, &long.layout, "ik"),
        op(&l64, &long.layout, "jk"),
        1.0,
        op(&gram0, &gram.layout, "ij"),
    );
    check("shared buffer: Gram matrix", gram.raw(), &third);
}

macro_rules! typed_tests {
    ($($case:ident),* $(,)?) => {
        mod f32_cases {
            $(#[test] fn $case() { super::$case::<f32>(&super::context()); })*
        }
        mod f16_cases {
            $(#[test] fn $case() { super::$case::<half::f16>(&super::context()); })*
        }
    };
}

typed_tests!(
    square,
    unit_extents,
    zero_k,
    empty_output,
    prime_sizes,
    transposed_views,
    strided_views,
    batched,
    rank4_two_axes,
    collapsible_axes,
    beta_variants,
    split_k,
    deterministic,
    non_finite_inputs,
    alpha_zero,
    lu_trailing_update,
    in_place_scale,
    overlap_rejected,
    invalid_metadata,
    shared_command_buffer,
);

/// A randomly generated contraction: extents per axis group plus a seed for the layouts.
#[derive(Clone, Debug)]
struct RandomCase {
    /// Extents of the batch, M, N, and K axes.
    groups: [Vec<usize>; 4],
    layout_seed: u64,
    alpha: f32,
    beta: f32,
}

fn random_case() -> impl Strategy<Value = RandomCase> {
    let extent = prop_oneof![
        1 => Just(0usize),
        2 => Just(1usize),
        8 => 2usize..=6,
        1 => 17usize..=40,
    ];
    let group = || proptest::collection::vec(extent.clone(), 0..=2);
    let alpha = prop_oneof![Just(1.0f32), Just(-0.75f32), Just(0.0f32)];
    let beta = prop_oneof![Just(0.0f32), Just(1.0f32), Just(-0.5f32)];
    (group(), group(), group(), group(), any::<u64>(), alpha, beta).prop_map(
        |(batch, m, n, k, layout_seed, alpha, beta)| RandomCase {
            groups: [batch, m, n, k],
            layout_seed,
            alpha,
            beta,
        },
    )
}

/// Builds a random layout for an operand carrying `labels`: shuffled logical axis order,
/// shuffled memory nesting, steps of 1 or 2, and a small offset.
fn random_operand(labels: &[(char, usize)], rng: &mut ChaCha8Rng) -> (Layout, String) {
    let mut logical = labels.to_vec();
    let mut order: Vec<usize> = (0..logical.len()).collect();
    for i in (1..logical.len()).rev() {
        logical.swap(i, rng.random_range(0..=i));
        order.swap(i, rng.random_range(0..=i));
    }
    let shape: Vec<usize> = logical.iter().map(|(_, n)| *n).collect();
    let steps: Vec<usize> = shape.iter().map(|_| rng.random_range(1..=2)).collect();
    let layout = Layout::with(&shape, &order, &steps, rng.random_range(0..8));
    (layout, logical.iter().map(|(l, _)| *l).collect())
}

fn run_random<T: Scalar>(ctx: &MetalContext, case: &RandomCase) {
    let mut groups = case.groups.clone();
    // Keep the reference affordable: shrink the largest extent until the work is bounded.
    while groups.iter().flatten().map(|n| (*n).max(1)).product::<usize>() > 400_000 {
        let largest = groups.iter_mut().flatten().max().expect("non-empty");
        *largest = largest.div_ceil(2);
    }
    let mut names = "abcdefgh".chars();
    let labelled: Vec<Vec<(char, usize)>> = groups
        .iter()
        .map(|g| g.iter().map(|n| (names.next().expect("at most 8 axes"), *n)).collect())
        .collect();
    let (batch, m, n, k) = (&labelled[0], &labelled[1], &labelled[2], &labelled[3]);
    let mut rng = ChaCha8Rng::seed_from_u64(case.layout_seed);
    let (a, la) = random_operand(&[batch.clone(), m.clone(), k.clone()].concat(), &mut rng);
    let (b, lb) = random_operand(&[batch.clone(), n.clone(), k.clone()].concat(), &mut rng);
    let (c, lc) = random_operand(&[batch.clone(), m.clone(), n.clone()].concat(), &mut rng);
    let name = format!("random {la},{lb}->{lc} {:?}", groups);
    let case = Case::new(name, (a, &la), (b, &lb), (c, &lc)).scalars(case.alpha, case.beta);
    case.run::<T>(ctx, rng.random());
}

fn random_contractions<T: Scalar>(cases: u32) {
    let ctx = context();
    let config = Config { cases, failure_persistence: None, ..Config::default() };
    let mut runner =
        TestRunner::new_with_rng(config, TestRng::deterministic_rng(RngAlgorithm::ChaCha));
    runner
        .run(&random_case(), |case| {
            run_random::<T>(&ctx, &case);
            Ok(())
        })
        .unwrap();
}

#[test]
fn random_contractions_f32() {
    random_contractions::<f32>(96);
}

#[test]
fn random_contractions_f16() {
    random_contractions::<f16>(96);
}

/// Exhaustive shape sweep around the tile boundaries, cycling through scalar pairs.
fn sweep_shapes<T: Scalar>() {
    let ctx = context();
    let extents = [1usize, 2, 7, 8, 9, 31, 32, 33, 63, 64, 65, 70];
    let inner = [0usize, 1, 2, 7, 8, 9, 31, 32, 33, 63, 64, 65, 100, 129];
    let scalars = [(1.0f32, 0.0f32), (-0.75, 1.0), (0.5, -0.5), (2.0, 0.0)];
    let (mut seed, mut worst, mut worst_gate) = (0u64, 0.0f64, 0.0f64);
    for m in extents {
        for n in extents {
            for k in inner {
                let (alpha, beta) = scalars[(seed % 4) as usize];
                let stats = Case::matmul(m, n, k).scalars(alpha, beta).run::<T>(&ctx, 5000 + seed);
                worst = worst.max(stats.normalized);
                worst_gate = worst_gate.max(stats.gate);
                seed += 1;
            }
        }
    }
    println!(
        "[sweep] shapes [{}]: {seed} cases, worst err/(sqrtK*u) = {worst:.2}, \
         worst gate ratio = {worst_gate:.3}",
        T::NAME
    );
}

/// Sweep of small outputs against long inner extents, the Split-K regime.
fn sweep_split_k<T: Scalar>() {
    let ctx = context();
    let (mut seed, mut worst, mut worst_gate) = (0u64, 0.0f64, 0.0f64);
    for m in [1usize, 3, 8, 17, 33] {
        for n in [1usize, 3, 8, 17, 33] {
            for k in [1_000usize, 4_097, 50_000, 262_145] {
                let beta = [0.0f32, -0.5][(seed % 2) as usize];
                let stats = Case::matmul(m, n, k).scalars(1.0, beta).run::<T>(&ctx, 6000 + seed);
                worst = worst.max(stats.normalized);
                worst_gate = worst_gate.max(stats.gate);
                seed += 1;
            }
        }
    }
    println!(
        "[sweep] split-K [{}]: {seed} cases, worst err/(sqrtK*u) = {worst:.2}, \
         worst gate ratio = {worst_gate:.3}",
        T::NAME
    );
}

#[test]
#[ignore = "exhaustive: run under the exclusive GPU lock"]
fn sweep_shapes_f32() {
    sweep_shapes::<f32>();
}

#[test]
#[ignore = "exhaustive: run under the exclusive GPU lock"]
fn sweep_shapes_f16() {
    sweep_shapes::<f16>();
}

#[test]
#[ignore = "exhaustive: run under the exclusive GPU lock"]
fn sweep_split_k_f32() {
    sweep_split_k::<f32>();
}

#[test]
#[ignore = "exhaustive: run under the exclusive GPU lock"]
fn sweep_split_k_f16() {
    sweep_split_k::<f16>();
}

/// `MPSDataTypeFloat32`: the float bit ORed with the bit width.
const MPS_DATA_TYPE_FLOAT32: u32 = 0x1000_0000 | 32;

/// Wraps a row-major `rows × cols` f32 buffer as an `MPSMatrix` (benchmark baseline only).
///
/// # Safety
/// `buffer` must hold at least `rows * cols` f32 elements; the caller releases the result.
unsafe fn mps_matrix(buffer: &metal::BufferRef, rows: usize, cols: usize) -> *mut Object {
    let descriptor: *mut Object = msg_send![
        class!(MPSMatrixDescriptor),
        matrixDescriptorWithRows: rows as u64
        columns: cols as u64
        rowBytes: (cols * 4) as u64
        dataType: MPS_DATA_TYPE_FLOAT32
    ];
    let matrix: *mut Object = msg_send![class!(MPSMatrix), alloc];
    msg_send![matrix, initWithBuffer: buffer.as_ptr() descriptor: descriptor]
}

/// Creates an `MPSMatrixMultiplication` kernel computing `C = A·B` for `m × k` by `k × n`.
///
/// # Safety
/// The caller releases the result.
unsafe fn mps_multiplication(ctx: &MetalContext, m: usize, n: usize, k: usize) -> *mut Object {
    let kernel: *mut Object = msg_send![class!(MPSMatrixMultiplication), alloc];
    msg_send![
        kernel,
        initWithDevice: ctx.device.as_ptr()
        transposeLeft: NO
        transposeRight: NO
        resultRows: m as u64
        resultColumns: n as u64
        interiorColumns: k as u64
        alpha: 1.0f64
        beta: 0.0f64
    ]
}

/// Runs `encode` in a fresh command buffer `reps` times after one warm-up and returns the
/// best and median wall-clock seconds from commit to completion.
fn time_runs(
    ctx: &MetalContext,
    reps: usize,
    encode: impl Fn(&metal::CommandBufferRef),
) -> (f64, f64) {
    let mut times = Vec::with_capacity(reps);
    for rep in 0..=reps {
        let cmd = ctx.queue.new_command_buffer();
        encode(cmd);
        let start = Instant::now();
        cmd.commit();
        cmd.wait_until_completed();
        if rep > 0 {
            times.push(start.elapsed().as_secs_f64());
        }
    }
    times.sort_by(f64::total_cmp);
    (times[0], times[times.len() / 2])
}

/// Gates `samples` randomly chosen entries of a dense `m × n` product against the unrounded
/// f64 reference and returns the worst `error / (√K·u)`.
fn spot_check(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    (m, n, k): (usize, usize, usize),
    samples: usize,
) -> f64 {
    let mut rng = ChaCha8Rng::seed_from_u64(0x5eed);
    let mut worst = 0.0f64;
    for _ in 0..samples {
        let (i, j) = (rng.random_range(0..m), rng.random_range(0..n));
        let (mut sum, mut abs) = (0.0f64, 0.0f64);
        for l in 0..k {
            let product = f64::from(a[i * k + l]) * f64::from(b[l * n + j]);
            sum += product;
            abs += product.abs();
        }
        let err = (f64::from(c[i * n + j]) - sum).abs();
        let tol = (k as f64 + 2.0) * U * abs + U * sum.abs();
        assert!(err <= tol, "bench entry ({i}, {j}): error {err:e} exceeds tolerance {tol:e}");
        worst = worst.max(err / ((k as f64).sqrt() * U));
    }
    worst
}

/// The spec benchmark: f32 square GEMM at 1024, 2048, and 4096 against
/// `MPSMatrixMultiplication`, in GFLOP/s (`2n³` flops over the best wall-clock time).
#[test]
#[ignore = "benchmark: run under the exclusive GPU lock"]
fn bench_square_f32_vs_mps() {
    let ctx = context();
    for (n, reps) in [(1024usize, 30usize), (2048, 12), (4096, 6)] {
        let mut rng = ChaCha8Rng::seed_from_u64(n as u64);
        let a = Tensor::<f32>::random(&ctx, Layout::dense(&[n, n]), &mut rng);
        let b = Tensor::<f32>::random(&ctx, Layout::dense(&[n, n]), &mut rng);
        let c = Tensor::<f32>::new(&ctx, Layout::dense(&[n, n]));
        let c_mps = Tensor::<f32>::new(&ctx, Layout::dense(&[n, n]));
        let (av, bv, cv) = (a.view(), b.view(), c.view());
        let ours =
            time_runs(&ctx, reps, |cmd| encode_gemm(&ctx, cmd, 1.0, &av, &bv, 0.0, &cv).unwrap());
        // SAFETY: the buffers hold n×n f32 matrices; every object is released below.
        let mps = unsafe {
            let (ma, mb, mc) = (
                mps_matrix(&a.buffer, n, n),
                mps_matrix(&b.buffer, n, n),
                mps_matrix(&c_mps.buffer, n, n),
            );
            let kernel = mps_multiplication(&ctx, n, n, n);
            let timing = time_runs(&ctx, reps, |cmd| {
                let _: () = msg_send![kernel, encodeToCommandBuffer: cmd.as_ptr() leftMatrix: ma rightMatrix: mb resultMatrix: mc];
            });
            for object in [kernel, ma, mb, mc] {
                let _: () = msg_send![object, release];
            }
            timing
        };
        let ours_error = spot_check(a.raw(), b.raw(), c.raw(), (n, n, n), 512);
        let mps_error = spot_check(a.raw(), b.raw(), c_mps.raw(), (n, n, n), 512);
        let gflops = |seconds: f64| 2.0 * (n as f64).powi(3) / seconds / 1e9;
        println!(
            "[bench] f32 {n}x{n}x{n}: linalg {:.1} GFLOP/s (median {:.1}), MPS {:.1} GFLOP/s (median {:.1}), \
             linalg/MPS = {:.1}%; err/(sqrtK*u): linalg {ours_error:.2}, MPS {mps_error:.2}",
            gflops(ours.0),
            gflops(ours.1),
            gflops(mps.0),
            gflops(mps.1),
            100.0 * mps.0 / ours.0,
        );
    }
}

/// f16 square GEMM at 1024, 2048, and 4096 next to f32 on the same shapes, in GFLOP/s.
/// There is no spec target; both types run the same `simdgroup_matrix` tiled kernel, so
/// f16 should stay within a small factor of f32.
#[test]
#[ignore = "benchmark: run under the exclusive GPU lock"]
fn bench_square_f16() {
    fn run<T: Scalar>(ctx: &MetalContext, n: usize, reps: usize) -> (f64, f64) {
        let mut rng = ChaCha8Rng::seed_from_u64(n as u64);
        let a = Tensor::<T>::random(ctx, Layout::dense(&[n, n]), &mut rng);
        let b = Tensor::<T>::random(ctx, Layout::dense(&[n, n]), &mut rng);
        let c = Tensor::<T>::new(ctx, Layout::dense(&[n, n]));
        let (av, bv, cv) = (a.view(), b.view(), c.view());
        let (best, median) =
            time_runs(ctx, reps, |cmd| encode_gemm(ctx, cmd, 1.0, &av, &bv, 0.0, &cv).unwrap());
        // Gate a sample of entries against the f64 reference so the timing is of a right answer.
        let (a64, b64) = (to_f64s(a.raw()), to_f64s(b.raw()));
        let mut rng = ChaCha8Rng::seed_from_u64(0x5eed);
        for _ in 0..256 {
            let (i, j) = (rng.random_range(0..n), rng.random_range(0..n));
            let (mut sum, mut abs) = (0.0f64, 0.0f64);
            for l in 0..n {
                let product = a64[i * n + l] * b64[l * n + j];
                sum += product;
                abs += product.abs();
            }
            let err = (c.raw()[i * n + j].to_f64() - sum).abs();
            let tol = (n as f64 + 2.0) * U * abs + T::U_OUT * sum.abs() + T::HALF_TINY;
            assert!(err <= tol, "bench entry ({i}, {j}): error {err:e} exceeds tolerance {tol:e}");
        }
        let gflops = |seconds: f64| 2.0 * (n as f64).powi(3) / seconds / 1e9;
        (gflops(best), gflops(median))
    }
    let ctx = context();
    for (n, reps) in [(1024usize, 30usize), (2048, 12), (4096, 6)] {
        let half = run::<f16>(&ctx, n, reps);
        let single = run::<f32>(&ctx, n, reps);
        println!(
            "[bench] square {n}x{n}x{n}: f16 {:.1} GFLOP/s (median {:.1}), f32 {:.1} GFLOP/s \
             (median {:.1}), f16/f32 = {:.1}% (median {:.1}%)",
            half.0,
            half.1,
            single.0,
            single.1,
            100.0 * half.0 / single.0,
            100.0 * half.1 / single.1,
        );
    }
}

/// Throughput of the remaining kernel paths: f16, batched small matrices, long-K dot
/// products (Split-K), rank-1 updates, and strided operands.
#[test]
#[ignore = "benchmark: run under the exclusive GPU lock"]
fn bench_other_paths() {
    fn bench<T: Scalar>(
        ctx: &MetalContext,
        name: &str,
        a: (Layout, &str),
        b: (Layout, &str),
        c: (Layout, &str),
        reps: usize,
    ) {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let work: usize = 2 * [a.1, b.1, c.1]
            .concat()
            .chars()
            .zip([&a.0.shape[..], &b.0.shape[..], &c.0.shape[..]].concat())
            .collect::<HashMap<char, usize>>()
            .values()
            .product::<usize>();
        let ta = Tensor::<T>::random(ctx, a.0, &mut rng);
        let tb = Tensor::<T>::random(ctx, b.0, &mut rng);
        let tc = Tensor::<T>::new(ctx, c.0);
        let (av, bv, cv) = (ta.view(), tb.view(), tc.view());
        let (best, median) = time_runs(ctx, reps, |cmd| {
            encode_contract(ctx, cmd, 1.0, (&av, a.1), (&bv, b.1), 0.0, (&cv, c.1)).unwrap()
        });
        println!(
            "[bench] {name} [{}]: {:.1} GFLOP/s (median {:.1}), best {:.3} ms",
            T::NAME,
            work as f64 / best / 1e9,
            work as f64 / median / 1e9,
            best * 1e3
        );
    }
    let ctx = context();
    let dense = |shape: &[usize]| Layout::dense(shape);
    bench::<f16>(
        &ctx,
        "square 2048",
        (dense(&[2048, 2048]), "mk"),
        (dense(&[2048, 2048]), "kn"),
        (dense(&[2048, 2048]), "mn"),
        8,
    );
    bench::<f32>(
        &ctx,
        "square 512",
        (dense(&[512, 512]), "mk"),
        (dense(&[512, 512]), "kn"),
        (dense(&[512, 512]), "mn"),
        30,
    );
    bench::<f32>(
        &ctx,
        "transposed 2048",
        (Layout::transposed(2048, 2048), "mk"),
        (Layout::transposed(2048, 2048), "kn"),
        (Layout::transposed(2048, 2048), "mn"),
        8,
    );
    bench::<f32>(
        &ctx,
        "batched 4096 x 32^3",
        (dense(&[4096, 32, 32]), "bmk"),
        (dense(&[4096, 32, 32]), "bkn"),
        (dense(&[4096, 32, 32]), "bmn"),
        20,
    );
    bench::<f32>(
        &ctx,
        "dot K=16M",
        (dense(&[1, 1 << 24]), "mk"),
        (dense(&[1 << 24, 1]), "kn"),
        (dense(&[1, 1]), "mn"),
        10,
    );
    bench::<f32>(
        &ctx,
        "32x32 K=1M",
        (dense(&[32, 1 << 20]), "mk"),
        (Layout::transposed(1 << 20, 32), "kn"),
        (dense(&[32, 32]), "mn"),
        10,
    );
    bench::<f32>(
        &ctx,
        "rank-1 4096^2",
        (dense(&[4096, 1]), "mk"),
        (dense(&[1, 4096]), "kn"),
        (dense(&[4096, 4096]), "mn"),
        20,
    );
    bench::<f32>(
        &ctx,
        "rank-4 ijkl,klmn->ijmn 48^6",
        (dense(&[48, 48, 48, 48]), "ijkl"),
        (dense(&[48, 48, 48, 48]), "klmn"),
        (dense(&[48, 48, 48, 48]), "ijmn"),
        6,
    );
}

// --- Audit regression tests (CON3) -------------------------------------------------------

fn fill_f32(buf: &metal::BufferRef, len: usize, value: f32) {
    // SAFETY: the shared buffer holds at least `len` f32s and no GPU work is in flight.
    unsafe { std::slice::from_raw_parts_mut(buf.contents() as *mut f32, len).fill(value) };
}

fn read_f32s(buf: &metal::BufferRef, len: usize) -> Vec<f32> {
    // SAFETY: the shared buffer holds at least `len` f32s and the GPU has finished.
    unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, len).to_vec() }
}

/// CON3: an f32 result whose exact value is subnormal must meet the spec gate against the
/// unrounded f64 reference, on the tiled, Split-K, and K = 1 paths. The only allowance beyond
/// the spec formula is half the smallest subnormal: the rounding error any correctly
/// rounded f32 output has below the normal range.
#[test]
fn audit_f32_subnormal_results_meet_the_spec_gate() {
    let ctx = context();
    for (path, m, k, n) in
        [("tiled", 64usize, 64usize, 64usize), ("split-K", 2, 50_000, 2), ("K=1", 33, 1, 17)]
    {
        let (a_buf, b_buf) = (alloc_shared::<f32>(&ctx, m * k), alloc_shared::<f32>(&ctx, k * n));
        let c_buf = alloc_shared::<f32>(&ctx, m * n);
        let (a_val, b_val) = (1.0e-20f32, 1.0e-20f32 / k as f32);
        fill_f32(&a_buf, m * k, a_val);
        fill_f32(&b_buf, k * n, b_val);
        fill_f32(&c_buf, m * n, f32::NAN);
        let a = TensorView::<f32>::row_major(&a_buf, 0, vec![m, k]);
        let b = TensorView::<f32>::row_major(&b_buf, 0, vec![k, n]);
        let c = TensorView::<f32>::row_major(&c_buf, 0, vec![m, n]);
        gemm(&ctx, 1.0, &a, &b, 0.0, &c).unwrap();
        let exact = k as f64 * f64::from(a_val) * f64::from(b_val);
        let gate = (k as f64 + 2.0) * U * exact + U * exact + 0.5 * f64::from(f32::from_bits(1));
        for (i, got) in read_f32s(&c_buf, m * n).into_iter().enumerate() {
            let err = (f64::from(got) - exact).abs();
            assert!(
                err <= gate,
                "{path}: C[{i}] = {got:e}, f64 reference {exact:e}, err {err:e} > gate {gate:e}"
            );
        }
    }
}

/// CON2/CON3: with K = 0, C = β·C. β = 1 must leave a subnormal C bit-identical, and
/// β = 0.5 must give the exactly representable subnormal half.
#[test]
fn audit_k0_keeps_subnormal_c() {
    let ctx = context();
    let sub = f32::from_bits(0x0000_1234);
    for beta in [1.0f32, 0.5] {
        let (a_buf, b_buf) = (alloc_shared::<f32>(&ctx, 0), alloc_shared::<f32>(&ctx, 0));
        let c_buf = alloc_shared::<f32>(&ctx, 16);
        fill_f32(&c_buf, 16, sub);
        let a = TensorView::<f32>::row_major(&a_buf, 0, vec![4, 0]);
        let b = TensorView::<f32>::row_major(&b_buf, 0, vec![0, 4]);
        let c = TensorView::<f32>::row_major(&c_buf, 0, vec![4, 4]);
        gemm(&ctx, 1.0, &a, &b, beta, &c).unwrap();
        let want = (f64::from(beta) * f64::from(sub)) as f32; // exact: 0x1234 or 0x091a
        for (i, got) in read_f32s(&c_buf, 16).into_iter().enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "β={beta}: C[{i}] = {:#010x}, want {:#010x}",
                got.to_bits(),
                want.to_bits()
            );
        }
    }
}

fn fill_f16(buf: &metal::BufferRef, len: usize, value: f16) {
    // SAFETY: the shared buffer holds at least `len` f16s and no GPU work is in flight.
    unsafe { std::slice::from_raw_parts_mut(buf.contents() as *mut f16, len).fill(value) };
}

fn read_f16s(buf: &metal::BufferRef, len: usize) -> Vec<f16> {
    // SAFETY: the shared buffer holds at least `len` f16s and the GPU has finished.
    unsafe { std::slice::from_raw_parts(buf.contents() as *const f16, len).to_vec() }
}

/// CON3, f16 variant: a result whose exact value is a subnormal f16 must meet the spec gate
/// (`u_out = 2⁻¹¹`) against the unrounded f64 reference, on the tiled, Split-K, and K = 1
/// paths. The only extra allowance is half the smallest f16 subnormal, `2⁻²⁵`. The tiled
/// and Split-K cases also feed subnormal f16 inputs.
#[test]
fn audit_f16_subnormal_results_meet_the_spec_gate() {
    let ctx = context();
    let cases = [
        ("tiled", 64usize, 64usize, 64usize, 0.01f32, 1.5e-5f32),
        ("split-K", 2, 50_000, 2, 2.0e-4, 1.8e-7),
        ("K=1", 33, 1, 17, 0.01, 0.001),
    ];
    for (path, m, k, n, a_val, b_val) in cases {
        let (a_val, b_val) = (f16::from_f32(a_val), f16::from_f32(b_val));
        let (a_buf, b_buf) = (alloc_shared::<f16>(&ctx, m * k), alloc_shared::<f16>(&ctx, k * n));
        let c_buf = alloc_shared::<f16>(&ctx, m * n);
        fill_f16(&a_buf, m * k, a_val);
        fill_f16(&b_buf, k * n, b_val);
        fill_f16(&c_buf, m * n, f16::NAN);
        let a = TensorView::<f16>::row_major(&a_buf, 0, vec![m, k]);
        let b = TensorView::<f16>::row_major(&b_buf, 0, vec![k, n]);
        let c = TensorView::<f16>::row_major(&c_buf, 0, vec![m, n]);
        gemm(&ctx, 1.0, &a, &b, 0.0, &c).unwrap();
        let exact = k as f64 * a_val.to_f64() * b_val.to_f64();
        assert!(
            exact > f16::from_bits(1).to_f64() && exact < f16::MIN_POSITIVE.to_f64(),
            "test bug: {path} reference {exact:e} is not a subnormal f16"
        );
        let gate = (k as f64 + 2.0) * U * exact + exact / 2048.0 + 0.5 * f16::from_bits(1).to_f64();
        for (i, got) in read_f16s(&c_buf, m * n).into_iter().enumerate() {
            let err = (got.to_f64() - exact).abs();
            assert!(
                err <= gate,
                "{path}: C[{i}] = {:e}, f64 reference {exact:e}, err {err:e} > gate {gate:e}",
                got.to_f64()
            );
        }
    }
}

/// CON2/CON3, f16 variant: with K = 0, C = β·C. β = 1 must leave a subnormal f16 C
/// bit-identical, and β = 2 must give the exactly representable subnormal double.
#[test]
fn audit_k0_keeps_subnormal_c_f16() {
    let ctx = context();
    let sub = f16::from_bits(0x0003);
    for (beta, want) in [(1.0f32, 0x0003u16), (2.0, 0x0006)] {
        let (a_buf, b_buf) = (alloc_shared::<f16>(&ctx, 0), alloc_shared::<f16>(&ctx, 0));
        let c_buf = alloc_shared::<f16>(&ctx, 16);
        fill_f16(&c_buf, 16, sub);
        let a = TensorView::<f16>::row_major(&a_buf, 0, vec![4, 0]);
        let b = TensorView::<f16>::row_major(&b_buf, 0, vec![0, 4]);
        let c = TensorView::<f16>::row_major(&c_buf, 0, vec![4, 4]);
        gemm(&ctx, 1.0, &a, &b, beta, &c).unwrap();
        for (i, got) in read_f16s(&c_buf, 16).into_iter().enumerate() {
            assert_eq!(
                got.to_bits(),
                want,
                "β={beta}: C[{i}] = {:#06x}, want {want:#06x}",
                got.to_bits()
            );
        }
    }
}

// --- Subnormal behaviour -----------------------------------------------------------------

/// One contraction on random operands scaled into a chosen magnitude range, gated like
/// every other case. `scales` multiplies the random A, B, and C₀.
#[allow(clippy::too_many_arguments)]
fn run_scaled(
    ctx: &MetalContext,
    name: &str,
    a: (Layout, &str),
    b: (Layout, &str),
    c: (Layout, &str),
    scales: (f64, f64, f64),
    (alpha, beta): (f32, f32),
    seed: u64,
) -> Stats {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut ta = Tensor::<f32>::random(ctx, a.0, &mut rng);
    let mut tb = Tensor::<f32>::random(ctx, b.0, &mut rng);
    let mut tc = Tensor::<f32>::random(ctx, c.0, &mut rng);
    for (tensor, scale) in [(&mut ta, scales.0), (&mut tb, scales.1), (&mut tc, scales.2)] {
        for x in tensor.raw_mut() {
            *x = (f64::from(*x) * scale) as f32;
        }
    }
    let (a0, b0, c0) = (ta.raw().to_vec(), tb.raw().to_vec(), tc.raw().to_vec());
    let expected = reference(
        alpha,
        Operand64 { data: &to_f64s(&a0), layout: &ta.layout, labels: a.1 },
        Operand64 { data: &to_f64s(&b0), layout: &tb.layout, labels: b.1 },
        beta,
        Operand64 { data: &to_f64s(&c0), layout: &tc.layout, labels: c.1 },
    );
    let (av, bv, cv) = (ta.view(), tb.view(), tc.view());
    assert_eq!(contract(ctx, alpha, (&av, a.1), (&bv, b.1), beta, (&cv, c.1)), Ok(()), "{name}");
    assert_untouched(name, &a0, ta.raw(), &[]);
    assert_untouched(name, &b0, tb.raw(), &[]);
    assert_untouched(name, &c0, tc.raw(), &expected.index);
    check(name, tc.raw(), &expected)
}

/// CON3: operands, products, `β·C`, and results in and around the f32 subnormal range meet
/// the same gate as everything else, on the tiled, Split-K, K = 1, and general paths.
#[test]
fn f32_subnormal_range_meets_the_gate() {
    let ctx = context();
    // (A scale, B scale, C₀ scale, α, β)
    let ranges = [
        // Every product is subnormal; so is the result.
        (1e-22, 1e-21, 1.0, 1.0f32, 0.0f32),
        // The same, added to a subnormal C.
        (1e-22, 1e-21, 1e-40, -0.75, 1.0),
        // Normal products and sums, subnormal α·acc and β·C.
        (1e-10, 1e-10, 1e-39, 3.0e-21, -0.5),
        // Subnormal A, large B: normal results from subnormal inputs.
        (1e-41, 1e30, 1e-11, 1.0, 0.7),
        // Normal products near the bottom of the range, cancelling into subnormal sums.
        (1e-19, 1e-19, 1e-38, 1.0, -1.0),
        // Normal everything, result scaled into the subnormal range by β and α.
        (1.0, 1.0, 1.0, 1.0e-40, 2.5e-41),
    ];
    let shapes = [(64usize, 64usize, 64usize), (70, 45, 50), (2, 2, 50_000), (33, 17, 1)];
    let mut seed = 2100;
    for (sa, sb, sc, alpha, beta) in ranges {
        for (m, n, k) in shapes {
            let name = format!("tiny {m}x{n}x{k} scales=({sa:e},{sb:e},{sc:e}) α={alpha} β={beta}");
            run_scaled(
                &ctx,
                &name,
                (Layout::dense(&[m, k]), "mk"),
                (Layout::dense(&[k, n]), "kn"),
                (Layout::dense(&[m, n]), "mn"),
                (sa, sb, sc),
                (alpha, beta),
                seed,
            );
            seed += 1;
        }
        let (i, j, k, l, m, n) = (5, 7, 6, 9, 4, 11);
        run_scaled(
            &ctx,
            &format!("tiny general scales=({sa:e},{sb:e},{sc:e}) α={alpha} β={beta}"),
            (Layout::with(&[i, j, k, l], &[2, 0, 3, 1], &[1, 1, 1, 1], 3), "ijkl"),
            (Layout::with(&[l, k, m, n], &[3, 0, 2, 1], &[1, 2, 1, 1], 0), "lkmn"),
            (Layout::with(&[i, m, j, n], &[1, 3, 0, 2], &[1, 1, 2, 1], 8), "imjn"),
            (sa, sb, sc),
            (alpha, beta),
            seed,
        );
        seed += 1;
    }
}

/// CON3: `C = β·C` (K = 0) is rounded once and never flushed, for subnormal C and for
/// products that land in the subnormal range, including ties and the smallest subnormal.
#[test]
fn scale_f32_is_correctly_rounded_in_the_subnormal_range() {
    let ctx = context();
    let values: Vec<f32> = [
        0x0000_0001u32,
        0x0000_0003,
        0x0000_1234,
        0x007f_ffff,
        0x0080_0000,
        0x0080_0001,
        0x00ff_ffff,
        0x0123_4567,
        0x3f80_0000,
        0x8000_0001,
        0x8000_1235,
        0x807f_ffff,
        0x8080_0003,
        0x0000_0000,
        0x8000_0000,
        0x7f7f_ffff,
    ]
    .into_iter()
    .map(f32::from_bits)
    .collect();
    let betas =
        [0.5f32, -0.5, 0.75, 1.5, -0.999, 3.0, 1.0e-3, 1.0e-30, 1.0e30, 2.0e-45, 1.2e-38, 0.25];
    for beta in betas {
        let (a_buf, b_buf) = (alloc_shared::<f32>(&ctx, 0), alloc_shared::<f32>(&ctx, 0));
        let c_buf = alloc_shared::<f32>(&ctx, values.len());
        // SAFETY: the shared buffer holds `values.len()` f32s and no GPU work is in flight.
        unsafe {
            std::slice::from_raw_parts_mut(c_buf.contents() as *mut f32, values.len())
                .copy_from_slice(&values);
        }
        let a = TensorView::<f32>::row_major(&a_buf, 0, vec![values.len(), 0]);
        let b = TensorView::<f32>::row_major(&b_buf, 0, vec![0, 1]);
        let c = TensorView::<f32>::row_major(&c_buf, 0, vec![values.len(), 1]);
        gemm(&ctx, 1.0, &a, &b, beta, &c).unwrap();
        for (got, value) in read_f32s(&c_buf, values.len()).into_iter().zip(&values) {
            let want = (f64::from(beta) * f64::from(*value)) as f32;
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "β={beta:e} · {:#010x}: got {:#010x}, want {:#010x}",
                value.to_bits(),
                got.to_bits(),
                want.to_bits()
            );
        }
    }
}

/// Test-only kernels that expose the raw float behaviour of the GPU.
const PROBE_SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

/// out[i] = a[i] * b[i] + c[i] as one scalar fused multiply-add.
kernel void contraction_probe_scalar(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* c [[buffer(2)]],
    device float* out [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    out[i] = fma(a[i], b[i], c[i]);
}

/// out[i] = c[i] + a[i] * b[i] for i < 8, computed by `simdgroup_multiply_accumulate` on
/// diagonal 8×8 matrices.
kernel void contraction_probe_mma(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* c [[buffer(2)]],
    device float* out [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float ta[64];
    threadgroup float tb[64];
    threadgroup float tc[64];
    for (uint e = lane; e < 64; e += 32) {
        uint row = e >> 3;
        bool diagonal = row == (e & 7);
        ta[e] = diagonal ? a[row] : 0.0f;
        tb[e] = diagonal ? b[row] : 0.0f;
        tc[e] = diagonal ? c[row] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix<float, 8, 8> ma, mb, mc;
    simdgroup_load(ma, ta, 8);
    simdgroup_load(mb, tb, 8);
    simdgroup_load(mc, tc, 8);
    simdgroup_multiply_accumulate(mc, ma, mb, mc);
    simdgroup_store(mc, tc, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 8) {
        out[lane] = tc[lane * 9];
    }
}
"#;

fn probe_options() -> metal::CompileOptions {
    let options = metal::CompileOptions::new();
    options.set_fast_math_enabled(false);
    options.set_language_version(metal::MTLLanguageVersion::V2_4);
    options
}

/// Pins the hardware behaviour the contraction kernels are built around: this GPU reads a
/// subnormal f32 operand as zero and flushes a subnormal f32 result to zero, both in scalar
/// arithmetic and inside `simdgroup_multiply_accumulate` (which is fused: one rounding per
/// `acc + a·b`). Normal-range results are IEEE. If this test starts failing because the
/// GPU honours subnormals, the rescaling pass in the kernels has become unnecessary.
#[test]
fn probe_subnormal_flush_is_pinned() {
    let ctx = context();
    let p2 = |e: i32| 2f32.powi(e);
    // (a, b, c, expected c + a·b on this GPU)
    let cases: [(f32, f32, f32, f32); 8] = [
        // A subnormal product is flushed (IEEE: 1e-40).
        (1.0e-20, 1.0e-20, 0.0, 0.0),
        // A subnormal operand is read as zero (IEEE: about 1e-30).
        (1.0e-40, 1.0e10, 0.0, 0.0),
        // A subnormal sum of normal terms is flushed (IEEE: 2^-127).
        (1.5 * p2(-126), 1.0, -p2(-126), 0.0),
        // A subnormal accumulator is read as zero (IEEE: 1e-40).
        (0.0, 0.0, 1.0e-40, 0.0),
        // Fused: the product is not rounded before the add (unfused gives 0).
        (1.0 + p2(-12), 1.0 + p2(-12), -(1.0 + p2(-11)), p2(-24)),
        // Normal range is IEEE.
        (3.0, 0.1, 1.0, 3.0f32.mul_add(0.1, 1.0)),
        (1.0e-20, 1.0e-17, 0.0, 1.0e-20f32 * 1.0e-17),
        (p2(-63), p2(-63), 0.0, p2(-126)),
    ];
    let n = cases.len();
    let buffers: Vec<metal::Buffer> = (0..4).map(|_| alloc_shared::<f32>(&ctx, n)).collect();
    for (slot, pick) in
        [(0usize, 0usize), (1, 1), (2, 2)].map(|(slot, field)| (slot, field)).into_iter()
    {
        let column: Vec<f32> = cases.iter().map(|c| [c.0, c.1, c.2][pick]).collect();
        // SAFETY: each shared buffer holds `n` f32s and no GPU work is in flight.
        unsafe {
            std::slice::from_raw_parts_mut(buffers[slot].contents() as *mut f32, n)
                .copy_from_slice(&column);
        }
    }
    for function in ["contraction_probe_scalar", "contraction_probe_mma"] {
        fill_f32(&buffers[3], n, f32::NAN);
        let pipeline =
            compute_pipeline(&ctx, "contraction_probe", PROBE_SOURCE, probe_options, function)
                .unwrap();
        let cmd = ctx.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        for (index, buffer) in buffers.iter().enumerate() {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        if function == "contraction_probe_scalar" {
            encoder.dispatch_threads(
                metal::MTLSize::new(n as u64, 1, 1),
                metal::MTLSize::new(1, 1, 1),
            );
        } else {
            encoder.dispatch_thread_groups(
                metal::MTLSize::new(1, 1, 1),
                metal::MTLSize::new(32, 1, 1),
            );
        }
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        for (i, (got, case)) in read_f32s(&buffers[3], n).into_iter().zip(&cases).enumerate() {
            println!(
                "[probe] {function} case {i}: {:e} + {:e}·{:e} = {got:e} (IEEE {:e})",
                case.2,
                case.0,
                case.1,
                case.0.mul_add(case.1, case.2)
            );
            assert_eq!(
                got.to_bits(),
                case.3.to_bits(),
                "{function} case {i}: {:e} + {:e}·{:e} gave {got:e}, pinned {:e}",
                case.2,
                case.0,
                case.1,
                case.3
            );
        }
    }
}

/// Pins the one documented exception to the CON3 gate. Rescaling by powers of two can
/// lift a product into the normal range only as far as the largest product of the same
/// contraction allows. Here one contraction holds products of 1e37 and 1e-40: the scale
/// that keeps 1e37 finite leaves 1e-40 subnormal inside `simdgroup_multiply_accumulate`, so the GPU
/// flushes it and C[1][1] is 0 where the exact value, 1e-40, is a representable subnormal.
/// The spec gate has no such exception; this is an open spec question, and this test
/// records the behaviour so a change in either direction is noticed.
#[test]
fn subnormal_product_beyond_rescale_is_flushed() {
    let ctx = context();
    let (a_buf, b_buf, c_buf) =
        (alloc_shared::<f32>(&ctx, 4), alloc_shared::<f32>(&ctx, 4), alloc_shared::<f32>(&ctx, 4));
    let (a_vals, b_vals) = ([1.0e18f32, 0.0, 0.0, 1.0e-20], [1.0e19f32, 0.0, 0.0, 1.0e-20]);
    // SAFETY: each shared buffer holds four f32s and no GPU work is in flight.
    unsafe {
        std::slice::from_raw_parts_mut(a_buf.contents() as *mut f32, 4).copy_from_slice(&a_vals);
        std::slice::from_raw_parts_mut(b_buf.contents() as *mut f32, 4).copy_from_slice(&b_vals);
    }
    fill_f32(&c_buf, 4, f32::NAN);
    let a = TensorView::<f32>::row_major(&a_buf, 0, vec![2, 2]);
    let b = TensorView::<f32>::row_major(&b_buf, 0, vec![2, 2]);
    let c = TensorView::<f32>::row_major(&c_buf, 0, vec![2, 2]);
    gemm(&ctx, 1.0, &a, &b, 0.0, &c).unwrap();
    let got = read_f32s(&c_buf, 4);
    let exact = [1.0e18f64 as f32 as f64 * f64::from(1.0e19f32), 0.0, 0.0, 1.0e-40];
    println!("[contraction] beyond-rescale: got {got:?}, f64 reference {exact:?}");
    for i in 0..3 {
        let err = (f64::from(got[i]) - exact[i]).abs();
        assert!(err <= 4.0 * U * exact[i], "C[{i}] = {:e}, reference {:e}", got[i], exact[i]);
    }
    assert_eq!(
        got[3].to_bits(),
        0,
        "C[3] = {:e}: the product 1e-20·1e-20 next to 1e18·1e19 is expected to be flushed",
        got[3]
    );
}
