//! Tests for the `reciprocal` primitive.
//!
//! Every numerical check compares GPU output with an inline pure-Rust reference computed in
//! f64 and rounded once to f32:
//!
//! - Real: `(1.0 / x as f64) as f32` is the correctly rounded quotient (the double rounding
//!   is innocuous because 53 ≥ 2·24 + 2). The spec gate is 1 ULP; the kernel is correctly
//!   rounded for all 2³² inputs (`sweep_real_exhaustive`), so the tests pin 0 ULP.
//! - Complex: `conj(z) / |z|²` evaluated in f64, which neither overflows nor underflows for
//!   f32 inputs. The gate is a normwise relative error of at most 4u (u = 2⁻²⁴), plus an
//!   absolute floor of one subnormal quantum (2⁻¹⁴⁹) because results below 2⁻¹²⁶ cannot be
//!   represented to relative precision. Each component must also lie within
//!   [`COMPLEX_ULP_GATE`] ULPs of the correctly rounded reference.
//!
//! `sweep_*` and `bench_*` tests are `#[ignore]`d and run under the exclusive GPU lock.

use std::marker::PhantomData;
use std::mem::size_of;
use std::time::Instant;

use linalg::abi::{alloc_shared, compute_pipeline, LinalgError, MetalContext, TensorView};
use linalg::ops::reciprocal::{compile_options, encode_reciprocal, reciprocal, ReciprocalElement};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// Complex f32 as (re, im).
type C32 = [f32; 2];

/// Unit roundoff of f32.
const U: f64 = 1.0 / 16_777_216.0;
/// Largest allowed ULP distance of a real result from the correctly rounded reference.
const REAL_ULP_GATE: u64 = 0;
/// Largest allowed normwise relative error of a complex result, in units of u.
const COMPLEX_NORMWISE_GATE: f64 = 4.0;
/// Largest allowed ULP distance of each complex component from the rounded reference.
const COMPLEX_ULP_GATE: u64 = 4;
/// Absolute error floor of the complex gate: one subnormal quantum, 2⁻¹⁴⁹.
const SUBNORMAL_QUANTUM: f64 = 1.401298464324817e-45;
/// Complex results at least this large are far enough from the subnormal range that the
/// normwise error is purely relative; the reported worst error is taken over these.
const NORMAL_RANGE_FLOOR: f64 = 7.888609052210118e-31;
/// 2¹²⁸, the value an overflowed f32 stands for when measuring errors.
const TWO_POW_128: f64 = 3.402823669209385e38;

/// A shared Metal buffer exposing its first `len` elements as a host slice.
struct Shared<T> {
    buffer: metal::Buffer,
    len: usize,
    marker: PhantomData<T>,
}

impl<T: Copy> Shared<T> {
    fn zeroed(ctx: &MetalContext, len: usize) -> Self {
        Self { buffer: alloc_shared::<T>(ctx, len), len, marker: PhantomData }
    }

    fn from_slice(ctx: &MetalContext, data: &[T]) -> Self {
        let mut shared = Self::zeroed(ctx, data.len());
        shared.slice_mut().copy_from_slice(data);
        shared
    }

    /// Host view of the elements. Only called while no command buffer is in flight.
    fn slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.buffer.contents() as *const T, self.len) }
    }

    fn slice_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.buffer.contents() as *mut T, self.len) }
    }

    /// Number of elements the rounded-up allocation can hold.
    fn capacity(&self) -> usize {
        self.buffer.length() as usize / size_of::<T>()
    }

    fn vector(&self) -> TensorView<'_, T> {
        TensorView::row_major(&self.buffer, 0, vec![self.len])
    }

    fn view(&self, offset: usize, shape: &[usize], strides: &[usize]) -> TensorView<'_, T> {
        TensorView::new(&self.buffer, offset, shape.to_vec(), strides.to_vec())
    }
}

/// A strided window: element `idx` lives at `offset + Σ idx[d] * strides[d]`.
#[derive(Clone, Debug)]
struct Layout {
    offset: usize,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl Layout {
    fn new(offset: usize, shape: &[usize], strides: &[usize]) -> Self {
        Self { offset, shape: shape.to_vec(), strides: strides.to_vec() }
    }

    /// Diagonals of `batch` column-major `n × n` matrices with leading dimension `ld`,
    /// stored back to back starting at element `offset`.
    fn diagonal(ld: usize, n: usize, batch: usize, offset: usize) -> Self {
        Self::new(offset, &[batch, n], &[ld * n, ld + 1])
    }

    /// Element indices in row-major order of the multi-index.
    fn indices(&self) -> Vec<usize> {
        let mut indices = vec![self.offset];
        for (&extent, &stride) in self.shape.iter().zip(&self.strides) {
            indices = indices
                .iter()
                .flat_map(|&base| (0..extent).map(move |i| base + i * stride))
                .collect();
        }
        indices
    }

    /// Smallest buffer length that holds every element.
    fn extent(&self) -> usize {
        self.indices().iter().max().map_or(self.offset, |&last| last + 1)
    }

    fn view<'a, T: Copy>(&self, buffer: &'a Shared<T>) -> TensorView<'a, T> {
        buffer.view(self.offset, &self.shape, &self.strides)
    }
}

/// Reciprocal of `input` computed by the GPU, out of place between contiguous vectors.
fn gpu_reciprocal<T: ReciprocalElement>(ctx: &MetalContext, input: &[T]) -> Vec<T> {
    let src = Shared::from_slice(ctx, input);
    let dst = Shared::<T>::zeroed(ctx, input.len());
    reciprocal(ctx, &src.vector(), &dst.vector()).unwrap();
    dst.slice().to_vec()
}

/// Correctly rounded real reciprocal.
fn reference_real(x: f32) -> f32 {
    (1.0 / x as f64) as f32
}

/// Reciprocal of a finite nonzero complex f32, accurate to f64 roundoff.
fn reference_complex(z: C32) -> [f64; 2] {
    let (a, b) = (z[0] as f64, z[1] as f64);
    let norm_sqr = a * a + b * b;
    [a / norm_sqr, -b / norm_sqr]
}

/// Distance in representable f32 values, with ∞ adjacent to `f32::MAX`. NaN matches only
/// NaN, and the signs (including the sign of zero) must agree; otherwise `u64::MAX`.
fn ulp_distance(got: f32, want: f32) -> u64 {
    if got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()) {
        return 0;
    }
    if got.is_nan() || want.is_nan() || got.is_sign_negative() != want.is_sign_negative() {
        return u64::MAX;
    }
    u64::from((got.to_bits() & 0x7fff_ffff).abs_diff(want.to_bits() & 0x7fff_ffff))
}

/// `distance <= REAL_ULP_GATE`. Kept as a function because an `allow` on `assert!` is ignored,
/// and the gate is 0 today, which makes `<=` look like an equality to clippy.
#[allow(clippy::absurd_extreme_comparisons)]
fn within_real_gate(distance: u64) -> bool {
    distance <= REAL_ULP_GATE
}

/// Asserts the real gate for one input.
fn assert_real(x: f32, got: f32) {
    let want = reference_real(x);
    assert!(
        within_real_gate(ulp_distance(got, want)),
        "1/{x:e} ({:#010x}): got {:#010x}, want {:#010x}",
        x.to_bits(),
        got.to_bits(),
        want.to_bits()
    );
}

/// Runs the GPU on `inputs` and asserts the real gate on every element.
fn check_real(ctx: &MetalContext, inputs: &[f32]) {
    let outputs = gpu_reciprocal(ctx, inputs);
    for (&x, &y) in inputs.iter().zip(&outputs) {
        assert_real(x, y);
    }
}

/// Error of one complex result: absolute normwise error, the exact result's modulus, and
/// the larger componentwise ULP distance from the rounded reference.
struct ComplexError {
    absolute: f64,
    modulus: f64,
    ulps: u64,
}

impl ComplexError {
    fn measure(z: C32, got: C32) -> Self {
        let want = reference_complex(z);
        let mut delta = [0.0f64; 2];
        let mut ulps = 0;
        for c in 0..2 {
            let rounded = want[c] as f32;
            ulps = ulps.max(ulp_distance(got[c], rounded));
            delta[c] = if !got[c].is_infinite() {
                got[c] as f64 - want[c]
            } else if rounded.is_infinite() {
                0.0
            } else {
                TWO_POW_128.copysign(got[c] as f64) - want[c]
            };
        }
        Self { absolute: delta[0].hypot(delta[1]), modulus: want[0].hypot(want[1]), ulps }
    }

    /// Normwise relative error in units of u.
    fn normwise_u(&self) -> f64 {
        self.absolute / (U * self.modulus)
    }

    fn within_gate(&self) -> bool {
        self.ulps <= COMPLEX_ULP_GATE
            && self.absolute <= COMPLEX_NORMWISE_GATE * U * self.modulus + SUBNORMAL_QUANTUM
    }

    /// True when the result is far enough above the subnormal range to count toward the
    /// reported worst relative error.
    fn in_normal_range(&self) -> bool {
        self.modulus >= NORMAL_RANGE_FLOOR
    }
}

/// Asserts the complex gate for one finite nonzero input and returns the measured error.
fn assert_complex(z: C32, got: C32) -> ComplexError {
    let error = ComplexError::measure(z, got);
    assert!(
        error.within_gate(),
        "1/({:e} + {:e}i) [{:#010x}, {:#010x}]: got [{:#010x}, {:#010x}], \
         normwise {:.3}u, {} ulps",
        z[0],
        z[1],
        z[0].to_bits(),
        z[1].to_bits(),
        got[0].to_bits(),
        got[1].to_bits(),
        error.normwise_u(),
        error.ulps
    );
    error
}

/// Runs the GPU on finite nonzero `inputs`, asserts the complex gate on every element, and
/// returns the worst normwise error (in u) among results in the normal range.
fn check_complex(ctx: &MetalContext, inputs: &[C32]) -> f64 {
    let outputs = gpu_reciprocal(ctx, inputs);
    let mut worst = 0.0f64;
    for (&z, &w) in inputs.iter().zip(&outputs) {
        let error = assert_complex(z, w);
        if error.in_normal_range() {
            worst = worst.max(error.normwise_u());
        }
    }
    worst
}

/// A float with random sign and significand whose magnitude lies in
/// [2^(field−127), 2^(field−126)). `field` is the biased exponent, extended below 1 to
/// subnormals (down to −22, the smallest subnormal).
fn float_with_field(rng: &mut ChaCha8Rng, field: i32) -> f32 {
    assert!((-22..=254).contains(&field));
    let fraction = rng.random::<u32>() & 0x007f_ffff;
    let magnitude = if field >= 1 {
        (field as u32) << 23 | fraction
    } else {
        let top = 1u32 << (22 + field);
        top | (fraction & (top - 1))
    };
    f32::from_bits(magnitude | (rng.random::<u32>() & 0x8000_0000))
}

/// Complex inputs with each component's exponent field drawn from `fields`, the imaginary
/// one within `spread` binades of the real one.
fn complex_inputs(
    seed: u64,
    count: usize,
    fields: std::ops::RangeInclusive<i32>,
    spread: i32,
) -> Vec<C32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            let re = rng.random_range(fields.clone());
            let lo = (*fields.start()).max(re - spread);
            let hi = (*fields.end()).min(re + spread);
            let im = rng.random_range(lo..=hi);
            [float_with_field(&mut rng, re), float_with_field(&mut rng, im)]
        })
        .collect()
}

/// Every sign combination of the complex number (a, b).
fn sign_variants(a: f32, b: f32) -> [C32; 4] {
    [[a, b], [-a, b], [a, -b], [-a, -b]]
}

fn bits(z: C32) -> [u32; 2] {
    [z[0].to_bits(), z[1].to_bits()]
}

// ---------------------------------------------------------------------------------------
// Subnormal-flush probe
// ---------------------------------------------------------------------------------------

/// Probe kernel: basic arithmetic on operands loaded from buffers, so nothing is folded at
/// compile time. Compiled with the production compile options.
const PROBE_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

/// Writes six probes per operand pair: the copied operand, product, quotient, sum,
/// comparison against zero, and fma(a, b, -1).
kernel void reciprocal_probe(device const float* lhs [[buffer(0)]],
                             device const float* rhs [[buffer(1)]],
                             device float* out [[buffer(2)]],
                             uint i [[thread_position_in_grid]])
{
    float a = lhs[i];
    float b = rhs[i];
    out[6 * i + 0] = a;
    out[6 * i + 1] = a * b;
    out[6 * i + 2] = a / b;
    out[6 * i + 3] = a + b;
    out[6 * i + 4] = (a == 0.0f) ? 1.0f : 0.0f;
    out[6 * i + 5] = fma(a, b, -1.0f);
}
"#;

const PROBE_OPS: [&str; 6] = ["copy", "mul", "div", "add", "eq0", "fma"];

/// Flush-to-zero model of this GPU: a subnormal becomes a zero of the same sign.
fn flush(x: f32) -> f32 {
    if x.is_subnormal() {
        0.0f32.copysign(x)
    } else {
        x
    }
}

/// IEEE 754 results of the six probes.
fn probe_ieee(a: f32, b: f32) -> [f32; 6] {
    [a, a * b, a / b, a + b, if a == 0.0 { 1.0 } else { 0.0 }, a.mul_add(b, -1.0)]
}

/// Results of the six probes under the pinned model: loads and stores keep the bits,
/// arithmetic and comparisons read subnormal operands as zero, subnormal results are
/// flushed to a signed zero, and `fma` is fused.
fn probe_flushed(a: f32, b: f32) -> [f32; 6] {
    let mut out = probe_ieee(flush(a), flush(b)).map(flush);
    out[0] = a;
    out
}

/// Pins how this GPU treats subnormals, which is why the production kernels classify,
/// normalize, and denormalize on integer bit patterns and keep every float operand and
/// result in the normal range.
#[test]
fn probe_subnormal_flush_is_pinned() {
    let ctx = MetalContext::new().unwrap();
    let min_normal = f32::MIN_POSITIVE;
    let third = 1.0f32 / 3.0;
    let cases: [(f32, f32); 12] = [
        (f32::from_bits(1), 2f32.powi(30)),
        (f32::from_bits(0x007f_ffff), 1.0),
        (-f32::from_bits(0x0040_0000), 2f32.powi(24)),
        (1.0, f32::from_bits(0x0040_0000)),
        (min_normal, 0.5),
        (-min_normal, 0.5),
        (1.0, 2f32.powi(127)),
        (-3.0, f32::MAX),
        (min_normal, -f32::from_bits(0x0080_0001)),
        (f32::from_bits(0x0040_0000), f32::from_bits(0x0040_0000)),
        (third, 3.0),
        (1.5, 2.5),
    ];
    let lhs = Shared::from_slice(&ctx, &cases.map(|c| c.0));
    let rhs = Shared::from_slice(&ctx, &cases.map(|c| c.1));
    let out = Shared::<f32>::zeroed(&ctx, 6 * cases.len());
    let pipeline = compute_pipeline(
        &ctx,
        "reciprocal_probe",
        PROBE_SOURCE,
        compile_options,
        "reciprocal_probe",
    )
    .unwrap();
    let cmd = ctx.queue.new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&lhs.buffer), 0);
    encoder.set_buffer(1, Some(&rhs.buffer), 0);
    encoder.set_buffer(2, Some(&out.buffer), 0);
    encoder.dispatch_threads(
        metal::MTLSize::new(cases.len() as u64, 1, 1),
        metal::MTLSize::new(1, 1, 1),
    );
    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let mut departures = 0;
    for (i, &(a, b)) in cases.iter().enumerate() {
        let (ieee, model) = (probe_ieee(a, b), probe_flushed(a, b));
        for op in 0..6 {
            let got = out.slice()[6 * i + op];
            assert_eq!(
                ulp_distance(got, model[op]),
                0,
                "{}({a:e}, {b:e}): gpu {:#010x}, flush model {:#010x}, ieee {:#010x}",
                PROBE_OPS[op],
                got.to_bits(),
                model[op].to_bits(),
                ieee[op].to_bits()
            );
            departures += usize::from(ulp_distance(got, ieee[op]) != 0);
        }
    }
    let probe = |case: usize, op: usize| out.slice()[6 * case + op].to_bits();
    // Loads and stores keep subnormal bit patterns.
    assert_eq!(probe(0, 0), 1);
    assert_eq!(probe(1, 0), 0x007f_ffff);
    // Subnormal operands read as zero: 2⁻¹⁴⁹·2³⁰ = 0, subnormal == 0, 1/subnormal = ∞.
    assert_eq!(probe(0, 1), 0);
    assert_eq!(probe(0, 4), 1.0f32.to_bits());
    assert_eq!(probe(3, 2), f32::INFINITY.to_bits());
    // Subnormal results flush to a zero of the same sign.
    assert_eq!(probe(4, 1), 0);
    assert_eq!(probe(5, 1), 0x8000_0000);
    assert_eq!(probe(6, 2), 0);
    assert_eq!(probe(8, 3), 0x8000_0000);
    // fma is fused: the residual of RN(1/3)·3 − 1 is exact and nonzero.
    assert_eq!(probe(10, 5), third.mul_add(3.0, -1.0).to_bits());
    assert_ne!(probe(10, 5) & 0x7fff_ffff, 0);
    assert!(departures > 0);
    println!(
        "subnormal-flush probe on {}: loads/stores keep subnormal bits; subnormal operands \
         read as zero (DAZ); subnormal results flush to signed zero (FTZ); fma is fused; \
         {departures} of {} probes depart from IEEE 754",
        ctx.device.name(),
        6 * cases.len()
    );
}

// ---------------------------------------------------------------------------------------
// Real f32
// ---------------------------------------------------------------------------------------

#[test]
fn real_simple_values_are_exact() {
    let ctx = MetalContext::new().unwrap();
    let inputs = [1.0f32, -1.0, 2.0, -4.0, 0.5, 0.125, 3.0, 10.0, -7.0, 0.1, 1.5, 255.0];
    let outputs = gpu_reciprocal(&ctx, &inputs);
    assert_eq!(&outputs[..6], &[1.0, -1.0, 0.5, -0.25, 2.0, 8.0]);
    for (&x, &y) in inputs.iter().zip(&outputs) {
        assert_real(x, y);
    }
}

#[test]
fn real_normal_values_match_reference() {
    let ctx = MetalContext::new().unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(0x5ec1_0001);
    let inputs: Vec<f32> = (0..400_000)
        .map(|_| {
            let field = rng.random_range(1..=254);
            float_with_field(&mut rng, field)
        })
        .collect();
    check_real(&ctx, &inputs);
}

#[test]
fn real_subnormal_inputs_give_large_or_infinite_results() {
    let ctx = MetalContext::new().unwrap();
    // 2⁻¹⁴⁹·k overflows exactly when k ≤ 2²¹.
    let edges = [1u32, 2, 3, 0x000f_ffff, 0x001f_ffff, 0x0020_0000, 0x0020_0001, 0x0020_0002];
    let more = [0x002a_aaaa, 0x003f_ffff, 0x0040_0000, 0x0040_0001, 0x0055_5555, 0x007f_ffff];
    let mut inputs: Vec<f32> =
        edges.iter().chain(&more).flat_map(|&k| [f32::from_bits(k), -f32::from_bits(k)]).collect();
    let outputs = gpu_reciprocal(&ctx, &inputs);
    assert_eq!(outputs[0], f32::INFINITY);
    assert_eq!(outputs[1], f32::NEG_INFINITY);
    assert_eq!(outputs[10], f32::INFINITY);
    assert_eq!(outputs[12].to_bits(), 0x7f7f_fff8);
    assert_eq!(outputs[13].to_bits(), 0xff7f_fff8);
    assert_eq!(outputs[20], 2f32.powi(127));

    let mut rng = ChaCha8Rng::seed_from_u64(0x5ec1_0002);
    inputs.extend((0..200_000).map(|_| {
        let field = rng.random_range(-22..=0);
        float_with_field(&mut rng, field)
    }));
    check_real(&ctx, &inputs);
}

#[test]
fn real_large_inputs_give_correctly_rounded_subnormals() {
    let ctx = MetalContext::new().unwrap();
    // 1/x is subnormal exactly when |x| > 2¹²⁶ (bits above 0x7e80_0000).
    let edges = [0x7e7f_ffffu32, 0x7e80_0000, 0x7e80_0001, 0x7e80_0002, 0x7eff_ffff];
    let more = [0x7f00_0000, 0x7f00_0001, 0x7f2a_aaaa, 0x7f7f_fffe, 0x7f7f_ffff];
    let mut inputs: Vec<f32> =
        edges.iter().chain(&more).flat_map(|&k| [f32::from_bits(k), -f32::from_bits(k)]).collect();
    let outputs = gpu_reciprocal(&ctx, &inputs);
    assert_eq!(outputs[2], f32::MIN_POSITIVE);
    assert_eq!(outputs[10].to_bits(), 0x0040_0000);
    assert_eq!(outputs[18].to_bits(), 0x0020_0000);
    assert_eq!(outputs[19].to_bits(), 0x8020_0000);

    let mut rng = ChaCha8Rng::seed_from_u64(0x5ec1_0003);
    inputs.extend((0..200_000).map(|_| {
        let field = rng.random_range(253..=254);
        float_with_field(&mut rng, field)
    }));
    check_real(&ctx, &inputs);
}

#[test]
fn real_powers_of_ten_and_extremes() {
    let ctx = MetalContext::new().unwrap();
    let mut inputs: Vec<f32> = (-45..=38)
        .flat_map(|k| {
            let x = format!("1e{k}").parse::<f32>().unwrap();
            [x, -x]
        })
        .collect();
    assert!(inputs.contains(&1e37) && inputs.contains(&1e-37) && inputs.contains(&-1e-45));
    inputs.extend([f32::MAX, -f32::MAX, f32::MIN_POSITIVE, -f32::MIN_POSITIVE, f32::EPSILON]);
    check_real(&ctx, &inputs);
}

#[test]
fn real_specials_follow_ieee() {
    let ctx = MetalContext::new().unwrap();
    let signaling = f32::from_bits(0x7f80_0001);
    let inputs = [0.0f32, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN, -f32::NAN, signaling];
    let outputs = gpu_reciprocal(&ctx, &inputs);
    let want = [f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0];
    for i in 0..4 {
        assert_eq!(outputs[i].to_bits(), want[i].to_bits(), "1/{:e}", inputs[i]);
    }
    for i in 4..7 {
        assert!(outputs[i].is_nan(), "1/NaN gave {:#010x}", outputs[i].to_bits());
        // The NaN is quiet and keeps the sign and payload of the operand.
        assert_eq!(outputs[i].to_bits(), inputs[i].to_bits() | 0x0040_0000);
    }
}

#[test]
fn real_random_bit_patterns_match_reference() {
    let ctx = MetalContext::new().unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(0x5ec1_0004);
    let inputs: Vec<f32> = (0..1_000_000).map(|_| f32::from_bits(rng.random())).collect();
    check_real(&ctx, &inputs);
}

// ---------------------------------------------------------------------------------------
// Complex f32
// ---------------------------------------------------------------------------------------

#[test]
fn complex_simple_values() {
    let ctx = MetalContext::new().unwrap();
    let inputs: [C32; 6] =
        [[2.0, 0.0], [2.0, -0.0], [0.0, 2.0], [-0.0, 2.0], [-4.0, 0.0], [1.0, 1.0]];
    let want: [C32; 6] =
        [[0.5, -0.0], [0.5, 0.0], [0.0, -0.5], [-0.0, -0.5], [-0.25, -0.0], [0.5, -0.5]];
    let outputs = gpu_reciprocal(&ctx, &inputs);
    for i in 0..inputs.len() {
        assert_eq!(bits(outputs[i]), bits(want[i]), "1/{:?}", inputs[i]);
    }
    check_complex(&ctx, &[[3.0, 4.0], [-3.0, 4.0], [0.1, -0.7], [1e3, 1e-3], [5.0, -12.0]]);
}

#[test]
fn complex_spec_inputs() {
    let ctx = MetalContext::new().unwrap();
    let mut inputs = Vec::new();
    // a = b = 10^±30, and |a/b| ≈ 10^±38.
    for (a, b) in [
        (1e30f32, 1e30f32),
        (1e-30, 1e-30),
        (1e19, 1e-19),
        (1e-19, 1e19),
        (1e38, 1.0),
        (1.0, 1e38),
        (1.0, 1e-38),
        (1e-38, 1.0),
        (3e37, 2.5e-1),
        (7e-20, 6e18),
    ] {
        inputs.extend(sign_variants(a, b));
    }
    let worst = check_complex(&ctx, &inputs);
    println!("complex spec inputs: worst normwise error {worst:.3}u");

    let outputs = gpu_reciprocal(&ctx, &[[1e30f32, 1e30], [1e-30, 1e-30]]);
    assert_eq!(outputs[0], [(0.5f64 / 1e30f32 as f64) as f32, -(0.5f64 / 1e30f32 as f64) as f32]);
    assert_eq!(outputs[1], [(0.5f64 / 1e-30f32 as f64) as f32, -(0.5f64 / 1e-30f32 as f64) as f32]);
}

#[test]
fn complex_random_inputs_meet_normwise_gate() {
    let ctx = MetalContext::new().unwrap();
    let moderate = check_complex(&ctx, &complex_inputs(0x5ec1_0010, 300_000, 87..=167, 80));
    let close = check_complex(&ctx, &complex_inputs(0x5ec1_0011, 300_000, 87..=167, 2));
    let wide = check_complex(&ctx, &complex_inputs(0x5ec1_0012, 300_000, -22..=254, 276));
    println!(
        "complex random: worst normwise error {moderate:.3}u (moderate), {close:.3}u \
         (comparable parts), {wide:.3}u (full range)"
    );
}

#[test]
fn complex_near_overflow_inputs() {
    let ctx = MetalContext::new().unwrap();
    let max = f32::MAX;
    let mut inputs = Vec::new();
    for (a, b) in [
        (max, max),
        (max, 0.0),
        (max, 1.0),
        (max, max / 2.0),
        (2f32.powi(127), 2f32.powi(127)),
        (2f32.powi(126), 2f32.powi(126)),
        (2f32.powi(126), 1.0),
        (f32::from_bits(0x7e80_0001), 3.0),
        (max, 2f32.powi(100)),
        (3e38, 1e38),
    ] {
        inputs.extend(sign_variants(a, b));
        inputs.extend(sign_variants(b, a));
    }
    inputs.extend(complex_inputs(0x5ec1_0020, 200_000, 245..=254, 9));
    inputs.extend(complex_inputs(0x5ec1_0021, 100_000, 200..=254, 54));
    check_complex(&ctx, &inputs);

    // 1/(MAX + MAX·i) = 2⁻¹²⁹(1 − i)(1 + 2⁻²⁴ + …): 2²⁰ subnormal quanta in each part.
    let outputs = gpu_reciprocal(&ctx, &[[max, max], [2f32.powi(127), 0.0]]);
    assert_eq!(bits(outputs[0]), [0x0010_0000, 0x8010_0000]);
    assert_eq!(bits(outputs[1]), [0x0040_0000, 0x8000_0000]);
}

#[test]
fn complex_near_underflow_inputs() {
    let ctx = MetalContext::new().unwrap();
    let tiny = f32::from_bits(1);
    let min_normal = f32::MIN_POSITIVE;
    let mut inputs = Vec::new();
    for (a, b) in [
        (tiny, tiny),
        (tiny, 0.0),
        (min_normal, min_normal),
        (min_normal, 0.0),
        (min_normal, tiny),
        (f32::from_bits(0x0020_0001), 0.0),
        (f32::from_bits(0x0020_0001), f32::from_bits(0x0000_0400)),
        (f32::from_bits(0x0030_0000), f32::from_bits(0x0030_0000)),
        (f32::from_bits(0x007f_ffff), f32::from_bits(0x0055_5555)),
        (1e-38, 1e-45),
        (2f32.powi(-20), tiny),
        (2f32.powi(-100), f32::from_bits(3)),
    ] {
        inputs.extend(sign_variants(a, b));
        inputs.extend(sign_variants(b, a));
    }
    inputs.extend(complex_inputs(0x5ec1_0030, 200_000, -22..=3, 25));
    inputs.extend(complex_inputs(0x5ec1_0031, 100_000, -22..=60, 82));
    check_complex(&ctx, &inputs);

    let outputs = gpu_reciprocal(
        &ctx,
        &[[tiny, tiny], [min_normal, min_normal], [-tiny, 0.0], [2f32.powi(-20), tiny]],
    );
    assert_eq!(outputs[0], [f32::INFINITY, f32::NEG_INFINITY]);
    assert_eq!(outputs[1], [2f32.powi(125), -(2f32.powi(125))]);
    assert_eq!(bits(outputs[2]), bits([f32::NEG_INFINITY, -0.0]));
    // The tiny imaginary part still yields a normal result: −b/a² = −2⁻¹⁰⁹.
    assert_eq!(outputs[3], [2f32.powi(20), -(2f32.powi(-109))]);
}

/// Pins the documented results for zero, infinite, and NaN operands (C99 Annex G style):
///
/// - an infinite part (even beside a NaN) makes `z` an infinity; `1/z` is the zero
///   `(copysign(0, re), copysign(0, −im))`
/// - otherwise a NaN part gives `(NaN, NaN)`
/// - `1/(±0 ± 0i)` is the infinity `(copysign(∞, re), copysign(0, −im))`, so that
///   `1/(x + 0i)` agrees with the real `1/x` for every real `x`
#[test]
fn complex_specials_follow_documented_table() {
    let ctx = MetalContext::new().unwrap();
    let inf = f32::INFINITY;
    let nan = f32::from_bits(0x7fc0_0000);
    let exact: [(C32, C32); 20] = [
        ([inf, 0.0], [0.0, -0.0]),
        ([-inf, 0.0], [-0.0, -0.0]),
        ([inf, -3.0], [0.0, 0.0]),
        ([-inf, -0.0], [-0.0, 0.0]),
        ([1.0, inf], [0.0, -0.0]),
        ([-1.0, -inf], [-0.0, 0.0]),
        ([0.0, inf], [0.0, -0.0]),
        ([inf, inf], [0.0, -0.0]),
        ([-inf, inf], [-0.0, -0.0]),
        ([inf, -inf], [0.0, 0.0]),
        ([-inf, -inf], [-0.0, 0.0]),
        ([inf, nan], [0.0, -0.0]),
        ([nan, -inf], [0.0, 0.0]),
        ([-inf, -nan], [-0.0, 0.0]),
        ([f32::MAX, inf], [0.0, -0.0]),
        ([f32::from_bits(1), -inf], [0.0, 0.0]),
        ([0.0, 0.0], [inf, -0.0]),
        ([-0.0, 0.0], [-inf, -0.0]),
        ([0.0, -0.0], [inf, 0.0]),
        ([-0.0, -0.0], [-inf, 0.0]),
    ];
    let outputs = gpu_reciprocal(&ctx, &exact.map(|case| case.0));
    for (i, (z, want)) in exact.iter().enumerate() {
        assert_eq!(bits(outputs[i]), bits(*want), "1/{z:?} gave {:?}", outputs[i]);
    }

    let signaling = f32::from_bits(0xff80_0001);
    let nans: [C32; 8] = [
        [nan, 1.0],
        [1.0, nan],
        [nan, nan],
        [nan, 0.0],
        [0.0, nan],
        [-nan, f32::MAX],
        [signaling, 2.0],
        [f32::from_bits(1), signaling],
    ];
    for (z, w) in nans.iter().zip(gpu_reciprocal(&ctx, &nans)) {
        assert!(w[0].is_nan() && w[1].is_nan(), "1/{z:?} gave {w:?}");
    }
}

/// `1/(x + 0i)` has the real reciprocal's real part and an imaginary part of −0.
#[test]
fn complex_on_the_real_axis_matches_real() {
    let ctx = MetalContext::new().unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(0x5ec1_0040);
    let mut reals = vec![0.0f32, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::MAX, 1e-45];
    reals.extend((0..100_000).map(|_| {
        let field = rng.random_range(-22..=254);
        float_with_field(&mut rng, field)
    }));
    let embedded: Vec<C32> = reals.iter().map(|&x| [x, 0.0]).collect();
    let outputs = gpu_reciprocal(&ctx, &embedded);
    for (&x, w) in reals.iter().zip(&outputs) {
        let want = reference_real(x);
        // Subnormal results are denormalized from a rounded quotient: at most 1 ULP off.
        let gate = u64::from(want.is_subnormal());
        assert!(ulp_distance(w[0], want) <= gate, "1/({x:e} + 0i) gave {w:?}");
        assert_eq!(w[1].to_bits(), (-0.0f32).to_bits(), "1/({x:e} + 0i) gave {w:?}");
    }
}

// ---------------------------------------------------------------------------------------
// Layouts: scalars, strided vectors, batched diagonals, in place and out of place
// ---------------------------------------------------------------------------------------

/// Element types of the layout tests: how to draw inputs and what the kernel must produce.
trait Element: ReciprocalElement + std::fmt::Debug {
    /// A random finite input.
    fn sample(rng: &mut ChaCha8Rng) -> Self;
    /// Bit pattern, for exact comparisons.
    fn pattern(self) -> u64;
    /// The values the primitive must write for `inputs`.
    fn expected(ctx: &MetalContext, inputs: &[Self]) -> Vec<Self>;
}

impl Element for f32 {
    fn sample(rng: &mut ChaCha8Rng) -> Self {
        let field = rng.random_range(-22..=254);
        float_with_field(rng, field)
    }

    fn pattern(self) -> u64 {
        u64::from(self.to_bits())
    }

    fn expected(_ctx: &MetalContext, inputs: &[Self]) -> Vec<Self> {
        inputs.iter().map(|&x| reference_real(x)).collect()
    }
}

impl Element for C32 {
    fn sample(rng: &mut ChaCha8Rng) -> Self {
        let re = rng.random_range(-22..=254);
        let im = rng.random_range((re - 30).max(-22)..=(re + 30).min(254));
        [float_with_field(rng, re), float_with_field(rng, im)]
    }

    fn pattern(self) -> u64 {
        u64::from(self[0].to_bits()) << 32 | u64::from(self[1].to_bits())
    }

    /// The contiguous out-of-place GPU result, itself checked against the f64 reference.
    fn expected(ctx: &MetalContext, inputs: &[Self]) -> Vec<Self> {
        let outputs = gpu_reciprocal(ctx, inputs);
        for (&z, &w) in inputs.iter().zip(&outputs) {
            assert_complex(z, w);
        }
        outputs
    }
}

fn random_elements<T: Element>(rng: &mut ChaCha8Rng, len: usize) -> Vec<T> {
    (0..len).map(|_| T::sample(rng)).collect()
}

fn assert_same_bits<T: Element>(got: &[T], want: &[T], what: &str) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.pattern(), w.pattern(), "{what}: element {i} is {g:?}, expected {w:?}");
    }
}

/// Runs `src → dst` between two buffers and checks that exactly the destination view's
/// elements changed, to the expected values.
fn check_out_of_place<T: Element>(ctx: &MetalContext, seed: u64, src: &Layout, dst: &Layout) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let src_data = random_elements::<T>(&mut rng, src.extent() + 3);
    let dst_data = random_elements::<T>(&mut rng, dst.extent() + 3);
    let src_buffer = Shared::from_slice(ctx, &src_data);
    let dst_buffer = Shared::from_slice(ctx, &dst_data);
    reciprocal(ctx, &src.view(&src_buffer), &dst.view(&dst_buffer)).unwrap();

    let inputs: Vec<T> = src.indices().iter().map(|&i| src_data[i]).collect();
    let mut want = dst_data;
    for (&i, &y) in dst.indices().iter().zip(&T::expected(ctx, &inputs)) {
        want[i] = y;
    }
    assert_same_bits(src_buffer.slice(), &src_data, &format!("source of {src:?}"));
    assert_same_bits(dst_buffer.slice(), &want, &format!("{src:?} → {dst:?}"));
}

/// Runs the view onto itself and checks that exactly its elements changed, to the expected
/// values.
fn check_in_place<T: Element>(ctx: &MetalContext, seed: u64, layout: &Layout) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let data = random_elements::<T>(&mut rng, layout.extent() + 3);
    let buffer = Shared::from_slice(ctx, &data);
    reciprocal(ctx, &layout.view(&buffer), &layout.view(&buffer)).unwrap();

    let inputs: Vec<T> = layout.indices().iter().map(|&i| data[i]).collect();
    let mut want = data;
    for (&i, &y) in layout.indices().iter().zip(&T::expected(ctx, &inputs)) {
        want[i] = y;
    }
    assert_same_bits(buffer.slice(), &want, &format!("in place {layout:?}"));
}

/// Out of place to a like-strided and to a contiguous destination, then in place.
fn check_layout<T: Element>(ctx: &MetalContext, seed: u64, layout: &Layout) {
    let contiguous = Layout::new(1, &layout.shape, &linalg::abi::row_major_strides(&layout.shape));
    check_out_of_place::<T>(ctx, seed, layout, layout);
    check_out_of_place::<T>(ctx, seed + 1, layout, &contiguous);
    check_out_of_place::<T>(ctx, seed + 2, &contiguous, layout);
    check_in_place::<T>(ctx, seed + 3, layout);
}

fn scalar_layouts<T: Element>() {
    let ctx = MetalContext::new().unwrap();
    check_layout::<T>(&ctx, 100, &Layout::new(0, &[], &[]));
    check_layout::<T>(&ctx, 110, &Layout::new(5, &[], &[]));
    check_layout::<T>(&ctx, 120, &Layout::new(7, &[1], &[9]));
    check_layout::<T>(&ctx, 130, &Layout::new(4099, &[1, 1, 1], &[0, 5, 1]));
}

fn vector_layouts<T: Element>() {
    let ctx = MetalContext::new().unwrap();
    check_layout::<T>(&ctx, 200, &Layout::new(0, &[1000], &[1]));
    check_layout::<T>(&ctx, 210, &Layout::new(2, &[333], &[3]));
    check_layout::<T>(&ctx, 220, &Layout::new(17, &[5000], &[7]));
    check_layout::<T>(&ctx, 230, &Layout::new(1, &[2], &[4097]));
    check_out_of_place::<T>(
        &ctx,
        240,
        &Layout::new(3, &[257], &[2]),
        &Layout::new(11, &[257], &[5]),
    );
    // A source axis of stride 0 broadcasts one element.
    check_out_of_place::<T>(&ctx, 250, &Layout::new(6, &[40], &[0]), &Layout::new(0, &[40], &[2]));
}

fn tensor_layouts<T: Element>() {
    let ctx = MetalContext::new().unwrap();
    check_layout::<T>(&ctx, 300, &Layout::new(0, &[33, 65], &[65, 1]));
    check_layout::<T>(&ctx, 310, &Layout::new(9, &[12, 10], &[40, 3]));
    check_layout::<T>(&ctx, 320, &Layout::new(2, &[5, 6, 7], &[100, 14, 2]));
    check_layout::<T>(&ctx, 330, &Layout::new(0, &[3, 1, 300], &[301, 77, 1]));
    // Column-major: the first axis has the smallest stride.
    check_layout::<T>(&ctx, 340, &Layout::new(4, &[20, 30], &[1, 23]));
}

fn diagonal_layouts<T: Element>() {
    let ctx = MetalContext::new().unwrap();
    let mut seed = 400;
    for (ld, n) in [(1, 1), (7, 7), (7, 5), (128, 128), (128, 100)] {
        for batch in [1, 3, 17] {
            for offset in [0, 11] {
                check_layout::<T>(&ctx, seed, &Layout::diagonal(ld, n, batch, offset));
                seed += 10;
            }
        }
    }
    // A single unbatched diagonal is a rank-1 view of stride ld + 1.
    check_layout::<T>(&ctx, seed, &Layout::new(3, &[128], &[129]));
}

#[test]
fn real_scalars() {
    scalar_layouts::<f32>();
}

#[test]
fn complex_scalars() {
    scalar_layouts::<C32>();
}

#[test]
fn real_strided_vectors() {
    vector_layouts::<f32>();
}

#[test]
fn complex_strided_vectors() {
    vector_layouts::<C32>();
}

#[test]
fn real_strided_tensors() {
    tensor_layouts::<f32>();
}

#[test]
fn complex_strided_tensors() {
    tensor_layouts::<C32>();
}

#[test]
fn real_batched_diagonals() {
    diagonal_layouts::<f32>();
}

#[test]
fn complex_batched_diagonals() {
    diagonal_layouts::<C32>();
}

fn in_place_equals_out_of_place<T: Element>(seed: u64) {
    let ctx = MetalContext::new().unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for layout in [
        Layout::new(0, &[4096], &[1]),
        Layout::new(5, &[777], &[3]),
        Layout::diagonal(128, 128, 4, 9),
    ] {
        let data = random_elements::<T>(&mut rng, layout.extent());
        let in_place = Shared::from_slice(&ctx, &data);
        reciprocal(&ctx, &layout.view(&in_place), &layout.view(&in_place)).unwrap();
        let source = Shared::from_slice(&ctx, &data);
        let out_of_place = Shared::from_slice(&ctx, &data);
        reciprocal(&ctx, &layout.view(&source), &layout.view(&out_of_place)).unwrap();
        assert_same_bits(in_place.slice(), out_of_place.slice(), &format!("{layout:?}"));
        assert_same_bits(source.slice(), &data, "out-of-place source");
    }
}

#[test]
fn real_in_place_equals_out_of_place() {
    in_place_equals_out_of_place::<f32>(500);
}

#[test]
fn complex_in_place_equals_out_of_place() {
    in_place_equals_out_of_place::<C32>(510);
}

// ---------------------------------------------------------------------------------------
// Metadata validation
// ---------------------------------------------------------------------------------------

/// Asserts that both entry points reject `src → dst` with an error matching `expected`.
fn assert_rejected<T: ReciprocalElement>(
    ctx: &MetalContext,
    src: &TensorView<'_, T>,
    dst: &TensorView<'_, T>,
    expected: fn(&LinalgError) -> bool,
) {
    let sync = reciprocal(ctx, src, dst).unwrap_err();
    assert!(expected(&sync), "sync wrapper returned {sync:?} for {src:?} → {dst:?}");
    let cmd = ctx.queue.new_command_buffer();
    let encoded = encode_reciprocal(ctx, cmd, src, dst).unwrap_err();
    assert!(expected(&encoded), "encoder returned {encoded:?} for {src:?} → {dst:?}");
    cmd.commit();
    cmd.wait_until_completed();
}

fn is_invalid_argument(error: &LinalgError) -> bool {
    matches!(error, LinalgError::InvalidArgument(_))
}

fn is_overlap(error: &LinalgError) -> bool {
    matches!(error, LinalgError::Overlap(_))
}

fn rejects_invalid_metadata<T: Element>() {
    let ctx = MetalContext::new().unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(600);
    let src_data = random_elements::<T>(&mut rng, 512);
    let dst_data = random_elements::<T>(&mut rng, 512);
    let src = Shared::from_slice(&ctx, &src_data);
    let dst = Shared::from_slice(&ctx, &dst_data);
    let capacity = dst.capacity();
    let reject = |s: TensorView<'_, T>, d: TensorView<'_, T>| {
        assert_rejected(&ctx, &s, &d, is_invalid_argument);
    };

    // Shapes differ in extent, in rank, or in factorization.
    reject(src.view(0, &[8], &[1]), dst.view(0, &[9], &[1]));
    reject(src.view(0, &[8], &[1]), dst.view(0, &[8, 1], &[1, 1]));
    reject(src.view(0, &[4, 2], &[2, 1]), dst.view(0, &[2, 4], &[4, 1]));
    // Shape and strides disagree in rank.
    reject(src.view(0, &[8], &[1, 1]), dst.view(0, &[8], &[1]));
    reject(src.view(0, &[8], &[1]), dst.view(0, &[8], &[]));
    // A view reaches past the end of its buffer.
    reject(src.view(capacity - 3, &[4], &[1]), dst.view(0, &[4], &[1]));
    reject(src.view(0, &[4], &[1]), dst.view(capacity - 3, &[4], &[1]));
    reject(src.view(0, &[2], &[1]), dst.view(0, &[2], &[capacity]));
    reject(src.view(capacity, &[], &[]), dst.view(0, &[], &[]));
    reject(src.view(0, &[2, 2], &[usize::MAX, 1]), dst.view(0, &[2, 2], &[2, 1]));
    // Rank above MAX_RANK.
    reject(src.view(0, &[2, 2, 2, 2], &[8, 4, 2, 1]), dst.view(0, &[2, 2, 2, 2], &[8, 4, 2, 1]));
    // A destination that addresses one element twice would race.
    reject(src.view(0, &[4], &[1]), dst.view(0, &[4], &[0]));
    reject(src.view(0, &[2, 2], &[2, 1]), dst.view(0, &[2, 2], &[1, 1]));
    reject(src.view(0, &[3, 4], &[4, 1]), dst.view(0, &[3, 4], &[3, 1]));
    reject(dst.view(0, &[4], &[0]), dst.view(0, &[4], &[0]));
    // A host pointer that is not the buffer base.
    let mut shifted = src.view(0, &[4], &[1]);
    shifted.host_ptr = unsafe { shifted.host_ptr.add(1) };
    reject(shifted, dst.view(0, &[4], &[1]));

    assert_same_bits(src.slice(), &src_data, "source after rejected calls");
    assert_same_bits(dst.slice(), &dst_data, "destination after rejected calls");
}

#[test]
fn real_rejects_invalid_metadata() {
    rejects_invalid_metadata::<f32>();
}

#[test]
fn complex_rejects_invalid_metadata() {
    rejects_invalid_metadata::<C32>();
}

fn rejects_partial_overlap<T: Element>() {
    let ctx = MetalContext::new().unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(700);
    let data = random_elements::<T>(&mut rng, 256);
    let buffer = Shared::from_slice(&ctx, &data);
    let reject = |s: TensorView<'_, T>, d: TensorView<'_, T>| {
        assert_rejected(&ctx, &s, &d, is_overlap);
    };

    // Shifted windows, in both directions, and by a single element.
    reject(buffer.view(0, &[10], &[1]), buffer.view(5, &[10], &[1]));
    reject(buffer.view(5, &[10], &[1]), buffer.view(0, &[10], &[1]));
    reject(buffer.view(0, &[10], &[1]), buffer.view(9, &[10], &[1]));
    // Same offset and shape but different strides.
    reject(buffer.view(0, &[4], &[1]), buffer.view(0, &[4], &[2]));
    reject(buffer.view(0, &[4, 4], &[4, 1]), buffer.view(0, &[4, 4], &[1, 4]));
    // A diagonal against the matrix row it crosses.
    reject(buffer.view(0, &[8], &[9]), buffer.view(8, &[8], &[1]));
    // A scalar inside a vector.
    reject(buffer.view(3, &[1], &[1]), buffer.view(3, &[1], &[5]));
    assert_same_bits(buffer.slice(), &data, "buffer after rejected calls");

    // Disjoint windows of one buffer are fine.
    let (lo, hi) = (Layout::new(0, &[100], &[1]), Layout::new(100, &[100], &[1]));
    reciprocal(&ctx, &lo.view(&buffer), &hi.view(&buffer)).unwrap();
    let mut want = data.clone();
    want[100..200].copy_from_slice(&T::expected(&ctx, &data[..100]));
    assert_same_bits(buffer.slice(), &want, "disjoint windows of one buffer");
}

#[test]
fn real_rejects_partial_overlap() {
    rejects_partial_overlap::<f32>();
}

#[test]
fn complex_rejects_partial_overlap() {
    rejects_partial_overlap::<C32>();
}

fn empty_views_are_no_ops<T: Element>() {
    let ctx = MetalContext::new().unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(800);
    let data = random_elements::<T>(&mut rng, 64);
    let src = Shared::from_slice(&ctx, &data);
    let dst = Shared::from_slice(&ctx, &data);
    reciprocal(&ctx, &src.view(0, &[0], &[1]), &dst.view(0, &[0], &[1])).unwrap();
    reciprocal(&ctx, &src.view(3, &[4, 0], &[8, 1]), &dst.view(60, &[4, 0], &[1, 1])).unwrap();
    reciprocal(&ctx, &dst.view(2, &[0, 5], &[5, 1]), &dst.view(2, &[0, 5], &[5, 1])).unwrap();
    let cmd = ctx.queue.new_command_buffer();
    encode_reciprocal(&ctx, cmd, &src.view(0, &[0], &[1]), &dst.view(1, &[0], &[1])).unwrap();
    cmd.commit();
    cmd.wait_until_completed();
    assert_same_bits(dst.slice(), &data, "destination after empty calls");
}

#[test]
fn real_empty_views_are_no_ops() {
    empty_views_are_no_ops::<f32>();
}

#[test]
fn complex_empty_views_are_no_ops() {
    empty_views_are_no_ops::<C32>();
}

// ---------------------------------------------------------------------------------------
// Encode-only entry point
// ---------------------------------------------------------------------------------------

/// Several encodes share one command buffer and run in order once it is committed.
#[test]
fn encode_composes_in_one_command_buffer() {
    let ctx = MetalContext::new().unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(900);
    let reals: Vec<f32> = (0..3000)
        .map(|_| {
            let field = rng.random_range(100..=150);
            float_with_field(&mut rng, field)
        })
        .collect();
    let complexes = complex_inputs(901, 3000, 100..=150, 10);
    let x = Shared::from_slice(&ctx, &reals);
    let y = Shared::<f32>::zeroed(&ctx, reals.len());
    let z = Shared::from_slice(&ctx, &complexes);
    let w = Shared::<C32>::zeroed(&ctx, complexes.len());

    let cmd = ctx.queue.new_command_buffer();
    encode_reciprocal(&ctx, cmd, &x.vector(), &y.vector()).unwrap();
    encode_reciprocal(&ctx, cmd, &y.vector(), &y.vector()).unwrap();
    encode_reciprocal(&ctx, cmd, &z.vector(), &w.vector()).unwrap();
    encode_reciprocal(&ctx, cmd, &x.vector(), &x.vector()).unwrap();
    // Nothing runs before the commit.
    assert!(y.slice().iter().all(|&v| v.to_bits() == 0));
    assert_same_bits(x.slice(), &reals, "source before commit");
    cmd.commit();
    cmd.wait_until_completed();

    let once: Vec<f32> = reals.iter().map(|&v| reference_real(v)).collect();
    let twice: Vec<f32> = once.iter().map(|&v| reference_real(v)).collect();
    assert_same_bits(y.slice(), &twice, "1/(1/x)");
    assert_same_bits(x.slice(), &once, "1/x in place");
    for (&input, &output) in complexes.iter().zip(w.slice()) {
        assert_complex(input, output);
    }
}

/// The LU use case: invert the pivot of a column-major matrix in place on the device.
fn inverts_pivot_in_place<T: Element>() {
    let ctx = MetalContext::new().unwrap();
    let (n, ld) = (6, 9);
    let mut rng = ChaCha8Rng::seed_from_u64(910);
    let data = random_elements::<T>(&mut rng, ld * n);
    let matrix = Shared::from_slice(&ctx, &data);
    let cmd = ctx.queue.new_command_buffer();
    for k in 0..n {
        let pivot = matrix.view(k * ld + k, &[], &[]);
        encode_reciprocal(&ctx, cmd, &pivot, &pivot).unwrap();
    }
    cmd.commit();
    cmd.wait_until_completed();

    let pivots: Vec<T> = (0..n).map(|k| data[k * ld + k]).collect();
    let mut want = data;
    for (k, &inverse) in T::expected(&ctx, &pivots).iter().enumerate() {
        want[k * ld + k] = inverse;
    }
    assert_same_bits(matrix.slice(), &want, "matrix with inverted pivots");
}

#[test]
fn real_inverts_pivot_in_place() {
    inverts_pivot_in_place::<f32>();
}

#[test]
fn complex_inverts_pivot_in_place() {
    inverts_pivot_in_place::<C32>();
}

// ---------------------------------------------------------------------------------------
// Sweeps (exclusive GPU lock)
// ---------------------------------------------------------------------------------------

/// Counts of ULP distances 0..=3, of distances ≥ 4, and of sign/NaN mismatches, plus the
/// worst input seen.
#[derive(Clone, Copy, Default)]
struct UlpHistogram {
    counts: [u64; 6],
    worst: Option<(u64, u32)>,
}

impl UlpHistogram {
    const LABELS: [&'static str; 6] = ["0", "1", "2", "3", ">=4", "sign/NaN mismatch"];

    fn record(mut self, input_bits: u32, distance: u64) -> Self {
        let bucket = if distance == u64::MAX { 5 } else { distance.min(4) as usize };
        self.counts[bucket] += 1;
        if distance > self.worst.map_or(0, |w| w.0) {
            self.worst = Some((distance, input_bits));
        }
        self
    }

    fn merge(mut self, other: Self) -> Self {
        for (count, add) in self.counts.iter_mut().zip(other.counts) {
            *count += add;
        }
        if other.worst.map_or(0, |w| w.0) > self.worst.map_or(0, |w| w.0) {
            self.worst = other.worst;
        }
        self
    }

    fn total(&self) -> u64 {
        self.counts.iter().sum()
    }

    fn max_distance(&self) -> u64 {
        self.worst.map_or(0, |w| w.0)
    }

    fn print(&self, title: &str) {
        println!("{title}: ULP histogram over {} values", self.total());
        for (label, count) in Self::LABELS.iter().zip(self.counts) {
            println!("  {label:>17} ULP: {count}");
        }
        if let Some((distance, input)) = self.worst {
            println!("  worst: {distance} ULP at input bits {input:#010x}");
        }
    }
}

/// The shard `(index, count)` of a sweep selected by `RECIPROCAL_SWEEP_SHARD=index/count`,
/// so a sweep can be split across several short holds of the GPU lock. Defaults to all.
fn sweep_shard() -> (u64, u64) {
    let Ok(spec) = std::env::var("RECIPROCAL_SWEEP_SHARD") else {
        return (0, 1);
    };
    let (index, count) = spec.split_once('/').expect("RECIPROCAL_SWEEP_SHARD must be index/count");
    let (index, count) = (index.parse::<u64>().unwrap(), count.parse::<u64>().unwrap());
    assert!(index < count, "shard index out of range");
    (index, count)
}

/// Every one of the 2³² f32 bit patterns, in place, against the rayon reference.
#[test]
#[ignore = "exhaustive sweep; run under the exclusive GPU lock"]
fn sweep_real_exhaustive() {
    const BLOCK: usize = 1 << 24;
    let ctx = MetalContext::new().unwrap();
    let (shard, shards) = sweep_shard();
    let blocks = (1u64 << 32) / BLOCK as u64;
    let range = blocks * shard / shards..blocks * (shard + 1) / shards;
    let mut buffer = Shared::<f32>::zeroed(&ctx, BLOCK);
    let mut histogram = UlpHistogram::default();
    let (mut gpu_seconds, started) = (0.0, Instant::now());
    for block in range.clone() {
        let base = (block * BLOCK as u64) as u32;
        buffer
            .slice_mut()
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = f32::from_bits(base + i as u32));
        let dispatched = Instant::now();
        reciprocal(&ctx, &buffer.vector(), &buffer.vector()).unwrap();
        gpu_seconds += dispatched.elapsed().as_secs_f64();
        let block_histogram = buffer
            .slice()
            .par_iter()
            .enumerate()
            .fold(UlpHistogram::default, |histogram, (i, &y)| {
                let input = base + i as u32;
                let want = reference_real(f32::from_bits(input));
                histogram.record(input, ulp_distance(y, want))
            })
            .reduce(UlpHistogram::default, UlpHistogram::merge);
        histogram = histogram.merge(block_histogram);
    }
    histogram.print(&format!(
        "real exhaustive sweep, shard {shard}/{shards} (bit patterns {:#010x}..={:#010x})",
        range.start * BLOCK as u64,
        range.end * BLOCK as u64 - 1
    ));
    println!(
        "  {:.1} s total, {:.1} s in GPU calls ({:.0} M elements/s)",
        started.elapsed().as_secs_f64(),
        gpu_seconds,
        histogram.total() as f64 / gpu_seconds / 1e6
    );
    assert_eq!(histogram.total(), (range.end - range.start) * BLOCK as u64);
    assert!(within_real_gate(histogram.max_distance()));
}

/// Outcome of a complex sweep block: componentwise ULP histogram, gate violations, and the
/// worst normwise error (in u) with its input, over results in the normal range.
#[derive(Clone, Copy, Default)]
struct ComplexSweep {
    histogram: UlpHistogram,
    violations: u64,
    worst: f64,
    worst_input: [u32; 2],
}

impl ComplexSweep {
    fn record(mut self, z: C32, w: C32) -> Self {
        let error = ComplexError::measure(z, w);
        self.histogram = self.histogram.record(z[0].to_bits(), error.ulps);
        self.violations += u64::from(!error.within_gate());
        if error.in_normal_range() && error.normwise_u() > self.worst {
            self.worst = error.normwise_u();
            self.worst_input = bits(z);
        }
        self
    }

    fn merge(mut self, other: Self) -> Self {
        self.histogram = self.histogram.merge(other.histogram);
        self.violations += other.violations;
        if other.worst > self.worst {
            self.worst = other.worst;
            self.worst_input = other.worst_input;
        }
        self
    }
}

/// 2²⁸ seeded random complex inputs per exponent regime against the rayon reference.
#[test]
#[ignore = "large random sweep; run under the exclusive GPU lock"]
fn sweep_complex_random() {
    const BLOCK: usize = 1 << 22;
    const BLOCKS: u64 = 64;
    const CHUNK: usize = 1 << 14;
    let ctx = MetalContext::new().unwrap();
    let regimes: [(&str, std::ops::RangeInclusive<i32>, i32); 5] = [
        ("moderate exponents, independent parts", 87..=167, 80),
        ("comparable parts, full range", -22..=254, 1),
        ("independent parts, full range", -22..=254, 276),
        ("inputs near overflow", 240..=254, 14),
        ("inputs near underflow", -22..=10, 32),
    ];
    let mut src = Shared::<C32>::zeroed(&ctx, BLOCK);
    let dst = Shared::<C32>::zeroed(&ctx, BLOCK);
    let mut overall = 0.0f64;
    for (regime, (name, fields, spread)) in regimes.into_iter().enumerate() {
        let mut sweep = ComplexSweep::default();
        for block in 0..BLOCKS {
            src.slice_mut().par_chunks_mut(CHUNK).enumerate().for_each(|(chunk, out)| {
                let seed = (regime as u64) << 48 | block << 24 | chunk as u64;
                out.copy_from_slice(&complex_inputs(seed, out.len(), fields.clone(), spread));
            });
            reciprocal(&ctx, &src.vector(), &dst.vector()).unwrap();
            let block_sweep = src
                .slice()
                .par_iter()
                .zip(dst.slice().par_iter())
                .fold(ComplexSweep::default, |sweep, (&z, &w)| sweep.record(z, w))
                .reduce(ComplexSweep::default, ComplexSweep::merge);
            sweep = sweep.merge(block_sweep);
        }
        sweep.histogram.print(&format!("complex sweep, {name} (componentwise)"));
        println!(
            "  worst normwise error {:.4}u at input bits [{:#010x}, {:#010x}]; {} gate violations",
            sweep.worst, sweep.worst_input[0], sweep.worst_input[1], sweep.violations
        );
        assert_eq!(sweep.violations, 0);
        overall = overall.max(sweep.worst);
    }
    println!("complex sweep: worst normwise error overall {overall:.4}u (gate 4u)");
}

// ---------------------------------------------------------------------------------------
// Benchmarks (exclusive GPU lock)
// ---------------------------------------------------------------------------------------

/// Median wall time of `runs` synchronous calls of `src → dst`, in seconds.
fn median_seconds<T: ReciprocalElement>(
    ctx: &MetalContext,
    src: &TensorView<'_, T>,
    dst: &TensorView<'_, T>,
    runs: usize,
) -> f64 {
    reciprocal(ctx, src, dst).unwrap();
    let mut samples: Vec<f64> = (0..runs)
        .map(|_| {
            let started = Instant::now();
            reciprocal(ctx, src, dst).unwrap();
            started.elapsed().as_secs_f64()
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    samples[runs / 2]
}

fn bench_throughput<T: Element>(name: &str) {
    let ctx = MetalContext::new().unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(1000);
    println!("{name}: synchronous out-of-place call, contiguous vectors, median of 15");
    for log2 in [10, 14, 18, 22, 24] {
        let len = 1usize << log2;
        let src = Shared::from_slice(&ctx, &random_elements::<T>(&mut rng, len));
        let dst = Shared::<T>::zeroed(&ctx, len);
        let seconds = median_seconds(&ctx, &src.vector(), &dst.vector(), 15);
        println!(
            "  2^{log2:<2} elements: {:>10.1} us  {:>8.3} ns/element  {:>8.1} M elements/s",
            seconds * 1e6,
            seconds * 1e9 / len as f64,
            len as f64 / seconds / 1e6
        );
    }
    let diagonal = Layout::diagonal(128, 128, 1024, 0);
    let buffer = Shared::from_slice(&ctx, &random_elements::<T>(&mut rng, diagonal.extent()));
    let seconds = median_seconds(&ctx, &diagonal.view(&buffer), &diagonal.view(&buffer), 15);
    println!(
        "  1024 diagonals of 128x128 in place: {:.1} us ({:.3} ns/element)",
        seconds * 1e6,
        seconds * 1e9 / (1024.0 * 128.0)
    );
}

#[test]
#[ignore = "timing; run under the exclusive GPU lock"]
fn bench_real_throughput() {
    bench_throughput::<f32>("real f32");
}

#[test]
#[ignore = "timing; run under the exclusive GPU lock"]
fn bench_complex_throughput() {
    bench_throughput::<C32>("complex f32");
}
