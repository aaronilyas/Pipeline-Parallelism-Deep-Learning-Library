//! Correctness tests for `linalg::ops::sqrt`.
//!
//! Non-ignored tests compare the GPU result with `(x as f64).sqrt() as f32`. Finite values
//! and infinities must match bitwise (signed zeros included). Any NaN matches any NaN.
//! `sweep_*` tests are ignored; they walk every non-negative `f32` pattern and print a ULP
//! histogram. Run them under the exclusive GPU lock.

use linalg::abi::{
    alloc_shared, compute_pipeline, row_major_strides, LinalgError, MetalContext, TensorView,
};
use linalg::ops::sqrt::{encode_sqrt, sqrt};
use metal::MTLCommandBufferStatus;
use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

const SEED: u64 = 0x5A17_5A17;
const MAX_RANK: usize = 8;

fn ctx() -> MetalContext {
    MetalContext::new().expect("Metal device")
}

fn reference(x: f32) -> f32 {
    (x as f64).sqrt() as f32
}

fn same_bits(got: f32, expect: f32) -> bool {
    if got.is_nan() && expect.is_nan() {
        true
    } else {
        got.to_bits() == expect.to_bits()
    }
}

/// Ordered-integer ULP distance. `+0` and `-0` are one ULP apart, so sign-of-zero bugs are
/// visible. NaNs are not ordered: both NaN is 0, otherwise `u32::MAX`.
fn ulp_distance(a: f32, b: f32) -> u32 {
    if a.is_nan() || b.is_nan() {
        return if a.is_nan() && b.is_nan() { 0 } else { u32::MAX };
    }
    fn ordered(x: f32) -> i32 {
        let bits = x.to_bits() as i32;
        if bits < 0 {
            0x7FFF_FFFFi32.wrapping_sub(bits)
        } else {
            bits
        }
    }
    ordered(a).wrapping_sub(ordered(b)).unsigned_abs()
}

fn unravel(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut index = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        let extent = shape[axis];
        if extent == 0 {
            return index;
        }
        index[axis] = linear % extent;
        linear /= extent;
    }
    index
}

fn read_at(view: &TensorView<'_, f32>, linear: usize) -> f32 {
    let index = unravel(linear, &view.shape);
    let element = view.element_index(&index);
    // SAFETY: `linear < view.len()` for a validated view, and no command buffer is in flight.
    unsafe { view.host_ptr.add(element).read() }
}

fn write_at(view: &TensorView<'_, f32>, linear: usize, value: f32) {
    let index = unravel(linear, &view.shape);
    let element = view.element_index(&index);
    // SAFETY: `linear < view.len()` for a validated view, and no command buffer is in flight.
    unsafe { view.host_ptr.add(element).write(value) }
}

fn write_linear(view: &TensorView<'_, f32>, values: &[f32]) {
    assert_eq!(values.len(), view.len());
    for (linear, &value) in values.iter().enumerate() {
        write_at(view, linear, value);
    }
}

fn assert_against(inputs: &[f32], dst: &TensorView<'_, f32>) {
    assert_eq!(inputs.len(), dst.len());
    for (linear, &input) in inputs.iter().enumerate() {
        let got = read_at(dst, linear);
        let expect = reference(input);
        assert!(
            same_bits(got, expect),
            "linear {linear}: sqrt({input:?} {:08x}) got {got:?} {:08x}, expect {expect:?} {:08x}, ulp {}",
            input.to_bits(),
            got.to_bits(),
            expect.to_bits(),
            ulp_distance(got, expect)
        );
    }
}

fn snapshot_inputs(view: &TensorView<'_, f32>) -> Vec<f32> {
    (0..view.len()).map(|linear| read_at(view, linear)).collect()
}

fn run_views(ctx: &MetalContext, src: &TensorView<'_, f32>, dst: &TensorView<'_, f32>) {
    // Snapshot first so an in-place launch, which overwrites `src`, still has the inputs.
    let inputs = snapshot_inputs(src);
    sqrt(ctx, src, dst).unwrap();
    assert_against(&inputs, dst);
}

fn with_views(
    ctx: &MetalContext,
    shape: &[usize],
    src_offset: usize,
    src_strides: &[usize],
    dst_offset: usize,
    dst_strides: &[usize],
    body: impl FnOnce(&TensorView<'_, f32>, &TensorView<'_, f32>),
) {
    let src_len = span(shape, src_strides, src_offset);
    let dst_len = span(shape, dst_strides, dst_offset);
    let src_buf = alloc_shared::<f32>(ctx, src_len);
    let dst_buf = alloc_shared::<f32>(ctx, dst_len);
    let src = TensorView::<f32>::new(&src_buf, src_offset, shape.to_vec(), src_strides.to_vec());
    let dst = TensorView::<f32>::new(&dst_buf, dst_offset, shape.to_vec(), dst_strides.to_vec());
    body(&src, &dst);
}

fn span(shape: &[usize], strides: &[usize], offset: usize) -> usize {
    if shape.iter().any(|&extent| extent == 0) {
        return offset.max(1);
    }
    offset
        + shape.iter().zip(strides).map(|(&extent, &stride)| (extent - 1) * stride).sum::<usize>()
        + 1
}

fn compile_options() -> metal::CompileOptions {
    let opts = metal::CompileOptions::new();
    opts.set_fast_math_enabled(false);
    opts.set_language_version(metal::MTLLanguageVersion::V3_1);
    opts
}

const PROBE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

/// Identity through a float register: detects load/store subnormal flushing.
kernel void sqrt_probe_identity [[host_name("sqrt_probe_identity")]] (
    device uint* in [[buffer(0)]],
    device uint* out [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
    float x = as_type<float>(in[i]);
    out[i] = as_type<uint>(x);
}

/// Naive `precise::sqrt` with no subnormal scaling. This is the flush probe, not the
/// production kernel.
kernel void sqrt_probe_sqrt [[host_name("sqrt_probe_sqrt")]] (
    device uint* in [[buffer(0)]],
    device uint* out [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
    float x = as_type<float>(in[i]);
    out[i] = as_type<uint>(precise::sqrt(x));
}
"#;

fn dispatch_probe(ctx: &MetalContext, function: &str, inputs: &[u32]) -> Vec<u32> {
    let pipeline = compute_pipeline(ctx, "sqrt_probe", PROBE_MSL, compile_options, function)
        .unwrap_or_else(|err| panic!("probe pipeline {function}: {err}"));
    let n = inputs.len();
    let src = alloc_shared::<u32>(ctx, n);
    let dst = alloc_shared::<u32>(ctx, n);
    // SAFETY: fresh shared buffers of at least `n` elements, GPU not yet encoded.
    unsafe {
        std::ptr::copy_nonoverlapping(inputs.as_ptr(), src.contents() as *mut u32, n);
    }
    let cmd = ctx.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&src), 0);
    enc.set_buffer(1, Some(&dst), 0);
    let threads = metal::MTLSize::new(n as u64, 1, 1);
    enc.dispatch_threads(threads, threads);
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed);
    // SAFETY: the command buffer completed and `dst` holds `n` u32s.
    unsafe { std::slice::from_raw_parts(dst.contents() as *const u32, n).to_vec() }
}

#[test]
fn probe_subnormal_flush_is_pinned() {
    let ctx = ctx();
    // Positive and negative subnormals, both zeros, a normal power of two, and sqrt(2)'s input.
    let inputs = [
        0x0000_0001u32,
        0x0000_0002,
        0x0040_0000,
        0x007F_FFFF,
        0x8000_0001,
        0x807F_FFFF,
        0x0000_0000,
        0x8000_0000,
        0x0080_0000, // min normal, 2^-126
        0x4000_0000, // 2
    ];
    let identity = dispatch_probe(&ctx, "sqrt_probe_identity", &inputs);
    let sqrt_out = dispatch_probe(&ctx, "sqrt_probe_sqrt", &inputs);

    // Pinned on Apple M2 (Metal 4, family Apple8), MSL 3.1, fast math off:
    // a float load/store keeps subnormal bits, and `precise::sqrt` flushes a subnormal input
    // to a signed zero. The same bits were observed with mathMode safe and with fast math on.
    // Normals are still correctly rounded.
    assert_eq!(identity, inputs, "load/store flushed or rewrote bits");
    let expect_sqrt = [
        0x0000_0000u32, // +subnormal -> +0
        0x0000_0000,
        0x0000_0000,
        0x0000_0000,
        0x8000_0000, // -subnormal -> -0 (not NaN)
        0x8000_0000,
        0x0000_0000, // +0
        0x8000_0000, // -0
        0x2000_0000, // sqrt(2^-126) = 2^-63
        0x3FB5_04F3, // sqrt(2)
    ];
    assert_eq!(sqrt_out, expect_sqrt, "precise::sqrt probe drifted from the pinned bits");
    println!(
        "subnormal-flush probe: load/store preserves subnormals; precise::sqrt flushes \
         +subnormal to +0 and -subnormal to -0; normals stay correctly rounded \
         (sqrt(2) = 0x3fb504f3)"
    );
}

#[test]
fn domain_specials() {
    let ctx = ctx();
    let bits = [
        0x0000_0000u32, // +0
        0x8000_0000,    // -0
        0x7F80_0000,    // +inf
        0xFF80_0000,    // -inf
        0x7FC0_0000,    // +qNaN
        0xFFC0_0000,    // -qNaN
        0x7F80_0001,    // payload NaN
        0x7FFF_FFFF,    // all-ones NaN
        0xBF80_0000,    // -1
        0xFF7F_FFFF,    // most negative finite
        0x8000_0001,    // -min subnormal
        0x807F_FFFF,    // -max subnormal
        0x0000_0001,    // +min subnormal
        0x007F_FFFF,    // +max subnormal
        0x0080_0000,    // +min normal
        0x3F80_0000,    // 1
        0x4000_0000,    // 2
        0x7F7F_FFFF,    // max finite
        0x3F00_0000,    // 0.5
    ];
    let values: Vec<f32> = bits.into_iter().map(f32::from_bits).collect();
    let shape = [values.len()];
    let strides = [1usize];
    with_views(&ctx, &shape, 0, &strides, 0, &strides, |src, dst| {
        write_linear(src, &values);
        run_views(&ctx, src, dst);
    });
}

#[test]
fn stratified_every_binade() {
    let ctx = ctx();
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);
    let mut bits = Vec::new();
    for sign in [0u32, 0x8000_0000] {
        for exp in 0u32..=255 {
            let base = sign | (exp << 23);
            bits.push(base);
            bits.push(base | 1);
            bits.push(base | 0x007F_FFFF);
            bits.push(base | 0x002A_AAAA);
            bits.push(base | (rng.random::<u32>() & 0x007F_FFFF));
        }
    }
    bits.sort_unstable();
    bits.dedup();
    let values: Vec<f32> = bits.iter().copied().map(f32::from_bits).collect();
    let shape = [values.len()];
    let strides = [1usize];
    with_views(&ctx, &shape, 3, &strides, 9, &strides, |src, dst| {
        write_linear(src, &values);
        // Poison the destination, including the padding before offset 9.
        // SAFETY: the allocation is zeroed and large enough; no GPU work is in flight.
        unsafe {
            let n = dst.buffer.length() as usize / 4;
            let ptr = dst.buffer.contents() as *mut u32;
            for i in 0..n {
                ptr.add(i).write(0x7FFF_FFFFu32);
            }
        }
        run_views(&ctx, src, dst);
    });
}

#[test]
fn all_subnormals_match_reference() {
    let ctx = ctx();
    // Every positive subnormal, then every negative subnormal. Zeros are in `domain_specials`.
    for sign in [0u32, 0x8000_0000] {
        let n = 0x007F_FFFFusize;
        let src_buf = alloc_shared::<f32>(&ctx, n);
        let dst_buf = alloc_shared::<f32>(&ctx, n);
        // SAFETY: buffers hold `n` f32s and the GPU has not been encoded.
        unsafe {
            let ptr = src_buf.contents() as *mut u32;
            for i in 0..n {
                ptr.add(i).write(sign | ((i as u32) + 1));
            }
        }
        let src = TensorView::<f32>::row_major(&src_buf, 0, vec![n]);
        let dst = TensorView::<f32>::row_major(&dst_buf, 0, vec![n]);
        sqrt(&ctx, &src, &dst).unwrap();
        // SAFETY: `sqrt` waited until the command buffer completed.
        let inputs = unsafe { std::slice::from_raw_parts(src_buf.contents() as *const f32, n) };
        let got = unsafe { std::slice::from_raw_parts(dst_buf.contents() as *const f32, n) };
        let miss = (0..n).into_par_iter().find_first(|&i| !same_bits(got[i], reference(inputs[i])));
        if let Some(i) = miss {
            let input = inputs[i];
            let expect = reference(input);
            panic!(
                "subnormal sign {sign:08x} index {i}: sqrt({:08x}) got {:08x} expect {:08x}",
                input.to_bits(),
                got[i].to_bits(),
                expect.to_bits()
            );
        }
    }
}

#[test]
fn scalars_rank0_and_strided_vectors() {
    let ctx = ctx();
    let mut rng = ChaCha8Rng::seed_from_u64(SEED ^ 0x51);
    let buf = alloc_shared::<f32>(&ctx, 64);
    let scalar = TensorView::<f32>::new(&buf, 4, vec![], vec![]);
    write_at(&scalar, 0, 9.0);
    let out_buf = alloc_shared::<f32>(&ctx, 8);
    let out = TensorView::<f32>::new(&out_buf, 2, vec![], vec![]);
    run_views(&ctx, &scalar, &out);
    // In place, same rank-0 view.
    write_at(&scalar, 0, 16.0);
    run_views(&ctx, &scalar, &scalar);

    let n = 32usize;
    let values: Vec<f32> = (0..n)
        .map(|_| {
            let exp = rng.random_range(1u32..254);
            let mant = rng.random::<u32>() & 0x007F_FFFF;
            f32::from_bits((exp << 23) | mant)
        })
        .collect();
    for stride in [1usize, 2, 7, 15] {
        let shape = [n];
        let strides = [stride];
        with_views(&ctx, &shape, 1, &strides, 0, &[1], |src, dst| {
            write_linear(src, &values);
            run_views(&ctx, src, dst);
        });
        // In place on the strided view.
        let inplace_buf = alloc_shared::<f32>(&ctx, span(&shape, &strides, 1));
        let inplace = TensorView::<f32>::new(&inplace_buf, 1, shape.to_vec(), strides.to_vec());
        write_linear(&inplace, &values);
        run_views(&ctx, &inplace, &inplace);
    }
}

#[test]
fn diagonals_with_ld_offset_and_batch() {
    let ctx = ctx();
    let mut rng = ChaCha8Rng::seed_from_u64(SEED ^ 0xD1A6);
    let specials = [
        0.0f32,
        -0.0,
        f32::from_bits(1),
        f32::from_bits(0x007F_FFFF),
        1.0,
        2.0,
        f32::INFINITY,
        -1.0,
        f32::NAN,
        f32::from_bits(0x8000_0001),
    ];
    for &ld in &[1usize, 7, 128] {
        for &offset in &[0usize, 5] {
            for &batch in &[1usize, 4] {
                let n = 6usize;
                let diag_stride = ld + 1;
                let batch_stride = (n - 1) * diag_stride + 4;
                let shape = [batch, n];
                let src_strides = [batch_stride, diag_stride];
                let count = batch * n;
                let mut values = Vec::with_capacity(count);
                for i in 0..count {
                    if i < specials.len() {
                        values.push(specials[i]);
                    } else {
                        let exp = rng.random_range(1u32..60);
                        let mant = rng.random::<u32>() & 0x007F_FFFF;
                        values.push(f32::from_bits((exp << 23) | mant));
                    }
                }
                let storage = span(&shape, &src_strides, offset);
                let buf = alloc_shared::<f32>(&ctx, storage);
                // SAFETY: `buf` holds `storage` f32s, GPU idle.
                unsafe {
                    let words = buf.length() as usize / 4;
                    let ptr = buf.contents() as *mut u32;
                    for i in 0..words {
                        ptr.add(i).write(0x1234_5678);
                    }
                }
                let diag =
                    TensorView::<f32>::new(&buf, offset, shape.to_vec(), src_strides.to_vec());
                write_linear(&diag, &values);
                let before = snapshot(&buf);

                // Out of place into a packed row-major view.
                let dst_buf = alloc_shared::<f32>(&ctx, count + 2);
                let dst = TensorView::<f32>::row_major(&dst_buf, 0, shape.to_vec());
                sqrt(&ctx, &diag, &dst).unwrap();
                assert_against(&values, &dst);
                assert_eq!(snapshot(&buf), before, "out-of-place sqrt wrote the source, ld {ld}");

                // In place on the diagonal. Off-diagonal sentinels must survive.
                sqrt(&ctx, &diag, &diag).unwrap();
                assert_against(&values, &diag);
                let after = snapshot(&buf);
                let touched = touched_indices(&shape, &src_strides, offset);
                for (word, (&left, &right)) in before.iter().zip(after.iter()).enumerate() {
                    if !touched.contains(&word) {
                        assert_eq!(
                            left, right,
                            "off-diagonal word {word} changed for ld {ld} offset {offset} batch {batch}"
                        );
                    }
                }
            }
        }
    }
}

fn snapshot(buf: &metal::BufferRef) -> Vec<u32> {
    let n = buf.length() as usize / 4;
    // SAFETY: shared buffer, no GPU work in flight. Caller sequences this around `sqrt`.
    unsafe { std::slice::from_raw_parts(buf.contents() as *const u32, n).to_vec() }
}

fn touched_indices(
    shape: &[usize],
    strides: &[usize],
    offset: usize,
) -> std::collections::HashSet<usize> {
    let mut set = std::collections::HashSet::new();
    let total: usize = shape.iter().product();
    for linear in 0..total {
        let index = unravel(linear, shape);
        let element = offset + index.iter().zip(strides).map(|(i, s)| i * s).sum::<usize>();
        set.insert(element);
    }
    set
}

#[test]
fn inplace_matches_out_of_place() {
    let ctx = ctx();
    let mut rng = ChaCha8Rng::seed_from_u64(SEED ^ 0x1111);
    let n = 128usize;
    let values: Vec<f32> = (0..n)
        .map(|i| {
            if i % 17 == 0 {
                f32::from_bits(i as u32 + 1) // subnormals and small values
            } else {
                let exp = rng.random_range(1u32..254);
                f32::from_bits((exp << 23) | (rng.random::<u32>() & 0x007F_FFFF))
            }
        })
        .collect();
    let shape = [n];
    let strides = [3usize];
    let src_buf = alloc_shared::<f32>(&ctx, span(&shape, &strides, 2));
    let src = TensorView::<f32>::new(&src_buf, 2, shape.to_vec(), strides.to_vec());
    write_linear(&src, &values);
    let copy = snapshot(&src_buf);

    let dst_buf = alloc_shared::<f32>(&ctx, n);
    let dst = TensorView::<f32>::row_major(&dst_buf, 0, vec![n]);
    sqrt(&ctx, &src, &dst).unwrap();

    // Restore and run in place.
    // SAFETY: GPU idle; `copy` has one word per allocated element.
    unsafe {
        std::ptr::copy_nonoverlapping(copy.as_ptr(), src_buf.contents() as *mut u32, copy.len());
    }
    sqrt(&ctx, &src, &src).unwrap();
    for linear in 0..n {
        let inplace = read_at(&src, linear);
        let outplace = read_at(&dst, linear);
        let expect = reference(values[linear]);
        assert!(
            same_bits(inplace, outplace) && same_bits(inplace, expect),
            "linear {linear}: in-place {:08x} out-of-place {:08x} reference {:08x}",
            inplace.to_bits(),
            outplace.to_bits(),
            expect.to_bits()
        );
    }
}

#[test]
fn encode_sqrt_does_not_commit() {
    let ctx = ctx();
    let n = 64usize;
    let src_buf = alloc_shared::<f32>(&ctx, n);
    let dst_buf = alloc_shared::<f32>(&ctx, n);
    let src = TensorView::<f32>::row_major(&src_buf, 0, vec![n]);
    let dst = TensorView::<f32>::row_major(&dst_buf, 0, vec![n]);
    let values: Vec<f32> = (0..n).map(|i| (i as f32) + 0.25).collect();
    write_linear(&src, &values);
    let poison = 0xDEC0_0DEDu32;
    // SAFETY: GPU idle, buffer holds `n` elements.
    unsafe {
        let ptr = dst_buf.contents() as *mut u32;
        for i in 0..n {
            ptr.add(i).write(poison);
        }
    }
    let cmd = ctx.queue.new_command_buffer();
    encode_sqrt(&ctx, cmd, &src, &dst).unwrap();
    encode_sqrt(&ctx, cmd, &src, &dst).unwrap();
    assert_eq!(cmd.status(), MTLCommandBufferStatus::NotEnqueued);
    let mid = snapshot(&dst_buf);
    assert!(mid.iter().take(n).all(|&word| word == poison), "encode_sqrt ran before commit");
    cmd.commit();
    cmd.wait_until_completed();
    assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed);
    assert_against(&values, &dst);
}

#[test]
fn partial_overlap_is_rejected_and_identical_view_is_inplace() {
    let ctx = ctx();
    let buf = alloc_shared::<f32>(&ctx, 64);
    let src = TensorView::<f32>::row_major(&buf, 0, vec![32]);
    let dst = TensorView::<f32>::row_major(&buf, 16, vec![32]);
    write_linear(&src, &[4.0; 32]);
    let before = snapshot(&buf);
    let err = sqrt(&ctx, &src, &dst).unwrap_err();
    assert!(matches!(err, LinalgError::Overlap(_)), "{err}");
    assert_eq!(snapshot(&buf), before);

    // Same elements, different shape: not the identical view.
    let alt = TensorView::<f32>::new(&buf, 0, vec![4, 8], vec![8, 1]);
    let err = sqrt(&ctx, &src, &alt).unwrap_err();
    assert!(matches!(err, LinalgError::Overlap(_)), "{err}");

    // Adjacent, non-overlapping ranges are legal.
    let left = TensorView::<f32>::row_major(&buf, 0, vec![16]);
    let right = TensorView::<f32>::row_major(&buf, 16, vec![16]);
    write_linear(&left, &[25.0; 16]);
    run_views(&ctx, &left, &right);

    // Identical metadata is in place, including a non-zero offset and stride.
    let strided = TensorView::<f32>::new(&buf, 2, vec![8], vec![2]);
    write_linear(&strided, &[36.0; 8]);
    run_views(&ctx, &strided, &strided);
}

#[test]
fn invalid_metadata_and_noninjective_dst() {
    let ctx = ctx();
    let buf = alloc_shared::<f32>(&ctx, 32);
    let other = alloc_shared::<f32>(&ctx, 32);
    let src = TensorView::<f32>::row_major(&buf, 0, vec![8]);
    let dst = TensorView::<f32>::row_major(&other, 0, vec![4]);
    let err = sqrt(&ctx, &src, &dst).unwrap_err();
    assert!(matches!(err, LinalgError::InvalidArgument(_)), "{err}");

    let bad = TensorView::<f32>::new(&buf, 0, vec![4, 4], vec![4]);
    let err = sqrt(&ctx, &bad, &bad).unwrap_err();
    assert!(matches!(err, LinalgError::InvalidArgument(_)), "{err}");

    let mut lied = TensorView::<f32>::row_major(&buf, 0, vec![4]);
    lied.host_ptr = lied.host_ptr.wrapping_add(1);
    let err = sqrt(&ctx, &lied, &lied).unwrap_err();
    assert!(matches!(err, LinalgError::InvalidArgument(_)), "{err}");

    // The shared allocation is rounded up to 16 KB, so the offset has to sit near the real end.
    let capacity = buf.length() as usize / std::mem::size_of::<f32>();
    let past = TensorView::<f32>::row_major(&buf, capacity - 2, vec![8]);
    let err = sqrt(&ctx, &past, &past).unwrap_err();
    assert!(matches!(err, LinalgError::InvalidArgument(_)), "{err}");

    // Zero stride with extent > 1 aliases every store. Shapes match so this is not a mismatch.
    let src4 = TensorView::<f32>::row_major(&other, 0, vec![4]);
    let broadcast = TensorView::<f32>::new(&buf, 0, vec![4], vec![0]);
    let err = sqrt(&ctx, &src4, &broadcast).unwrap_err();
    assert!(matches!(err, LinalgError::InvalidArgument(_)), "{err}");

    // shape [2, 2], strides [2, 2]: (1, 0) and (0, 1) land on the same element.
    let alias = TensorView::<f32>::new(&buf, 0, vec![2, 2], vec![2, 2]);
    let err = sqrt(&ctx, &alias, &alias).unwrap_err();
    assert!(matches!(err, LinalgError::InvalidArgument(_)), "{err}");

    let ones = vec![1usize; MAX_RANK + 1];
    let high = TensorView::<f32>::new(&buf, 0, ones.clone(), ones);
    let err = sqrt(&ctx, &high, &high).unwrap_err();
    assert!(matches!(err, LinalgError::InvalidArgument(_)), "{err}");

    // A legal broadcast read (source stride 0) into an injective destination.
    let src_b = TensorView::<f32>::new(&buf, 0, vec![4], vec![0]);
    // Every logical index aliases element 0, so one store fills the view.
    write_at(&src_b, 0, 49.0);
    let dst_b = TensorView::<f32>::row_major(&other, 0, vec![4]);
    run_views(&ctx, &src_b, &dst_b);
}

#[test]
fn empty_view_is_a_noop() {
    let ctx = ctx();
    let buf = alloc_shared::<f32>(&ctx, 8);
    let other = alloc_shared::<f32>(&ctx, 8);
    // SAFETY: GPU idle.
    unsafe {
        std::ptr::write_bytes(buf.contents() as *mut u8, 0xAB, buf.length() as usize);
        std::ptr::write_bytes(other.contents() as *mut u8, 0xCD, other.length() as usize);
    }
    let src = TensorView::<f32>::new(&buf, 3, vec![0, 4], vec![4, 1]);
    let dst = TensorView::<f32>::new(&other, 1, vec![0, 4], vec![4, 1]);
    let before_src = snapshot(&buf);
    let before_dst = snapshot(&other);
    sqrt(&ctx, &src, &dst).unwrap();
    assert_eq!(snapshot(&buf), before_src);
    assert_eq!(snapshot(&other), before_dst);
}

#[test]
fn higher_rank_strided_layouts() {
    let ctx = ctx();
    let mut rng = ChaCha8Rng::seed_from_u64(SEED ^ 0x0A11);
    for rank in 1..=MAX_RANK {
        let shape: Vec<usize> =
            (0..rank).map(|axis| if axis == rank - 1 { 5 } else { 2 }).collect();
        let packed = row_major_strides(&shape);
        let src_strides: Vec<usize> = packed.iter().map(|stride| stride * 3).collect();
        let count: usize = shape.iter().product();
        let values: Vec<f32> = (0..count)
            .map(|_| {
                let exp = rng.random_range(1u32..40);
                f32::from_bits((exp << 23) | (rng.random::<u32>() & 0x0000_FFFF))
            })
            .collect();
        with_views(&ctx, &shape, 2, &src_strides, 0, &packed, |src, dst| {
            write_linear(src, &values);
            run_views(&ctx, src, dst);
        });
    }
}

#[test]
#[ignore = "exhaustive 2^31 non-negative patterns; hold /tmp/linalg-gpu.lock"]
fn sweep_nonnegative_patterns() {
    let ctx = ctx();
    const CHUNK: usize = 1 << 22;
    const TOTAL: u64 = 1 << 31;
    let src_buf = alloc_shared::<f32>(&ctx, CHUNK);
    let dst_buf = alloc_shared::<f32>(&ctx, CHUNK);
    let started = std::time::Instant::now();
    let mut hist = [0u64; 5];
    let mut first_bad: Option<(u32, u32, u32)> = None;
    let mut start = 0u64;
    while start < TOTAL {
        let n = ((TOTAL - start) as usize).min(CHUNK);
        // SAFETY: buffers hold CHUNK >= n elements; the previous command buffer has completed
        // because `sqrt` waits, and this is the first touch of the chunk.
        unsafe {
            let ptr = src_buf.contents() as *mut u32;
            for i in 0..n {
                ptr.add(i).write((start as u32).wrapping_add(i as u32));
            }
        }
        let src = TensorView::<f32>::row_major(&src_buf, 0, vec![n]);
        let dst = TensorView::<f32>::row_major(&dst_buf, 0, vec![n]);
        sqrt(&ctx, &src, &dst).unwrap();
        // SAFETY: `sqrt` waited until completion. Both slices live until this iteration ends.
        let inputs = unsafe { std::slice::from_raw_parts(src_buf.contents() as *const f32, n) };
        let got = unsafe { std::slice::from_raw_parts(dst_buf.contents() as *const f32, n) };
        let local = (0..n)
            .into_par_iter()
            .fold(
                || [0u64; 5],
                |mut acc, i| {
                    let expect = reference(inputs[i]);
                    let y = got[i];
                    if expect.is_nan() {
                        if y.is_nan() {
                            acc[0] += 1;
                        } else {
                            acc[4] += 1;
                        }
                    } else {
                        let bucket = (ulp_distance(y, expect) as usize).min(4);
                        acc[bucket] += 1;
                    }
                    acc
                },
            )
            .reduce(
                || [0u64; 5],
                |mut a, b| {
                    for slot in 0..5 {
                        a[slot] += b[slot];
                    }
                    a
                },
            );
        if first_bad.is_none() && local[1..].iter().any(|&count| count != 0) {
            for i in 0..n {
                let expect = reference(inputs[i]);
                if !same_bits(got[i], expect) {
                    first_bad = Some((inputs[i].to_bits(), got[i].to_bits(), expect.to_bits()));
                    break;
                }
            }
        }
        for slot in 0..5 {
            hist[slot] += local[slot];
        }
        start += n as u64;
        if start % (1 << 28) == 0 || start == TOTAL {
            eprintln!(
                "sweep_nonnegative_patterns {start}/{TOTAL} elapsed {:.1}s",
                started.elapsed().as_secs_f64()
            );
        }
    }
    println!("sqrt ULP histogram over {TOTAL} non-negative f32 patterns");
    println!("  ulp 0: {}", hist[0]);
    println!("  ulp 1: {}", hist[1]);
    println!("  ulp 2: {}", hist[2]);
    println!("  ulp 3: {}", hist[3]);
    println!("  ulp >= 4 (or non-NaN where NaN was required): {}", hist[4]);
    println!("  max finite ulp bucket: {}", max_bucket(&hist));
    println!("  elapsed {:.1}s", started.elapsed().as_secs_f64());
    if let Some((input, got, expect)) = first_bad {
        println!("  first mismatch: sqrt({input:08x}) = {got:08x}, reference {expect:08x}");
    }
    // Bitwise equality on every pattern (NaN matching any NaN) is the IEEE rounding target
    // and rules out an rsqrt approximation, which misses by more than 1 ULP on many inputs.
    assert_eq!(hist[1] + hist[2] + hist[3] + hist[4], 0, "ulp histogram {hist:?}");
    assert_eq!(hist[0], TOTAL);
}

fn max_bucket(hist: &[u64; 5]) -> usize {
    hist.iter().rposition(|&count| count != 0).unwrap_or(0)
}
