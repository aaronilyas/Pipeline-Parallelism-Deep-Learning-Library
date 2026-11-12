//! Right-looking LU and one Cholesky step, composed only from the six primitives.
//!
//! The LU is unblocked partial pivoting on a 256×256 f32 matrix. Each column uses two
//! command buffers: one for the pivot argmax, then one for the row swap, the reciprocal
//! scale, and the trailing update. The pivot index is a host read, so the swap cannot
//! share the argmax buffer. The Cholesky diagonal step is a single command buffer.

use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use linalg::abi::alloc_shared;
use linalg::abi::LinalgError;
use linalg::abi::MetalContext;
use linalg::abi::TensorView;
use linalg::ops::argmax::encode_argmax;
use linalg::ops::contraction::encode_gemm;
use linalg::ops::permutation::encode_materialize;
use linalg::ops::reciprocal::encode_reciprocal;
use linalg::ops::slicing::slice;
use linalg::ops::sqrt::encode_sqrt;

const LU_N: usize = 256;
const CHOLESKY_N: usize = 32;

fn context() -> MetalContext {
    MetalContext::new().expect("a Metal device is required")
}

fn must(result: Result<(), LinalgError>, what: &str) {
    if let Err(err) = result {
        panic!("{what}: {err}");
    }
}

fn commit(cmd: &metal::CommandBufferRef) {
    cmd.commit();
    cmd.wait_until_completed();
    assert_eq!(
        cmd.status(),
        metal::MTLCommandBufferStatus::Completed,
        "command buffer status {:?}",
        cmd.status()
    );
}

fn write_f32(buf: &metal::BufferRef, values: &[f32]) {
    assert!(buf.length() as usize >= std::mem::size_of_val(values));
    // SAFETY: `buf` is a shared allocation at least as long as `values`, and no GPU work is in flight.
    unsafe {
        std::ptr::copy_nonoverlapping(values.as_ptr(), buf.contents() as *mut f32, values.len());
    }
}

fn read_f32(buf: &metal::BufferRef, len: usize) -> Vec<f32> {
    assert!(buf.length() as usize >= len * std::mem::size_of::<f32>());
    // SAFETY: the buffer holds `len` f32s and the caller has waited for the GPU.
    unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, len).to_vec() }
}

fn read_u32(buf: &metal::BufferRef) -> u32 {
    // SAFETY: the buffer holds one u32 and the caller has waited for the GPU.
    unsafe { *(buf.contents() as *const u32) }
}

fn swap_rows(a: &mut [f32], n: usize, i: usize, j: usize) {
    if i != j {
        for col in 0..n {
            a.swap(i * n + col, j * n + col);
        }
    }
}

/// Diagonally dominant matrix, then a few whole-row swaps so the first pivot is not row 0.
fn planted_matrix(n: usize, seed: u64, swaps: &[(usize, usize)]) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut value = rng.random_range(-1.0f32..=1.0);
            if i == j {
                value += n as f32;
            }
            a[i * n + j] = value;
        }
    }
    for &(i, j) in swaps {
        swap_rows(&mut a, n, i, j);
    }
    a
}

fn symmetric_dominant(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            let value = if i == j { n as f32 } else { rng.random_range(-0.5f32..=0.5) };
            a[i * n + j] = value;
            a[j * n + i] = value;
        }
    }
    a
}

/// `‖PA − LU‖_F / ‖A‖_F` from an in-place factorization. `perm[i]` is the original row now at `i`.
fn lu_residual(original: &[f32], lu: &[f32], perm: &[usize], n: usize) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f64;
            for t in 0..=i.min(j) {
                let lower = if i == t { 1.0 } else { f64::from(lu[i * n + t]) };
                acc += lower * f64::from(lu[t * n + j]);
            }
            let diff = f64::from(original[perm[i] * n + j]) - acc;
            num += diff * diff;
            den += f64::from(original[perm[i] * n + j]).powi(2);
        }
    }
    (num / den).sqrt()
}

/// Unblocked right-looking LU with partial pivoting. Returns the command-buffer count.
fn factor_lu(ctx: &MetalContext, original: &[f32]) -> (Vec<f32>, Vec<usize>, usize) {
    let n = LU_N;
    let matrix_buf = alloc_shared::<f32>(ctx, n * n);
    let index_buf = alloc_shared::<u32>(ctx, 1);
    let scalar_buf = alloc_shared::<f32>(ctx, 1);
    let row_buf = alloc_shared::<f32>(ctx, n);
    write_f32(&matrix_buf, original);

    let mut perm: Vec<usize> = (0..n).collect();
    let mut buffers = 0usize;
    for k in 0..n - 1 {
        metal::objc::rc::autoreleasepool(|| {
            let matrix = TensorView::<f32>::row_major(&matrix_buf, 0, vec![n, n]);
            let panel = slice(&matrix, &[k..n, k..k + 1]).expect("pivot column slice");
            // Argmax takes a rank-1 column. The slice keeps the unit axis, so drop it.
            let column = TensorView::<f32>::new(
                panel.buffer,
                panel.offset,
                vec![n - k],
                vec![panel.strides[0]],
            );
            let pivot_slot = TensorView::<u32>::row_major(&index_buf, 0, vec![1]);
            let search = ctx.queue.new_command_buffer();
            must(encode_argmax(ctx, search, &column, &pivot_slot), "pivot argmax");
            commit(search);
            buffers += 1;

            let relative = read_u32(&index_buf) as usize;
            assert!(relative < n - k, "pivot {relative} is outside the remaining column");
            let piv = k + relative;

            let update = ctx.queue.new_command_buffer();
            if piv != k {
                let from = slice(&matrix, &[piv..piv + 1, 0..n]).expect("pivot row");
                let temp = TensorView::<f32>::row_major(&row_buf, 0, vec![1, n]);
                let row_k = slice(&matrix, &[k..k + 1, 0..n]).expect("diagonal row");
                must(encode_materialize(ctx, update, &from, &temp), "save pivot row");
                must(encode_materialize(ctx, update, &row_k, &from), "write row k over the pivot");
                must(encode_materialize(ctx, update, &temp, &row_k), "restore the pivot row");
                perm.swap(k, piv);
            }
            let pivot = slice(&matrix, &[k..k + 1, k..k + 1]).expect("pivot");
            let inv = TensorView::<f32>::row_major(&scalar_buf, 0, vec![1, 1]);
            must(encode_reciprocal(ctx, update, &pivot, &inv), "pivot reciprocal");
            let below = slice(&matrix, &[k + 1..n, k..k + 1]).expect("multiplier column");
            must(encode_gemm(ctx, update, 1.0, &below, &inv, 0.0, &below), "scale multipliers");
            let right = slice(&matrix, &[k..k + 1, k + 1..n]).expect("pivot row tail");
            let trail = slice(&matrix, &[k + 1..n, k + 1..n]).expect("trailing block");
            must(encode_gemm(ctx, update, -1.0, &below, &right, 1.0, &trail), "trailing update");
            commit(update);
            buffers += 1;
        });
    }
    (read_f32(&matrix_buf, n * n), perm, buffers)
}

#[test]
fn lu_256_matches_an_f64_residual() {
    let swaps = [(0usize, 40usize), (1, 80), (2, 200)];
    let original = planted_matrix(LU_N, 11, &swaps);
    let ctx = context();
    let (lu, perm, buffers) = factor_lu(&ctx, &original);
    let residual = lu_residual(&original, &lu, &perm, LU_N);
    let swaps_used = perm.iter().enumerate().filter(|(i, &p)| *i != p).count();
    println!(
        "[composition] LU n={LU_N}: ||PA-LU||/||A|| = {residual:.3e}, \
         permuted rows = {swaps_used}, command buffers = {buffers}"
    );
    assert_eq!(perm[0], 40, "the planted first pivot must win the column search");
    assert!(swaps_used > 1, "partial pivoting swapped only the planted first row");
    assert_eq!(buffers, 2 * (LU_N - 1), "one search buffer and one update buffer per column");
    // The matrix is a row permutation of a diagonally dominant matrix, so growth stays small.
    // 8·n·ε covers the f32 trailing products without accepting a wrong update.
    let limit = 8.0 * LU_N as f64 * f64::from(f32::EPSILON);
    assert!(residual <= limit, "LU residual {residual:.3e} exceeds {limit:.3e}");
}

#[test]
fn cholesky_diagonal_step_uses_sqrt() {
    let n = CHOLESKY_N;
    let original = symmetric_dominant(n, 19);
    let ctx = context();
    let matrix_buf = alloc_shared::<f32>(&ctx, n * n);
    let root_buf = alloc_shared::<f32>(&ctx, 1);
    let inv_buf = alloc_shared::<f32>(&ctx, 1);
    write_f32(&matrix_buf, &original);

    let buffers = metal::objc::rc::autoreleasepool(|| {
        let matrix = TensorView::<f32>::row_major(&matrix_buf, 0, vec![n, n]);
        let diag = slice(&matrix, &[0..1, 0..1]).expect("diagonal");
        let root = TensorView::<f32>::row_major(&root_buf, 0, vec![1, 1]);
        let inv = TensorView::<f32>::row_major(&inv_buf, 0, vec![1, 1]);
        let below = slice(&matrix, &[1..n, 0..1]).expect("lower column");
        let trail = slice(&matrix, &[1..n, 1..n]).expect("trailing block");
        // The column, read with a unit leading extent, is the transpose the outer product needs.
        let across = TensorView::<f32>::new(
            below.buffer,
            below.offset,
            vec![1, n - 1],
            vec![1, below.strides[0]],
        );
        let cmd = ctx.queue.new_command_buffer();
        must(encode_sqrt(&ctx, cmd, &diag, &root), "diagonal sqrt");
        must(encode_reciprocal(&ctx, cmd, &root, &inv), "reciprocal of the sqrt");
        must(encode_gemm(&ctx, cmd, 1.0, &below, &inv, 0.0, &below), "scale the column");
        must(encode_gemm(&ctx, cmd, -1.0, &below, &across, 1.0, &trail), "rank-1 update");
        commit(cmd);
        1usize
    });

    let root = read_f32(&root_buf, 1)[0];
    let got = read_f32(&matrix_buf, n * n);
    let a00 = f64::from(original[0]);
    let expect_root = (a00.sqrt()) as f32;
    println!(
        "[composition] Cholesky step n={n}: sqrt got {root:.8} expect {expect_root:.8}, \
         command buffers = {buffers}"
    );
    assert_eq!(buffers, 1);
    assert_eq!(root.to_bits(), expect_root.to_bits(), "sqrt must be correctly rounded");
    assert_eq!(got[0].to_bits(), original[0].to_bits(), "the stored diagonal is not replaced");

    let l00 = a00.sqrt();
    let mut column_err = 0.0f64;
    for i in 1..n {
        let want = f64::from(original[i * n]) / l00;
        column_err = column_err.max((f64::from(got[i * n]) - want).abs());
    }
    let mut trail_err = 0.0f64;
    for i in 1..n {
        let li = f64::from(original[i * n]) / l00;
        for j in 1..n {
            let lj = f64::from(original[j * n]) / l00;
            let want = f64::from(original[i * n + j]) - li * lj;
            trail_err = trail_err.max((f64::from(got[i * n + j]) - want).abs());
        }
    }
    println!("[composition] Cholesky column err {column_err:.3e}, trailing err {trail_err:.3e}");
    // One reciprocal and one product, then one product-and-subtract. A few ulps of the diagonal.
    let limit = 32.0 * f64::from(n as f32) * f64::from(f32::EPSILON);
    assert!(column_err <= limit, "column error {column_err:.3e} exceeds {limit:.3e}");
    assert!(trail_err <= limit, "trailing error {trail_err:.3e} exceeds {limit:.3e}");
}
