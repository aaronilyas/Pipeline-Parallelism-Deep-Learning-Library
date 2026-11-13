//! Contract v2 on CUDA: transfers through pinned staging, `sync`, the five training ops, the
//! free list, and f32-only cuBLAS math. GPU results are checked against a test-local `f64`
//! reference within 1e-5 absolute, and, where the kernel keeps the CPU reference's order, bit
//! for bit against it.

use cudarc::cublas::sys::cublasMath_t;
use cudarc::driver::sys as driver_sys;
use linalg_core::{transpose_matrix, Backend, DType, Error, Identified, Tensor};

use crate::invariants::require_device;
use crate::{CudaBackend, CudaStore};

type T = Tensor<CudaStore>;

const TOL: f64 = 1e-5;

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn floats(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4).map(|c| f32::from_ne_bytes(c.try_into().unwrap())).collect()
}

fn halves(raw: &[u8]) -> Vec<u16> {
    raw.chunks_exact(2).map(|c| u16::from_ne_bytes(c.try_into().unwrap())).collect()
}

fn up(be: &CudaBackend, shape: &[usize], values: &[f32]) -> T {
    be.upload(DType::F32, shape, &bytes(values)).unwrap()
}

fn down(be: &CudaBackend, t: &T) -> Vec<f32> {
    floats(&be.download(t).unwrap())
}

fn pattern(count: usize, seed: usize) -> Vec<f32> {
    (0..count).map(|i| ((i * 17 + seed * 31 + 3) % 101) as f32 / 101.0 - 0.5).collect()
}

fn alias(t: &T) -> T {
    Tensor {
        storage: t.storage.clone(),
        dtype: t.dtype,
        shape: t.shape.clone(),
        strides: t.strides.clone(),
        offset: t.offset,
    }
}

fn assert_close(got: &[f32], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((f64::from(*g) - w).abs() <= tol, "{what}[{i}]: got {g}, want {w}");
    }
}

fn assert_bits(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "{what}[{i}]: got {g}, want {w}");
    }
}

/// The CPU reference's `F32` to `BF16` rule, written out independently.
fn bf16_ref(bits: u32) -> u16 {
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let low = (bits >> 16) as u16;
    let rest = bits & 0xffff;
    if rest > 0x8000 || (rest == 0x8000 && low & 1 == 1) {
        low + 1
    } else {
        low
    }
}

/// `CUDA_SUCCESS` when every call queued on the backend's (null) stream has finished,
/// `CUDA_ERROR_NOT_READY` while some still run.
fn stream_query() -> driver_sys::CUresult {
    // SAFETY: the backend under test loaded the driver; the null stream always exists.
    unsafe { (driver_sys::culib().cuStreamQuery)(std::ptr::null_mut()) }
}

#[test]
fn every_v2_method_is_implemented_on_cuda() {
    let be = require_device();
    let mut x = be.alloc(DType::F32, &[2, 3]).unwrap();
    let b = up(&be, &[3], &[0.5, -0.5, 1.0]);
    let mut y = be.alloc(DType::F32, &[2, 3]).unwrap();
    let mut db = be.alloc(DType::F32, &[3]).unwrap();
    let t = up(&be, &[2, 3], &[0.0; 6]);
    let mut dy = be.alloc(DType::F32, &[2, 3]).unwrap();
    let mut w = up(&be, &[3], &[1.0, 1.0, 1.0]);
    let mut h = be.alloc(DType::BF16, &[2, 3]).unwrap();
    let mut out = vec![0u8; 24];
    let mut loss = -1.0;

    let results = [
        ("sync", be.sync()),
        ("upload_into", be.upload_into(&bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), &mut x)),
        ("add_bias", be.add_bias(&x, &b, &mut y)),
        ("bias_grad", be.bias_grad(&y, &mut db)),
        ("mse_grad", be.mse_grad(&y, &t, &mut dy, &mut loss)),
        ("sgd_step", be.sgd_step(&mut w, &mut db, 0.5)),
        ("convert f32->bf16", be.convert(&y, &mut h)),
        ("convert bf16->f32", be.convert(&h, &mut x)),
        ("download_into", be.download_into(&x, &mut out)),
        ("sync", be.sync()),
    ];
    for (name, result) in results {
        assert_eq!(result, Ok(()), "{name}");
    }
    assert_eq!(floats(&out), vec![1.5, 1.5, 4.0, 4.5, 4.5, 7.0]);
    assert_eq!(down(&be, &db), vec![0.0; 3], "sgd_step zeroes dw");
    assert_eq!(down(&be, &w), vec![-2.0, -2.0, -4.5]);
    let want = [1.5f64, 1.5, 4.0, 4.5, 4.5, 7.0].iter().map(|v| v * v).sum::<f64>() / 6.0;
    assert!((f64::from(loss) - want).abs() <= TOL, "loss {loss} vs {want}");
    assert_eq!(
        be.debug_launches(),
        vec![
            "add-bias-cuda",
            "bias-grad-cuda",
            "mse-grad-cuda",
            "sgd-step-cuda",
            "convert-f32-bf16-cuda",
            "convert-bf16-f32-cuda",
        ]
    );
}

#[test]
fn add_bias_bias_grad_and_sgd_match_f64_and_the_cpu_order_bit_for_bit() {
    let be = require_device();
    let (rows, cols) = (256, 4096);
    let xs = pattern(rows * cols, 1);
    let bs = pattern(cols, 2);
    let x = up(&be, &[rows, cols], &xs);
    let b = up(&be, &[cols], &bs);
    let mut y = be.alloc(DType::F32, &[rows, cols]).unwrap();
    be.add_bias(&x, &b, &mut y).unwrap();
    let got = down(&be, &y);
    let want: Vec<f64> =
        (0..rows * cols).map(|i| f64::from(xs[i]) + f64::from(bs[i % cols])).collect();
    assert_close(&got, &want, TOL, "add_bias");
    let bits: Vec<f32> = (0..rows * cols).map(|i| xs[i] + bs[i % cols]).collect();
    assert_bits(&got, &bits, "add_bias");
    let mut in_place = alias(&x);
    be.add_bias(&x, &b, &mut in_place).unwrap();
    assert_bits(&down(&be, &x), &bits, "add_bias in place");

    let (grows, gcols) = (64, 4096);
    let ds = pattern(grows * gcols, 3);
    let seed = pattern(gcols, 4);
    let dpre = up(&be, &[grows, gcols], &ds);
    let mut db = up(&be, &[gcols], &seed);
    be.bias_grad(&dpre, &mut db).unwrap();
    let got = down(&be, &db);
    let want: Vec<f64> = (0..gcols)
        .map(|j| f64::from(seed[j]) + (0..grows).map(|i| f64::from(ds[i * gcols + j])).sum::<f64>())
        .collect();
    assert_close(&got, &want, TOL, "bias_grad");
    let bits: Vec<f32> = (0..gcols)
        .map(|j| {
            let mut s = 0.0f32;
            for i in 0..grows {
                s += ds[i * gcols + j];
            }
            seed[j] + s
        })
        .collect();
    assert_bits(&got, &bits, "bias_grad");
    let empty = up(&be, &[0, gcols], &[]);
    be.bias_grad(&empty, &mut db).unwrap();
    assert_bits(&down(&be, &db), &bits, "bias_grad with B = 0");

    // In every odd element w = fl(lr * dw), so `w - (lr * dw)` is exactly 0 while a fused
    // multiply-add would leave the product's rounding error.
    let lr = 0.1f32;
    let gs: Vec<f32> = pattern(4096 * 64, 6).iter().map(|g| g * 3.3).collect();
    let ws: Vec<f32> = pattern(4096 * 64, 5)
        .iter()
        .zip(&gs)
        .enumerate()
        .map(|(i, (w, g))| if i % 2 == 1 { lr * g } else { *w })
        .collect();
    let mut w = up(&be, &[64, 4096], &ws);
    let mut dw = up(&be, &[64, 4096], &gs);
    be.sgd_step(&mut w, &mut dw, lr).unwrap();
    let got = down(&be, &w);
    let want: Vec<f64> =
        ws.iter().zip(&gs).map(|(w, g)| f64::from(*w) - f64::from(lr) * f64::from(*g)).collect();
    assert_close(&got, &want, TOL, "sgd_step");
    let bits: Vec<f32> = ws.iter().zip(&gs).map(|(w, g)| w - lr * g).collect();
    let fused = ws.iter().zip(&gs).filter(|(w, g)| (-lr).mul_add(**g, **w) != **w - lr * **g);
    assert!(fused.count() > 0, "the fixture tells fused from unfused");
    assert_bits(&got, &bits, "sgd_step");
    assert!(down(&be, &dw).iter().all(|v| v.to_bits() == 0), "dw is +0.0");
}

#[test]
fn mse_grad_matches_f64_at_the_boundary_shape() {
    let be = require_device();
    let (rows, cols) = (256, 4096);
    let n = rows * cols;
    let ys = pattern(n, 7);
    let ts = pattern(n, 8);
    let y = up(&be, &[rows, cols], &ys);
    let t = up(&be, &[rows, cols], &ts);
    let mut dy = be.alloc(DType::F32, &[rows, cols]).unwrap();
    let mut loss = 0.0f32;
    be.mse_grad(&y, &t, &mut dy, &mut loss).unwrap();
    let d: Vec<f64> = ys.iter().zip(&ts).map(|(a, b)| f64::from(*a) - f64::from(*b)).collect();
    let want_dy: Vec<f64> = d.iter().map(|v| 2.0 * v / n as f64).collect();
    let want_loss = d.iter().map(|v| v * v).sum::<f64>() / n as f64;
    let got = down(&be, &dy);
    assert_close(&got, &want_dy, TOL, "mse dy");
    assert!((f64::from(loss) - want_loss).abs() <= TOL, "loss {loss} vs {want_loss}");
    let bits: Vec<f32> = ys.iter().zip(&ts).map(|(a, b)| (2.0 * (a - b)) / n as f32).collect();
    assert_bits(&got, &bits, "mse dy");

    let mut in_place = alias(&y);
    let mut again = 0.0f32;
    be.mse_grad(&y, &t, &mut in_place, &mut again).unwrap();
    assert_eq!(again.to_bits(), loss.to_bits(), "fixed summation order");
    assert_bits(&down(&be, &y), &bits, "mse dy in place");

    let mut zeros = be.alloc(DType::F32, &[rows, cols]).unwrap();
    let mut zero_loss = 1.0f32;
    be.mse_grad(&t, &t, &mut zeros, &mut zero_loss).unwrap();
    assert_eq!(zero_loss, 0.0, "y and target may share storage");
}

#[test]
fn convert_matches_the_reference_rule_bit_for_bit() {
    let be = require_device();
    let mut cases: Vec<u32> = vec![
        0,
        0x8000_0000,
        0x3f80_8000,
        0x3f81_8000,
        0x3f80_8001,
        0x3f80_7fff,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x7f7f_7fff,
        0x7f80_0000,
        0xff80_0000,
        0x0001_0000,
        0x8001_8000,
        0x0000_0001,
        0x7f80_0001,
        0xff80_0001,
        0x7fc0_0000,
        0xffff_ffff,
        0x7f81_2345,
    ];
    cases.extend(pattern(4096, 9).iter().map(|v| (v * 1e3).to_bits()));
    cases.extend((0..4096u32).map(|i| i.wrapping_mul(2_654_435_761)));
    let values: Vec<f32> = cases.iter().map(|b| f32::from_bits(*b)).collect();
    let x = up(&be, &[values.len()], &values);
    let mut h = be.alloc(DType::BF16, &[values.len()]).unwrap();
    be.convert(&x, &mut h).unwrap();
    let got = halves(&be.download(&h).unwrap());
    for (bits, g) in cases.iter().zip(&got) {
        assert_eq!(*g, bf16_ref(*bits), "{bits:#010x}");
    }
    assert_eq!(got[6], 0x7f80, "f32::MAX overflows to +inf");
    assert_eq!(got[11], 0x0001, "subnormal kept");
    assert_eq!(got[14], 0x7fc0, "signaling NaN becomes quiet");
    assert_eq!(got[15], 0xffc0, "and keeps its sign");

    let all: Vec<u16> = (0..=u16::MAX).collect();
    let raw: Vec<u8> = all.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let hb = be.upload(DType::BF16, &[all.len()], &raw).unwrap();
    let mut back = be.alloc(DType::F32, &[all.len()]).unwrap();
    be.convert(&hb, &mut back).unwrap();
    for (h, f) in all.iter().zip(down(&be, &back)) {
        assert_eq!(f.to_bits(), u32::from(*h) << 16);
    }
}

#[test]
fn errors_come_in_contract_order_and_change_nothing() {
    let be = require_device();
    let x = up(&be, &[2, 3], &pattern(6, 11));
    let b = up(&be, &[3], &pattern(3, 12));
    let b_wrong = up(&be, &[4], &pattern(4, 12));
    let h = be.alloc(DType::BF16, &[2, 3]).unwrap();
    let hb = be.alloc(DType::BF16, &[3]).unwrap();
    let seed = pattern(6, 13);
    let mut y = up(&be, &[2, 3], &seed);
    let mut odd = alias(&y);
    odd.strides = vec![1, 2];
    let mut x_odd = alias(&x);
    x_odd.strides = vec![1, 2];
    let mut loss = 42.0f32;

    assert_eq!(be.add_bias(&h, &b_wrong, &mut y), Err(Error::Shape));
    assert_eq!(be.add_bias(&h, &b, &mut y), Err(Error::Dtype));
    let bias_on_y = Tensor { shape: vec![3], strides: vec![1], ..alias(&y) };
    assert_eq!(be.add_bias(&x, &bias_on_y, &mut y), Err(Error::Alias));
    assert_eq!(be.add_bias(&x, &b, &mut odd), Err(Error::UnsupportedLayout));

    let mut db = up(&be, &[3], &[1.0, 2.0, 3.0]);
    assert_eq!(be.bias_grad(&h, &mut alias(&b_wrong)), Err(Error::Shape));
    assert_eq!(be.bias_grad(&x, &mut alias(&hb)), Err(Error::Dtype));
    let mut db_on_x = Tensor { shape: vec![3], strides: vec![1], ..alias(&x) };
    assert_eq!(be.bias_grad(&x, &mut db_on_x), Err(Error::Alias));
    assert_eq!(be.bias_grad(&x_odd, &mut db), Err(Error::UnsupportedLayout));

    let empty = be.alloc(DType::F32, &[0, 3]).unwrap();
    assert_eq!(be.mse_grad(&empty, &empty, &mut alias(&empty), &mut loss), Err(Error::Shape));
    assert_eq!(be.mse_grad(&x, &b, &mut y, &mut loss), Err(Error::Shape));
    assert_eq!(be.mse_grad(&h, &x, &mut y, &mut loss), Err(Error::Dtype));
    assert_eq!(be.mse_grad(&x, &y, &mut alias(&y), &mut loss), Err(Error::Alias));
    assert_eq!(be.mse_grad(&x, &x, &mut odd, &mut loss), Err(Error::UnsupportedLayout));
    assert_eq!(loss, 42.0, "loss unchanged on error");

    let mut w = up(&be, &[3], &[1.0, 2.0, 3.0]);
    let mut w_alias = alias(&w);
    assert_eq!(be.sgd_step(&mut w, &mut alias(&x), 0.1), Err(Error::Shape));
    assert_eq!(be.sgd_step(&mut w, &mut alias(&hb), 0.1), Err(Error::Dtype));
    assert_eq!(be.sgd_step(&mut w, &mut w_alias, 0.1), Err(Error::Alias));
    let mut w_shift = alias(&w);
    w_shift.offset = 1;
    assert_eq!(be.sgd_step(&mut w_shift, &mut db, 0.1), Err(Error::UnsupportedLayout));

    let mut h_out = be.alloc(DType::BF16, &[2, 3]).unwrap();
    assert_eq!(be.convert(&x, &mut alias(&hb)), Err(Error::Shape));
    assert_eq!(be.convert(&x, &mut y), Err(Error::Dtype));
    assert_eq!(be.convert(&h, &mut alias(&h)), Err(Error::Dtype));
    let mut h_on_x = Tensor { dtype: DType::BF16, ..alias(&x) };
    assert_eq!(be.convert(&x, &mut h_on_x), Err(Error::Alias));
    assert_eq!(be.convert(&x_odd, &mut h_out), Err(Error::UnsupportedLayout));

    let mut t = be.alloc(DType::F32, &[3, 4]).unwrap();
    let mut short = vec![0xAAu8; 47];
    assert_eq!(be.download_into(&t, &mut short), Err(Error::Shape));
    assert!(short.iter().all(|b| *b == 0xAA));
    assert_eq!(be.upload_into(&[1u8; 49], &mut t), Err(Error::Shape));
    let mut tt = transpose_matrix(&t).unwrap();
    assert_eq!(be.upload_into(&[1u8; 48], &mut tt), Err(Error::UnsupportedLayout));
    assert_eq!(be.download_into(&tt, &mut [0u8; 48]), Err(Error::UnsupportedLayout));
    assert_eq!(down(&be, &t), vec![0.0; 12]);

    assert_eq!(down(&be, &y), seed);
    assert_eq!(down(&be, &db), vec![1.0, 2.0, 3.0]);
    assert_eq!(down(&be, &w), vec![1.0, 2.0, 3.0]);
    assert!(be.debug_launches().is_empty(), "{:?}", be.debug_launches());
}

/// Every op but the reads returns while the stream still runs: nothing synchronizes before
/// `sync` or a download. Each transfer is one device-host copy through staging that is
/// allocated once.
#[test]
fn ops_are_stream_ordered_and_transfers_are_one_copy() {
    let be = require_device();
    let width = 4096;
    let a = up(&be, &[width, width], &pattern(width * width, 1));
    let bm = up(&be, &[width, width], &pattern(width * width, 2));
    let mut c = be.alloc(DType::F32, &[width, width]).unwrap();
    let bias = up(&be, &[width], &pattern(width, 3));
    let act = be.alloc(DType::F32, &[256, width]).unwrap();
    let mut h = be.alloc(DType::BF16, &[256, width]).unwrap();
    let mut db = be.alloc(DType::F32, &[width]).unwrap();
    let mut w = up(&be, &[width], &pattern(width, 4));
    let host = vec![0u8; 256 * width * 2];
    be.sync().unwrap();

    // A 4096^3 gemm keeps the stream busy for milliseconds on either GPU.
    be.gemm(&a, &bm, &mut c, 1.0, 0.0).unwrap();
    let mut act_view = alias(&act);
    be.add_bias(&act, &bias, &mut act_view).unwrap();
    let mut gelu_out = be.alloc(DType::F32, &[256, width]).unwrap();
    be.gelu(&act, &mut gelu_out).unwrap();
    be.convert(&act, &mut h).unwrap();
    be.bias_grad(&act, &mut db).unwrap();
    be.sgd_step(&mut w, &mut db, 1e-3).unwrap();
    be.upload_into(&host, &mut h).unwrap();
    assert_eq!(
        stream_query(),
        driver_sys::CUresult::CUDA_ERROR_NOT_READY,
        "an op synchronized the stream"
    );
    be.sync().unwrap();
    assert_eq!(stream_query(), driver_sys::CUresult::CUDA_SUCCESS);

    let copies = be.debug_device_host_copies();
    let pinned = be.debug_pinned_allocs();
    let mut out = vec![0u8; host.len()];
    for round in 0..10 {
        be.convert(&act, &mut h).unwrap();
        be.download_into(&h, &mut out).unwrap();
        be.upload_into(&out, &mut h).unwrap();
        assert_eq!(be.debug_device_host_copies(), copies + 2 * (round + 1), "one copy each");
    }
    assert!(be.debug_pinned_allocs() - pinned <= 2, "staging is allocated once per direction");
    let after = be.debug_pinned_allocs();
    for _ in 0..5 {
        be.download_into(&h, &mut out).unwrap();
        be.upload_into(&out, &mut h).unwrap();
    }
    assert_eq!(be.debug_pinned_allocs(), after, "steady state allocates no staging");
}

#[test]
fn cublas_math_stays_f32_without_tf32() {
    let be = require_device();
    assert_eq!(be.math_mode_for_test(), cublasMath_t::CUBLAS_DEFAULT_MATH);
}

#[test]
fn dropped_store_returns_its_buffer_to_the_free_list() {
    let be = require_device();
    let first = be.alloc(DType::F32, &[123, 457]).unwrap();
    let id = first.storage.storage_id();
    let before = be.debug_mem_allocs();
    drop(first);
    let second = be.alloc(DType::F32, &[457, 123]).unwrap();
    assert_eq!(second.storage.storage_id(), id, "same size, same buffer");
    assert_eq!(be.debug_mem_allocs(), before, "no allocator call");
    assert!(down(&be, &second).iter().all(|v| *v == 0.0), "alloc still zero-fills");
    let mut third = be.upload(DType::F32, &[7], &bytes(&[1.0; 7])).unwrap();
    let id3 = third.storage.storage_id();
    drop(std::mem::replace(&mut third, be.alloc(DType::F32, &[1]).unwrap()));
    let fourth = be.upload(DType::F32, &[7], &bytes(&[2.0; 7])).unwrap();
    assert_eq!(fourth.storage.storage_id(), id3, "upload reuses too");
    assert_eq!(down(&be, &fourth), vec![2.0; 7]);
}

/// One training step of a small W-shaped MLP, every temporary allocated inside it and dropped
/// at its end.
fn train_step(be: &CudaBackend, params: &mut [(T, T)], x: &T, target: &T) -> f32 {
    let batch = x.shape[0];
    let mut pres = Vec::new();
    let mut acts: Vec<T> = Vec::new();
    for (w, b) in params.iter() {
        let width = w.shape[1];
        let input = acts.last().map(alias).unwrap_or_else(|| alias(x));
        let mut pre = be.alloc(DType::F32, &[batch, width]).unwrap();
        be.gemm(&input, w, &mut pre, 1.0, 0.0).unwrap();
        let mut pre_view = alias(&pre);
        be.add_bias(&pre, b, &mut pre_view).unwrap();
        let mut act = be.alloc(DType::F32, &[batch, width]).unwrap();
        be.gelu(&pre, &mut act).unwrap();
        pres.push(pre);
        acts.push(act);
    }
    let mut dy = be.alloc(DType::F32, &acts.last().unwrap().shape).unwrap();
    let mut loss = 0.0;
    be.mse_grad(acts.last().unwrap(), target, &mut dy, &mut loss).unwrap();
    for layer in (0..params.len()).rev() {
        let (w, b) = &mut params[layer];
        let mut dpre = be.alloc(DType::F32, &pres[layer].shape).unwrap();
        be.gelu_backward(&pres[layer], &dy, &mut dpre).unwrap();
        let input = if layer == 0 { alias(x) } else { alias(&acts[layer - 1]) };
        let mut dw = be.alloc(DType::F32, &w.shape).unwrap();
        be.gemm(&transpose_matrix(&input).unwrap(), &dpre, &mut dw, 1.0, 0.0).unwrap();
        let mut db = be.alloc(DType::F32, &b.shape).unwrap();
        be.bias_grad(&dpre, &mut db).unwrap();
        let mut dx = be.alloc(DType::F32, &input.shape).unwrap();
        be.gemm(&dpre, &transpose_matrix(w).unwrap(), &mut dx, 1.0, 0.0).unwrap();
        be.sgd_step(w, &mut dw, 1e-3).unwrap();
        be.sgd_step(b, &mut db, 1e-3).unwrap();
        dy = dx;
    }
    loss
}

#[test]
fn steady_state_steps_make_no_allocator_calls() {
    let be = require_device();
    let (batch, width, layers) = (64, 512, 3);
    let mut params: Vec<(T, T)> = (0..layers)
        .map(|l| {
            let w: Vec<f32> = pattern(width * width, l).iter().map(|v| v * 0.05).collect();
            (up(&be, &[width, width], &w), up(&be, &[width], &pattern(width, 10 + l)))
        })
        .collect();
    let x = up(&be, &[batch, width], &pattern(batch * width, 20));
    let target = up(&be, &[batch, width], &pattern(batch * width, 21));
    let first = train_step(&be, &mut params, &x, &target);
    let after_first = be.debug_mem_allocs();
    let mut last = first;
    for _ in 0..3 {
        last = train_step(&be, &mut params, &x, &target);
    }
    assert_eq!(be.debug_mem_allocs(), after_first, "steps 2..4 called the allocator");
    assert!(last < first, "training lowers the loss: {first} -> {last}");
}

#[cfg(not(debug_assertions))]
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// The boundary's device work: `convert` of a 256x4096 tensor plus `download_into` of the
/// result, and `upload_into` plus `convert`, each within 2 ms (median of 20).
#[cfg(not(debug_assertions))]
#[test]
fn boundary_convert_and_transfer_take_at_most_2_ms() {
    use std::time::Instant;

    let be = require_device();
    let (rows, cols) = (256, 4096);
    let act = up(&be, &[rows, cols], &pattern(rows * cols, 1));
    let mut wire = be.alloc(DType::BF16, &[rows, cols]).unwrap();
    let mut back = be.alloc(DType::F32, &[rows, cols]).unwrap();
    let mut host = vec![0u8; rows * cols * 2];
    be.sync().unwrap();
    let (mut send, mut recv) = (Vec::new(), Vec::new());
    for _ in 0..25 {
        let t = Instant::now();
        be.convert(&act, &mut wire).unwrap();
        be.download_into(&wire, &mut host).unwrap();
        send.push(t.elapsed().as_secs_f64());
        let t = Instant::now();
        be.upload_into(&host, &mut wire).unwrap();
        be.convert(&wire, &mut back).unwrap();
        be.sync().unwrap();
        recv.push(t.elapsed().as_secs_f64());
    }
    let (s, r) = (median(send[5..].to_vec()), median(recv[5..].to_vec()));
    eprintln!(
        "[bench] boundary 256x4096 cuda: convert+download_into {:.3} ms, upload_into+convert {:.3} ms",
        s * 1e3,
        r * 1e3
    );
    assert!(s <= 2e-3, "convert + download_into {:.3} ms", s * 1e3);
    assert!(r <= 2e-3, "upload_into + convert {:.3} ms", r * 1e3);
}
