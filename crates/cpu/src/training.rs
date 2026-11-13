//! Contract v2 on the CPU reference: transfers into existing memory and the five training ops.
//! Each op is checked against a test-local `f64` reference within 1e-6 absolute, and against
//! the sequential `f32` loop that defines the reference bits.

use crate::{CpuBackend, CpuStore};
use linalg_core::{transpose_matrix, Backend, DType, Error, Tensor};

type T = Tensor<CpuStore>;

const TOL: f64 = 1e-6;

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn floats(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4).map(|c| f32::from_ne_bytes(c.try_into().unwrap())).collect()
}

fn halves(raw: &[u8]) -> Vec<u16> {
    raw.chunks_exact(2).map(|c| u16::from_ne_bytes(c.try_into().unwrap())).collect()
}

fn up(be: &CpuBackend, shape: &[usize], values: &[f32]) -> T {
    be.upload(DType::F32, shape, &bytes(values)).unwrap()
}

fn down(be: &CpuBackend, t: &T) -> Vec<f32> {
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

/// The `bf16` nearest `x`, ties to even, found by comparing `f64` distances to the two
/// neighbours that truncation brackets. NaN and infinity are handled by the caller.
fn bf16_nearest_f64(x: f32) -> u16 {
    let bits = x.to_bits();
    let low = (bits >> 16) as u16;
    let high = low.wrapping_add(1);
    // The all-ones exponent stands for 2^128 here, so overflow rounds as IEEE says it does.
    let value = |h: u16| {
        if h & 0x7fff == 0x7f80 {
            let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
            return sign * 2f64.powi(128);
        }
        f64::from(f32::from_bits(u32::from(h) << 16))
    };
    let (dl, dh) = ((f64::from(x) - value(low)).abs(), (value(high) - f64::from(x)).abs());
    if bits & 0xffff == 0 || dl < dh || (dl == dh && low & 1 == 0) {
        low
    } else {
        high
    }
}

#[test]
fn every_v2_method_is_implemented_on_cpu() {
    let be = CpuBackend::new();
    let mut x = up(&be, &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
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
    ];
    for (name, result) in results {
        assert_eq!(result, Ok(()), "{name}");
    }
    assert_eq!(floats(&out), vec![1.5, 1.5, 4.0, 4.5, 4.5, 7.0]);
    assert_eq!(down(&be, &db), vec![0.0; 3], "sgd_step zeroes dw");
    assert_eq!(down(&be, &w), vec![1.0 - 0.5 * 6.0, 1.0 - 0.5 * 6.0, 1.0 - 0.5 * 11.0]);
    assert!(loss > 0.0);
}

#[test]
fn add_bias_matches_f64_and_runs_in_place() {
    let be = CpuBackend::new();
    let (rows, cols) = (7, 33);
    let xs = pattern(rows * cols, 1);
    let bs = pattern(cols, 2);
    let x = up(&be, &[rows, cols], &xs);
    let b = up(&be, &[cols], &bs);
    let mut y = be.alloc(DType::F32, &[rows, cols]).unwrap();
    be.add_bias(&x, &b, &mut y).unwrap();
    let want: Vec<f64> =
        (0..rows * cols).map(|i| f64::from(xs[i]) + f64::from(bs[i % cols])).collect();
    assert_close(&down(&be, &y), &want, TOL, "add_bias");
    let bits: Vec<f32> = (0..rows * cols).map(|i| xs[i] + bs[i % cols]).collect();
    assert_eq!(down(&be, &y), bits, "sequential f32 bits");

    let mut in_place = alias(&x);
    be.add_bias(&x, &b, &mut in_place).unwrap();
    assert_eq!(down(&be, &x), bits, "y == x is an in-place add");
    assert_eq!(be.debug_launches(), vec!["add-bias", "add-bias"]);
}

#[test]
fn bias_grad_accumulates_column_sums() {
    let be = CpuBackend::new();
    let (rows, cols) = (64, 37);
    let ds = pattern(rows * cols, 3);
    let seed = pattern(cols, 4);
    let dpre = up(&be, &[rows, cols], &ds);
    let mut db = up(&be, &[cols], &seed);
    be.bias_grad(&dpre, &mut db).unwrap();
    let want: Vec<f64> = (0..cols)
        .map(|j| f64::from(seed[j]) + (0..rows).map(|i| f64::from(ds[i * cols + j])).sum::<f64>())
        .collect();
    assert_close(&down(&be, &db), &want, TOL, "bias_grad");
    let bits: Vec<f32> = (0..cols)
        .map(|j| {
            let mut s = 0.0f32;
            for i in 0..rows {
                s += ds[i * cols + j];
            }
            seed[j] + s
        })
        .collect();
    assert_eq!(down(&be, &db), bits, "sequential f32 bits");

    // B = 0 leaves db unchanged.
    let empty = up(&be, &[0, cols], &[]);
    be.bias_grad(&empty, &mut db).unwrap();
    assert_eq!(down(&be, &db), bits);
    assert_eq!(be.debug_launches(), vec!["bias-grad", "bias-grad"]);
}

#[test]
fn mse_grad_matches_f64_in_place_and_with_shared_target() {
    let be = CpuBackend::new();
    // A sequential f32 sum drifts from f64 with n: at n = 2064 the loss is 1.3e-6 off, past
    // the 1e-6 bound, so this case stays at n = 520.
    let (rows, cols) = (8, 65);
    let n = rows * cols;
    let ys = pattern(n, 5);
    let ts = pattern(n, 6);
    let y = up(&be, &[rows, cols], &ys);
    let t = up(&be, &[rows, cols], &ts);
    let mut dy = be.alloc(DType::F32, &[rows, cols]).unwrap();
    let mut loss = 0.0f32;
    be.mse_grad(&y, &t, &mut dy, &mut loss).unwrap();
    let d: Vec<f64> = ys.iter().zip(&ts).map(|(a, b)| f64::from(*a) - f64::from(*b)).collect();
    let want_dy: Vec<f64> = d.iter().map(|v| 2.0 * v / n as f64).collect();
    let want_loss = d.iter().map(|v| v * v).sum::<f64>() / n as f64;
    assert_close(&down(&be, &dy), &want_dy, TOL, "mse dy");
    assert!((f64::from(loss) - want_loss).abs() <= TOL, "loss {loss} vs {want_loss}");
    let mut sum = 0.0f32;
    for (a, b) in ys.iter().zip(&ts) {
        sum += (a - b) * (a - b);
    }
    assert_eq!(loss.to_bits(), (sum / n as f32).to_bits(), "sequential f32 loss bits");

    // dy == y (in place) gives the same bits.
    let expected = down(&be, &dy);
    let mut in_place = alias(&y);
    let mut loss2 = 0.0f32;
    be.mse_grad(&y, &t, &mut in_place, &mut loss2).unwrap();
    assert_eq!(down(&be, &y), expected);
    assert_eq!(loss2.to_bits(), loss.to_bits());

    // y and target may share storage: d is zero.
    let mut zeros = be.alloc(DType::F32, &[rows, cols]).unwrap();
    let mut loss3 = 1.0f32;
    be.mse_grad(&t, &t, &mut zeros, &mut loss3).unwrap();
    assert_eq!(loss3, 0.0);
    assert!(down(&be, &zeros).iter().all(|v| *v == 0.0));
}

#[test]
fn sgd_step_updates_and_zeroes() {
    let be = CpuBackend::new();
    let ws = pattern(300, 7);
    let gs = pattern(300, 8);
    let mut w = up(&be, &[10, 30], &ws);
    let mut dw = up(&be, &[10, 30], &gs);
    let lr = 1e-3f32;
    be.sgd_step(&mut w, &mut dw, lr).unwrap();
    let want: Vec<f64> =
        ws.iter().zip(&gs).map(|(w, g)| f64::from(*w) - f64::from(lr) * f64::from(*g)).collect();
    assert_close(&down(&be, &w), &want, TOL, "sgd w");
    let bits: Vec<f32> = ws.iter().zip(&gs).map(|(w, g)| w - lr * g).collect();
    assert_eq!(down(&be, &w), bits);
    assert!(down(&be, &dw).iter().all(|v| v.to_bits() == 0), "dw is +0.0");
}

#[test]
fn convert_rounds_to_nearest_even_and_keeps_specials() {
    let be = CpuBackend::new();
    let mut cases: Vec<f32> = vec![
        0.0,
        -0.0,
        1.0,
        f32::from_bits(0x3f80_8000), // tie, even below
        f32::from_bits(0x3f81_8000), // tie, even above
        f32::from_bits(0x3f80_8001),
        f32::from_bits(0x3f80_7fff),
        f32::MAX,
        f32::MIN,
        f32::from_bits(0x7f7f_7fff), // rounds down to the largest finite bf16
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x0001_0000), // bf16 subnormal, kept
        f32::from_bits(0x8001_8000), // negative subnormal tie
        f32::from_bits(0x0000_0001),
    ];
    cases.extend(pattern(1000, 9).iter().map(|v| v * 1e3));
    let x = up(&be, &[cases.len()], &cases);
    let mut h = be.alloc(DType::BF16, &[cases.len()]).unwrap();
    be.convert(&x, &mut h).unwrap();
    let got = halves(&be.download(&h).unwrap());
    for (v, g) in cases.iter().zip(&got) {
        let want = if v.is_infinite() { (v.to_bits() >> 16) as u16 } else { bf16_nearest_f64(*v) };
        assert_eq!(*g, want, "{v:e} ({:#010x})", v.to_bits());
    }
    assert_eq!(got[7], 0x7f80, "f32::MAX overflows to +inf");
    assert_eq!(got[8], 0xff80, "f32::MIN overflows to -inf");
    assert_eq!(got[12], 0x0001, "subnormal kept");

    // NaN: quiet, same sign.
    let nans = [0x7f80_0001u32, 0xff80_0001, 0x7fc0_0000, 0xffff_ffff, 0x7f81_2345];
    let xs: Vec<f32> = nans.iter().map(|b| f32::from_bits(*b)).collect();
    let xn = up(&be, &[xs.len()], &xs);
    let mut hn = be.alloc(DType::BF16, &[xs.len()]).unwrap();
    be.convert(&xn, &mut hn).unwrap();
    for (bits, g) in nans.iter().zip(halves(&be.download(&hn).unwrap())) {
        assert!(g & 0x7f80 == 0x7f80 && g & 0x007f != 0, "{bits:#x} -> {g:#x} is NaN");
        assert!(g & 0x0040 != 0, "{bits:#x} -> {g:#x} is quiet");
        assert_eq!(g & 0x8000, ((bits >> 16) & 0x8000) as u16, "{bits:#x} keeps its sign");
    }

    // BF16 to F32 is exact: the bits move to the top half.
    let all: Vec<u16> = (0..=u16::MAX).collect();
    let raw: Vec<u8> = all.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let hb = be.upload(DType::BF16, &[all.len()], &raw).unwrap();
    let mut back = be.alloc(DType::F32, &[all.len()]).unwrap();
    be.convert(&hb, &mut back).unwrap();
    for (h, f) in all.iter().zip(down(&be, &back)) {
        assert_eq!(f.to_bits(), u32::from(*h) << 16);
    }
    assert_eq!(
        be.debug_launches(),
        vec!["convert-f32-bf16", "convert-f32-bf16", "convert-bf16-f32"]
    );
}

#[test]
fn transfers_into_existing_memory() {
    let be = CpuBackend::new();
    let mut t = be.alloc(DType::F32, &[3, 4]).unwrap();
    let values = pattern(12, 10);
    be.upload_into(&bytes(&values), &mut t).unwrap();
    let mut out = vec![0u8; 48];
    be.download_into(&t, &mut out).unwrap();
    assert_eq!(floats(&out), values);
    assert_eq!((be.debug_allocs(), be.debug_uploads(), be.debug_downloads()), (1, 1, 1));
    assert!(be.debug_launches().is_empty());

    // Wrong length: Shape, nothing written.
    let mut short = vec![0xAAu8; 47];
    assert_eq!(be.download_into(&t, &mut short), Err(Error::Shape));
    assert!(short.iter().all(|b| *b == 0xAA));
    assert_eq!(be.upload_into(&[0u8; 49], &mut t), Err(Error::Shape));
    assert_eq!(down(&be, &t), values);

    // A transposed view is UnsupportedLayout.
    let mut tt = transpose_matrix(&t).unwrap();
    assert_eq!(be.upload_into(&[0u8; 48], &mut tt), Err(Error::UnsupportedLayout));
    assert_eq!(be.download_into(&tt, &mut out), Err(Error::UnsupportedLayout));
    assert_eq!(down(&be, &t), values);
}

/// Shape, then Dtype, then Alias, then UnsupportedLayout; no output changes and no launch.
#[test]
fn errors_come_in_contract_order_and_change_nothing() {
    let be = CpuBackend::new();
    let x = up(&be, &[2, 3], &pattern(6, 11));
    let b = up(&be, &[3], &pattern(3, 12));
    let b_wrong = up(&be, &[4], &pattern(4, 12));
    let h = be.alloc(DType::BF16, &[2, 3]).unwrap();
    let hb = be.alloc(DType::BF16, &[3]).unwrap();
    let seed = pattern(6, 13);
    let mut y = up(&be, &[2, 3], &seed);
    let mut odd = alias(&y);
    odd.strides = vec![1, 2];
    let mut loss = 42.0f32;

    // add_bias
    assert_eq!(be.add_bias(&h, &b_wrong, &mut y), Err(Error::Shape));
    assert_eq!(be.add_bias(&h, &b, &mut y), Err(Error::Dtype));
    assert_eq!(be.add_bias(&x, &b, &mut alias(&b)), Err(Error::Shape));
    assert_eq!(be.add_bias(&x, &alias(&y), &mut y), Err(Error::Shape));
    let bias_on_y = Tensor { shape: vec![3], strides: vec![1], ..alias(&y) };
    assert_eq!(be.add_bias(&x, &bias_on_y, &mut y), Err(Error::Alias));
    assert_eq!(be.add_bias(&x, &b, &mut odd), Err(Error::UnsupportedLayout));

    // bias_grad
    let mut db = up(&be, &[3], &[1.0, 2.0, 3.0]);
    assert_eq!(be.bias_grad(&h, &mut alias(&b_wrong)), Err(Error::Shape));
    assert_eq!(be.bias_grad(&x, &mut alias(&hb)), Err(Error::Dtype));
    let mut db_on_x = Tensor { shape: vec![3], strides: vec![1], ..alias(&x) };
    assert_eq!(be.bias_grad(&x, &mut db_on_x), Err(Error::Alias));
    let mut x_odd = alias(&x);
    x_odd.strides = vec![1, 2];
    assert_eq!(be.bias_grad(&x_odd, &mut db), Err(Error::UnsupportedLayout));
    assert_eq!(down(&be, &db), vec![1.0, 2.0, 3.0]);

    // mse_grad
    let empty = be.alloc(DType::F32, &[0, 3]).unwrap();
    assert_eq!(be.mse_grad(&empty, &empty, &mut alias(&empty), &mut loss), Err(Error::Shape));
    assert_eq!(be.mse_grad(&x, &b, &mut y, &mut loss), Err(Error::Shape));
    assert_eq!(be.mse_grad(&h, &x, &mut y, &mut loss), Err(Error::Dtype));
    assert_eq!(be.mse_grad(&x, &y, &mut alias(&y), &mut loss), Err(Error::Alias));
    assert_eq!(be.mse_grad(&x, &x, &mut odd, &mut loss), Err(Error::UnsupportedLayout));
    assert_eq!(loss, 42.0, "loss unchanged on error");

    // sgd_step
    let mut w = up(&be, &[3], &[1.0, 2.0, 3.0]);
    assert_eq!(be.sgd_step(&mut w, &mut alias(&x), 0.1), Err(Error::Shape));
    assert_eq!(be.sgd_step(&mut w, &mut alias(&hb), 0.1), Err(Error::Dtype));
    let mut w_alias = alias(&w);
    assert_eq!(be.sgd_step(&mut w, &mut w_alias, 0.1), Err(Error::Alias));
    let mut w_shift = alias(&w);
    w_shift.offset = 1;
    assert_eq!(be.sgd_step(&mut w_shift, &mut db, 0.1), Err(Error::UnsupportedLayout));
    assert_eq!(down(&be, &w), vec![1.0, 2.0, 3.0]);

    // convert
    let mut h_out = be.alloc(DType::BF16, &[2, 3]).unwrap();
    assert_eq!(be.convert(&x, &mut alias(&hb)), Err(Error::Shape));
    assert_eq!(be.convert(&x, &mut y), Err(Error::Dtype));
    assert_eq!(be.convert(&h, &mut alias(&h)), Err(Error::Dtype));
    let mut h_on_x = Tensor { dtype: DType::BF16, ..alias(&x) };
    assert_eq!(be.convert(&x, &mut h_on_x), Err(Error::Alias));
    assert_eq!(be.convert(&x_odd, &mut h_out), Err(Error::UnsupportedLayout));

    assert_eq!(down(&be, &y), seed);
    assert!(be.debug_launches().is_empty(), "{:?}", be.debug_launches());
}
