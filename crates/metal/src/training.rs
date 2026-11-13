//! Contract v2 on Metal: transfers into existing memory, `sync`, the five training ops, and the
//! non-waiting launch path. GPU results are checked against a test-local `f64` reference within
//! 1e-5 absolute, and, where the kernel keeps the CPU reference's order, bit for bit against it.

use linalg_core::{transpose_matrix, Backend, DType, Error, Tensor};

use crate::{MetalBackend, MetalStore};

type T = Tensor<MetalStore>;

const TOL: f64 = 1e-5;

fn backend() -> MetalBackend {
    MetalBackend::new().expect("a Metal device")
}

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn floats(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn halves(raw: &[u8]) -> Vec<u16> {
    raw.chunks_exact(2).map(|c| u16::from_le_bytes(c.try_into().unwrap())).collect()
}

fn up(be: &MetalBackend, shape: &[usize], values: &[f32]) -> T {
    be.upload(DType::F32, shape, &bytes(values)).unwrap()
}

fn down(be: &MetalBackend, t: &T) -> Vec<f32> {
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

#[test]
fn every_v2_method_is_implemented_on_metal() {
    let be = backend();
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
            "add-bias-metal",
            "bias-grad-metal",
            "mse-grad-metal",
            "sgd-step-metal",
            "convert-f32-bf16-metal",
            "convert-bf16-f32-metal",
        ]
    );
}

#[test]
fn add_bias_bias_grad_and_sgd_match_f64_and_the_cpu_order_bit_for_bit() {
    let be = backend();
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

    // In place: y == x.
    let mut in_place = alias(&x);
    be.add_bias(&x, &b, &mut in_place).unwrap();
    assert_bits(&down(&be, &x), &bits, "add_bias in place");

    // bias_grad: rows summed in order, then db + s, as the CPU reference does.
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

    // sgd_step: in every odd element w = fl(lr * dw), so `w - (lr * dw)` is exactly 0 while a
    // fused multiply-add would leave the product's rounding error.
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
    let be = backend();
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

    // Fixed order: the same inputs give the same loss bits, in place too.
    let mut in_place = alias(&y);
    let mut again = 0.0f32;
    be.mse_grad(&y, &t, &mut in_place, &mut again).unwrap();
    assert_eq!(again.to_bits(), loss.to_bits());
    assert_bits(&down(&be, &y), &bits, "mse dy in place");

    let mut zeros = be.alloc(DType::F32, &[rows, cols]).unwrap();
    let mut zero_loss = 1.0f32;
    be.mse_grad(&t, &t, &mut zeros, &mut zero_loss).unwrap();
    assert_eq!(zero_loss, 0.0, "y and target may share storage");
}

#[test]
fn convert_matches_the_reference_rule_bit_for_bit() {
    let be = backend();
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
    let raw: Vec<u8> = all.iter().flat_map(|v| v.to_le_bytes()).collect();
    let hb = be.upload(DType::BF16, &[all.len()], &raw).unwrap();
    let mut back = be.alloc(DType::F32, &[all.len()]).unwrap();
    be.convert(&hb, &mut back).unwrap();
    for (h, f) in all.iter().zip(down(&be, &back)) {
        assert_eq!(f.to_bits(), u32::from(*h) << 16);
    }
}

#[test]
fn errors_come_in_contract_order_and_change_nothing() {
    let be = backend();
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

/// The compute ops enqueue without waiting; `sync`, `download`, and `download_into` wait once;
/// each transfer is one host copy.
#[test]
fn compute_ops_do_not_wait_and_reads_wait_once() {
    let be = backend();
    let (rows, width) = (64, 256);
    let x = up(&be, &[rows, width], &pattern(rows * width, 1));
    let w = up(&be, &[width, width], &pattern(width * width, 2));
    let b = up(&be, &[width], &pattern(width, 3));
    let mut pre = be.alloc(DType::F32, &[rows, width]).unwrap();
    let mut act = be.alloc(DType::F32, &[rows, width]).unwrap();
    let mut dx = be.alloc(DType::F32, &[rows, width]).unwrap();
    let mut db = be.alloc(DType::F32, &[width]).unwrap();
    let mut w2 = up(&be, &[width], &pattern(width, 4));
    let mut h = be.alloc(DType::BF16, &[rows, width]).unwrap();
    assert_eq!(be.debug_waits(), 0);

    be.gemm(&x, &w, &mut pre, 1.0, 0.0).unwrap();
    let mut pre_view = alias(&pre);
    be.add_bias(&pre, &b, &mut pre_view).unwrap();
    be.gelu(&pre, &mut act).unwrap();
    be.gelu_backward(&pre, &act, &mut dx).unwrap();
    be.bias_grad(&dx, &mut db).unwrap();
    be.sgd_step(&mut w2, &mut db, 1e-3).unwrap();
    be.convert(&act, &mut h).unwrap();
    assert_eq!(be.debug_waits(), 0, "seven ops, no wait");

    be.sync().unwrap();
    assert_eq!(be.debug_waits(), 1, "sync waits once");
    let mut out = vec![0u8; rows * width * 2];
    let copies = be.debug_host_copies();
    be.download_into(&h, &mut out).unwrap();
    assert_eq!((be.debug_waits(), be.debug_host_copies()), (2, copies + 1));
    be.download(&act).unwrap();
    assert_eq!(be.debug_waits(), 3, "download waits once");
    be.upload_into(&out, &mut h).unwrap();
    assert_eq!(be.debug_host_copies(), copies + 2, "upload_into is one host copy");
}

/// The non-waiting legacy entry point gives the waiting entry point's bits, on the tiled plan
/// and on the split-K plan.
#[test]
fn nowait_gemm_is_bitwise_equal_to_the_waiting_legacy_gemm() {
    use linalg::abi::{alloc_shared, MetalContext, TensorView};
    use linalg::ops::contraction::{gemm, gemm_nowait};

    let ctx = MetalContext::new().unwrap();
    for (m, k, n) in [(37, 1500, 29), (256, 512, 384), (64, 64, 64)] {
        let av = pattern(m * k, 1);
        let bv = pattern(k * n, 2);
        let la = alloc_shared::<f32>(&ctx, m * k);
        let lb = alloc_shared::<f32>(&ctx, k * n);
        let wait_c = alloc_shared::<f32>(&ctx, m * n);
        let async_c = alloc_shared::<f32>(&ctx, m * n);
        // SAFETY: fresh shared buffers of the right lengths; no GPU work in flight.
        unsafe {
            std::ptr::copy_nonoverlapping(av.as_ptr(), la.contents().cast::<f32>(), m * k);
            std::ptr::copy_nonoverlapping(bv.as_ptr(), lb.contents().cast::<f32>(), k * n);
        }
        let a = TensorView::<f32>::row_major(&la, 0, vec![m, k]);
        let b = TensorView::<f32>::row_major(&lb, 0, vec![k, n]);
        let cw = TensorView::<f32>::row_major(&wait_c, 0, vec![m, n]);
        let ca = TensorView::<f32>::row_major(&async_c, 0, vec![m, n]);
        gemm(&ctx, 0.5, &a, &b, 0.0, &cw).unwrap();
        let cmd = gemm_nowait(&ctx, 0.5, &a, &b, 0.0, &ca).unwrap();
        cmd.wait_until_completed();
        // SAFETY: both command buffers have completed.
        let (x, y) = unsafe {
            (
                std::slice::from_raw_parts(wait_c.contents().cast::<f32>(), m * n),
                std::slice::from_raw_parts(async_c.contents().cast::<f32>(), m * n),
            )
        };
        assert!(x.iter().any(|v| *v != 0.0));
        assert_bits(y, x, &format!("{m}x{k}x{n}"));
    }
}

#[cfg(not(debug_assertions))]
/// One forward layer of W: `pre = x @ W`, `pre += b`, `act = gelu(pre)`.
fn forward(
    be: &MetalBackend,
    x: &T,
    layers: &[(T, T)],
    pre: &mut [T],
    act: &mut [T],
    wait_each: bool,
) {
    for (index, (w, b)) in layers.iter().enumerate() {
        let input = if index == 0 { x } else { &act[index - 1] };
        let input = alias(input);
        be.gemm(&input, w, &mut pre[index], 1.0, 0.0).unwrap();
        if wait_each {
            be.sync().unwrap();
        }
        let mut in_place = alias(&pre[index]);
        be.add_bias(&pre[index], b, &mut in_place).unwrap();
        if wait_each {
            be.sync().unwrap();
        }
        be.gelu(&pre[index], &mut act[index]).unwrap();
        if wait_each {
            be.sync().unwrap();
        }
    }
    be.sync().unwrap();
}

#[cfg(not(debug_assertions))]
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// W's forward, 12 layers at batch 256 and width 4096: enqueueing every op and waiting once
/// takes at most 0.85x the time of waiting after each op, with the same bits.
#[cfg(not(debug_assertions))]
#[test]
fn twelve_layer_forward_without_waits_is_at_most_0_85x_and_bitwise_equal() {
    use std::time::Instant;

    let (batch, width, layers) = (256, 4096, 12);
    let be = backend();
    let scale = 1.0 / (width as f32).sqrt();
    let x = up(&be, &[batch, width], &pattern(batch * width, 1));
    let params: Vec<(T, T)> = (0..layers)
        .map(|l| {
            let w: Vec<f32> = pattern(width * width, 10 + l).iter().map(|v| v * scale).collect();
            (up(&be, &[width, width], &w), up(&be, &[width], &pattern(width, 40 + l)))
        })
        .collect();
    let fresh = |be: &MetalBackend| -> Vec<T> {
        (0..layers).map(|_| be.alloc(DType::F32, &[batch, width]).unwrap()).collect()
    };
    let (mut pre_a, mut act_a) = (fresh(&be), fresh(&be));
    let (mut pre_w, mut act_w) = (fresh(&be), fresh(&be));

    let (mut async_times, mut wait_times) = (Vec::new(), Vec::new());
    for run in 0..7 {
        let t = Instant::now();
        forward(&be, &x, &params, &mut pre_a, &mut act_a, false);
        let a = t.elapsed().as_secs_f64();
        let t = Instant::now();
        forward(&be, &x, &params, &mut pre_w, &mut act_w, true);
        let w = t.elapsed().as_secs_f64();
        if run >= 2 {
            async_times.push(a);
            wait_times.push(w);
        }
    }
    for l in 0..layers {
        assert_bits(&down(&be, &act_a[l]), &down(&be, &act_w[l]), &format!("layer {l}"));
    }
    let (a, w) = (median(async_times), median(wait_times));
    eprintln!(
        "[bench] 12-layer forward 256x4096: enqueue-all {:.3} ms, wait-each {:.3} ms, ratio {:.3}",
        a * 1e3,
        w * 1e3,
        a / w
    );
    assert!(a <= 0.85 * w, "ratio {:.3} exceeds 0.85", a / w);
}

/// The boundary's device work: `convert` of a 256x4096 tensor plus `download_into` of the
/// result, and `upload_into` plus `convert`, each within 2 ms (median of 20).
#[cfg(not(debug_assertions))]
#[test]
fn boundary_convert_and_transfer_take_at_most_2_ms() {
    use std::time::Instant;

    let be = backend();
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
        "[bench] boundary 256x4096 metal: convert+download_into {:.3} ms, upload_into+convert {:.3} ms",
        s * 1e3,
        r * 1e3
    );
    assert!(s <= 2e-3, "convert + download_into {:.3} ms", s * 1e3);
    assert!(r <= 2e-3, "upload_into + convert {:.3} ms", r * 1e3);
}
