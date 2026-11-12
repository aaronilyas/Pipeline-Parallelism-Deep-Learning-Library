//! Unit tests. They live in the library so they see the `cfg(test)` counters.

#[cfg(not(target_os = "macos"))]
mod linux {
    use crate::{Error, MetalBackend};

    #[test]
    fn new_without_metal_is_unsupported_device() {
        assert_eq!(MetalBackend::new().err(), Some(Error::UnsupportedDevice));
    }
}

/// Reads the manifest as text, so it runs on every OS.
#[test]
fn cargo_toml_has_no_linalg_cpu_dependency() {
    let manifest = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");
    let text = std::fs::read_to_string(manifest).expect("read manifest");
    assert!(!text.contains("linalg-cpu"), "metal must not depend on linalg-cpu");
    assert!(!text.contains("linalg_cpu"), "metal must not depend on linalg-cpu");
}

#[cfg(target_os = "macos")]
mod macos {

    use linalg_core::{
        row_major_strides, transpose_matrix, Backend, DType, Error, Identified, Tensor,
    };

    use crate::{MetalBackend, MetalStore};

    type T = Tensor<MetalStore>;

    fn backend() -> MetalBackend {
        MetalBackend::new().expect("a Metal device")
    }

    fn bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn floats(raw: &[u8]) -> Vec<f32> {
        raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    fn up(be: &MetalBackend, shape: &[usize], values: &[f32]) -> T {
        be.upload(DType::F32, shape, &bytes(values)).expect("upload")
    }

    fn down(be: &MetalBackend, t: &T) -> Vec<f32> {
        floats(&be.download(t).expect("download"))
    }

    fn pattern(rows: usize, cols: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                out.push(((i * 17 + j * 3) % 100) as f32 / 100.0 - 0.5);
            }
        }
        out
    }

    /// Reference `beta*C + alpha*A*B` in f64 for row-major `a` [m,k], `b` [k,n], `c` [m,n].
    #[allow(clippy::too_many_arguments)]
    fn reference(
        a: &[f32],
        b: &[f32],
        c: &[f32],
        m: usize,
        k: usize,
        n: usize,
        alpha: f32,
        beta: f32,
    ) -> Vec<f64> {
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for p in 0..k {
                    acc += f64::from(a[i * k + p]) * f64::from(b[p * n + j]);
                }
                let old = if beta == 0.0 { 0.0 } else { f64::from(beta) * f64::from(c[i * n + j]) };
                out[i * n + j] = old + f64::from(alpha) * acc;
            }
        }
        out
    }

    fn transposed(rows: usize, cols: usize, row_major: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                out[j * rows + i] = row_major[i * cols + j];
            }
        }
        out
    }

    fn assert_close(got: &[f32], want: &[f64], tol: f64, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((f64::from(*g) - w).abs() <= tol, "{what}[{i}]: got {g}, want {w}");
        }
    }

    #[test]
    fn f32_gemm_matches_f64_reference_within_1e_3_absolute() {
        let (m, k, n) = (32, 32, 32);
        let a = pattern(m, k);
        let b = pattern(k, n);
        let c0 = transposed(m, n, &pattern(m, n));
        for (alpha, beta) in [(1.0f32, 0.0f32), (0.75, -0.5)] {
            for mode in ["plain", "a_transposed", "b_transposed"] {
                let be = backend();
                let a_t = match mode {
                    "a_transposed" => {
                        transpose_matrix(&up(&be, &[k, m], &transposed(m, k, &a))).unwrap()
                    }
                    _ => up(&be, &[m, k], &a),
                };
                let b_t = match mode {
                    "b_transposed" => {
                        transpose_matrix(&up(&be, &[n, k], &transposed(k, n, &b))).unwrap()
                    }
                    _ => up(&be, &[k, n], &b),
                };
                let mut c = up(&be, &[m, n], &c0);
                be.gemm(&a_t, &b_t, &mut c, alpha, beta).expect("gemm");
                let want = reference(&a, &b, &c0, m, k, n, alpha, beta);
                assert_close(&down(&be, &c), &want, 1e-3, mode);
                assert_eq!(be.debug_launches(), vec!["gemm-f32-metal"], "{mode}");
            }
        }
    }

    #[test]
    fn beta_zero_clears_a_nan_destination() {
        let be = backend();
        let (m, k, n) = (32, 32, 32);
        let a = pattern(m, k);
        let b = pattern(k, n);
        let at = up(&be, &[m, k], &a);
        let bt = up(&be, &[k, n], &b);
        let mut c = up(&be, &[m, n], &vec![f32::NAN; m * n]);
        be.gemm(&at, &bt, &mut c, 1.0, 0.0).expect("gemm");
        assert_close(
            &down(&be, &c),
            &reference(&a, &b, &vec![0.0; m * n], m, k, n, 1.0, 0.0),
            1e-3,
            "beta0",
        );

        // alpha 0 does not read A or B, so NaN there cannot reach C.
        let an = up(&be, &[m, k], &vec![f32::NAN; m * k]);
        let bn = up(&be, &[k, n], &vec![f32::NAN; k * n]);
        let mut c = up(&be, &[m, n], &vec![f32::NAN; m * n]);
        be.gemm(&an, &bn, &mut c, 0.0, 0.0).expect("gemm alpha 0");
        assert!(down(&be, &c).iter().all(|v| *v == 0.0), "alpha 0 beta 0 gives zeros");

        // K = 0 with beta 0 gives zeros.
        let ak = up(&be, &[3, 0], &[]);
        let bk = up(&be, &[0, 5], &[]);
        let mut c = up(&be, &[3, 5], &[f32::NAN; 15]);
        be.gemm(&ak, &bk, &mut c, 1.0, 0.0).expect("gemm k0");
        assert!(down(&be, &c).iter().all(|v| *v == 0.0), "k 0 beta 0 gives zeros");
    }

    #[test]
    fn transposed_b_alloc_count_is_flat() {
        let be = backend();
        let (m, k, n) = (32, 32, 32);
        let a = up(&be, &[m, k], &pattern(m, k));
        let b_rm = up(&be, &[n, k], &pattern(n, k));
        let b = transpose_matrix(&b_rm).unwrap();
        let mut c = be.alloc(DType::F32, &[m, n]).unwrap();
        let (allocs, uploads, downloads) =
            (be.debug_allocs(), be.debug_uploads(), be.debug_downloads());
        assert_eq!(allocs, 1, "alloc counts one");
        be.gemm(&a, &b, &mut c, 1.0, 0.0).expect("gemm");
        assert_eq!(be.debug_allocs(), allocs);
        assert_eq!(be.debug_uploads(), uploads);
        assert_eq!(be.debug_downloads(), downloads);
        assert_eq!(be.debug_launches(), vec!["gemm-f32-metal"]);
        let want = reference(
            &a_vals(m, k),
            &transposed(n, k, &pattern(n, k)),
            &vec![0.0; m * n],
            m,
            k,
            n,
            1.0,
            0.0,
        );
        assert_close(&down(&be, &c), &want, 1e-3, "transposed b");
    }

    fn a_vals(m: usize, k: usize) -> Vec<f32> {
        pattern(m, k)
    }

    fn gelu_ref(x: f64) -> f64 {
        0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2))
    }

    fn gelu_prime_ref(x: f64) -> f64 {
        let phi = 0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2));
        let pdf = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
        phi + x * pdf
    }

    /// Maclaurin series; accurate to ~1e-13 for |x| <= 3.
    fn erf(x: f64) -> f64 {
        let mut term = x;
        let mut sum = x;
        for n in 1..200 {
            term *= -x * x / n as f64;
            sum += term / (2 * n + 1) as f64;
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    }

    fn grid(count: usize) -> Vec<f32> {
        (0..count).map(|i| -4.0 + 8.0 * i as f32 / (count - 1) as f32).collect()
    }

    #[test]
    fn gelu_is_one_gpu_launch_and_does_not_download() {
        let be = backend();
        let xs = grid(4097);
        let x = up(&be, &[xs.len()], &xs);
        let mut y = be.alloc(DType::F32, &[xs.len()]).unwrap();
        let (allocs, uploads) = (be.debug_allocs(), be.debug_uploads());
        be.gelu(&x, &mut y).expect("gelu");
        assert_eq!(be.debug_launches(), vec!["gelu-metal"]);
        assert_eq!(be.debug_downloads(), 0);
        assert_eq!(be.debug_allocs(), allocs);
        assert_eq!(be.debug_uploads(), uploads);
        let want: Vec<f64> = xs.iter().map(|v| gelu_ref(f64::from(*v))).collect();
        assert_close(&down(&be, &y), &want, 1e-5, "gelu");
    }

    #[test]
    fn gelu_backward_is_one_gpu_launch_and_matches_reference() {
        let be = backend();
        let xs = grid(4097);
        let dys: Vec<f32> = (0..xs.len()).map(|i| 0.25 + (i % 7) as f32 * 0.125).collect();
        let pre = up(&be, &[xs.len()], &xs);
        let dy = up(&be, &[xs.len()], &dys);
        let mut dx = be.alloc(DType::F32, &[xs.len()]).unwrap();
        let (allocs, uploads) = (be.debug_allocs(), be.debug_uploads());
        be.gelu_backward(&pre, &dy, &mut dx).expect("gelu_backward");
        assert_eq!(be.debug_launches(), vec!["gelu-backward-metal"]);
        assert_eq!(be.debug_downloads(), 0);
        assert_eq!(be.debug_allocs(), allocs);
        assert_eq!(be.debug_uploads(), uploads);
        let want: Vec<f64> = xs
            .iter()
            .zip(&dys)
            .map(|(x, d)| f64::from(*d) * gelu_prime_ref(f64::from(*x)))
            .collect();
        assert_close(&down(&be, &dx), &want, 1e-5, "gelu_backward");
    }

    #[test]
    fn f16_gemm_does_not_launch_contract_general_f16() {
        let be = backend();
        for dtype in [DType::F16, DType::BF16] {
            let a = be.alloc(dtype, &[8, 8]).unwrap();
            let b = be.alloc(dtype, &[8, 8]).unwrap();
            let mut c = be.alloc(dtype, &[8, 8]).unwrap();
            let err = be.gemm(&a, &b, &mut c, 1.0, 0.0).expect_err("half gemm is unsupported");
            assert!(matches!(err, Error::UnsupportedDevice | Error::Dtype), "{err:?}");
            assert!(be.debug_launches().is_empty(), "no launch for {dtype:?}");
            let mut y = be.alloc(dtype, &[8, 8]).unwrap();
            assert!(be.gelu(&a, &mut y).is_err());
            assert!(be.debug_launches().is_empty());
        }
    }

    #[test]
    fn repeated_f32_gemm_is_bitwise_stable() {
        let be = backend();
        let (m, k, n) = (37, 1500, 29);
        let a = up(&be, &[m, k], &pattern(m, k));
        let b = up(&be, &[k, n], &pattern(k, n));
        let mut first = be.alloc(DType::F32, &[m, n]).unwrap();
        let mut second = be.alloc(DType::F32, &[m, n]).unwrap();
        be.gemm(&a, &b, &mut first, 1.0, 0.0).unwrap();
        be.gemm(&a, &b, &mut second, 1.0, 0.0).unwrap();
        let (x, y) = (down(&be, &first), down(&be, &second));
        assert!(x.iter().any(|v| *v != 0.0));
        assert!(x.iter().zip(&y).all(|(p, q)| p.to_bits() == q.to_bits()));
    }

    #[test]
    fn alias_and_unsupported_layout_do_not_write_c() {
        let be = backend();
        let (m, k, n) = (8, 8, 8);
        let a = up(&be, &[m, k], &pattern(m, k));
        let b = up(&be, &[k, n], &pattern(k, n));
        let seed = vec![7.0f32; m * n];
        let mut c = up(&be, &[m, n], &seed);

        let alias = a.storage.clone();
        let b_alias = Tensor { storage: alias, ..clone_meta(&b) };
        assert_eq!(a.storage.storage_id(), b_alias.storage.storage_id());
        assert_eq!(be.gemm(&a, &b_alias, &mut c, 1.0, 0.0), Err(Error::Alias));
        assert_eq!(down(&be, &c), seed);

        let mut c_alias = Tensor { storage: a.storage.clone(), ..clone_meta(&c) };
        assert_eq!(be.gemm(&a, &b, &mut c_alias, 1.0, 1.0), Err(Error::Alias));

        // Both operands transposed.
        let at = transpose_matrix(&a).unwrap();
        let bt = transpose_matrix(&b).unwrap();
        assert_eq!(be.gemm(&at, &bt, &mut c, 1.0, 0.0), Err(Error::UnsupportedLayout));
        assert_eq!(down(&be, &c), seed);

        // Non-zero offset.
        let mut shifted = clone_meta(&a);
        shifted.storage = a.storage.clone();
        shifted.offset = 1;
        assert_eq!(be.gemm(&shifted, &b, &mut c, 1.0, 0.0), Err(Error::UnsupportedLayout));

        // Transposed C.
        let mut ct = transpose_matrix(&c).unwrap();
        assert_eq!(be.gemm(&a, &b, &mut ct, 1.0, 0.0), Err(Error::UnsupportedLayout));
        assert_eq!(down(&be, &c), seed);
        assert!(be.debug_launches().is_empty());
    }

    fn clone_meta(t: &T) -> T {
        Tensor {
            storage: t.storage.clone(),
            dtype: t.dtype,
            shape: t.shape.clone(),
            strides: t.strides.clone(),
            offset: t.offset,
        }
    }

    #[test]
    fn buffers_have_distinct_ids_and_clone_shares() {
        let be = backend();
        assert_eq!(be.kind(), linalg_core::DeviceKind::Metal);
        let empty_a = be.alloc(DType::F32, &[0, 4]).unwrap();
        let empty_b = be.alloc(DType::F32, &[0]).unwrap();
        let full = be.alloc(DType::F32, &[4]).unwrap();
        let ids =
            [empty_a.storage.storage_id(), empty_b.storage.storage_id(), full.storage.storage_id()];
        assert!(ids[0] != ids[1] && ids[0] != ids[2] && ids[1] != ids[2], "{ids:?}");
        assert_eq!(full.storage.clone().storage_id(), ids[2]);
        assert_eq!(full.strides, row_major_strides(&full.shape));
        assert_eq!(be.debug_allocs(), 3);
        assert_eq!(down(&be, &full), vec![0.0; 4]);
        let up_t = up(&be, &[2], &[1.5, -2.5]);
        assert_eq!(be.debug_uploads(), 1);
        assert_eq!(down(&be, &up_t), vec![1.5, -2.5]);
        assert_eq!(be.debug_downloads(), 2);
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn gemm_1024_cubed_is_within_1_30x_of_legacy() {
        use std::time::Instant;

        use linalg::abi::{alloc_shared, MetalContext, TensorView};
        use linalg::ops::contraction::gemm;

        let n = 1024;
        let vals = pattern(n, n);
        let be = backend();
        let a = up(&be, &[n, n], &vals);
        let b = up(&be, &[n, n], &vals);
        let mut c = be.alloc(DType::F32, &[n, n]).unwrap();

        let ctx = MetalContext::new().unwrap();
        let la = alloc_shared::<f32>(&ctx, n * n);
        let lb = alloc_shared::<f32>(&ctx, n * n);
        let lc = alloc_shared::<f32>(&ctx, n * n);
        // SAFETY: fresh shared buffers of at least n*n floats; no GPU work in flight.
        unsafe {
            std::ptr::copy_nonoverlapping(vals.as_ptr(), la.contents().cast::<f32>(), n * n);
            std::ptr::copy_nonoverlapping(vals.as_ptr(), lb.contents().cast::<f32>(), n * n);
        }
        let (va, vb, vc) = (
            TensorView::<f32>::row_major(&la, 0, vec![n, n]),
            TensorView::<f32>::row_major(&lb, 0, vec![n, n]),
            TensorView::<f32>::row_major(&lc, 0, vec![n, n]),
        );

        let mut legacy = Vec::new();
        let mut mine = Vec::new();
        for run in 0..7 {
            let t = Instant::now();
            gemm(&ctx, 1.0, &va, &vb, 0.0, &vc).unwrap();
            let l = t.elapsed().as_secs_f64();
            let t = Instant::now();
            be.gemm(&a, &b, &mut c, 1.0, 0.0).unwrap();
            let m = t.elapsed().as_secs_f64();
            if run >= 2 {
                legacy.push(l);
                mine.push(m);
            }
        }
        let median = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let (l, m) = (median(&mut legacy), median(&mut mine));
        eprintln!(
            "[bench] legacy {:.3} ms, metal backend {:.3} ms, ratio {:.3}",
            l * 1e3,
            m * 1e3,
            m / l
        );
        assert!(m <= 1.30 * l, "ratio {:.3} exceeds 1.30", m / l);
    }
}
