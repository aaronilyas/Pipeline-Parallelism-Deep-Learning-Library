use crate::CudaBackend;
use linalg_core::Error;

#[cfg(feature = "cuda")]
use crate::CudaStore;
#[cfg(feature = "cuda")]
use linalg_core::{
    row_major_strides, transpose_matrix, Backend, DType, DeviceKind, Identified, Tensor,
};

#[cfg(feature = "cuda")]
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for value in values {
        out.extend_from_slice(&value.to_ne_bytes());
    }
    out
}

#[cfg(feature = "cuda")]
fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap())).collect()
}

#[cfg(feature = "cuda")]
fn upload_f32(backend: &CudaBackend, shape: &[usize], values: &[f32]) -> Tensor<CudaStore> {
    backend.upload(DType::F32, shape, &f32_bytes(values)).unwrap()
}

#[cfg(feature = "cuda")]
fn metal_fixture(rows: usize, cols: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            out.push((((i * 17 + j * 3) % 100) as f32) / 100.0 - 0.5);
        }
    }
    out
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn gemm_f64(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    alpha: f32,
    beta: f32,
    c: &[f32],
) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc =
                if beta == 0.0 { 0.0 } else { f64::from(beta) * f64::from(c[row * n + col]) };
            if alpha != 0.0 {
                for inner in 0..k {
                    acc += f64::from(alpha)
                        * f64::from(a[row * k + inner])
                        * f64::from(b[inner * n + col]);
                }
            }
            out[row * n + col] = acc;
        }
    }
    out
}

#[cfg(feature = "cuda")]
fn transposed(
    backend: &CudaBackend,
    logical_rows: usize,
    logical_cols: usize,
    logical: &[f32],
) -> Tensor<CudaStore> {
    assert_eq!(logical.len(), logical_rows * logical_cols);
    let mut physical = vec![0.0; logical.len()];
    for row in 0..logical_rows {
        for col in 0..logical_cols {
            physical[col * logical_rows + row] = logical[row * logical_cols + col];
        }
    }
    let uploaded = upload_f32(backend, &[logical_cols, logical_rows], &physical);
    let view = transpose_matrix(&uploaded).unwrap();
    assert_eq!(view.shape, vec![logical_rows, logical_cols]);
    assert_eq!(view.strides, vec![1, logical_rows]);
    assert_eq!(view.offset, 0);
    view
}

#[cfg(feature = "cuda")]
fn assert_close(got: &[f32], expected: &[f64], tol: f64) {
    assert_eq!(got.len(), expected.len());
    for (index, (g, e)) in got.iter().zip(expected).enumerate() {
        let delta = (f64::from(*g) - *e).abs();
        assert!(delta <= tol, "index {index}: got {g} expected {e} delta {delta}");
    }
}

#[cfg(not(feature = "cuda"))]
#[test]
fn feature_off_new_is_unsupported_device() {
    assert!(matches!(CudaBackend::new(), Err(Error::UnsupportedDevice)));
}

#[test]
fn manifest_does_not_depend_on_linalg_cpu() {
    let manifest = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"));
    assert!(
        !manifest.contains("linalg-cpu") && !manifest.contains("linalg_cpu"),
        "linalg-cuda must not depend on linalg-cpu"
    );
}

#[cfg(feature = "cuda")]
fn require_device() -> CudaBackend {
    match CudaBackend::new() {
        Ok(backend) => {
            assert_eq!(backend.kind(), DeviceKind::Cuda);
            backend
        }
        Err(Error::UnsupportedDevice) => {
            panic!(
                "not run: host has feature cuda but CudaBackend::new returned UnsupportedDevice"
            );
        }
        Err(other) => panic!("unexpected new() error: {other:?}"),
    }
}

#[cfg(feature = "cuda")]
#[test]
fn f32_gemm_matches_f64_reference_within_1e_3() {
    let backend = require_device();
    const M: usize = 32;
    const K: usize = 32;
    const N: usize = 32;
    let a = metal_fixture(M, K);
    let b = metal_fixture(K, N);
    let c0 = metal_fixture(M, N);
    let expected = gemm_f64(M, N, K, &a, &b, 1.0, 0.0, &c0);

    let run = |a: &Tensor<CudaStore>, b: &Tensor<CudaStore>| {
        let mut c = upload_f32(&backend, &[M, N], &c0);
        backend.gemm(a, b, &mut c, 1.0, 0.0).unwrap();
        let got = f32_values(&backend.download(&c).unwrap());
        assert_close(&got, &expected, 1e-3);
    };

    let a_rm = upload_f32(&backend, &[M, K], &a);
    let b_rm = upload_f32(&backend, &[K, N], &b);
    run(&a_rm, &b_rm);

    let a_t = transposed(&backend, M, K, &a);
    run(&a_t, &b_rm);

    let b_t = transposed(&backend, K, N, &b);
    run(&a_rm, &b_t);
}

#[cfg(feature = "cuda")]
#[test]
fn transposed_operand_does_not_allocate() {
    let backend = require_device();
    let a = metal_fixture(8, 4);
    let b = metal_fixture(4, 6);
    let a_t = transposed(&backend, 8, 4, &a);
    let b_rm = upload_f32(&backend, &[4, 6], &b);
    let mut c = upload_f32(&backend, &[8, 6], &[0.0; 8 * 6]);
    let before = backend.debug_allocs();
    backend.gemm(&a_t, &b_rm, &mut c, 1.0, 0.0).unwrap();
    assert_eq!(backend.debug_allocs(), before, "transposed gemm must not allocate a packed copy");
}

#[cfg(feature = "cuda")]
#[test]
fn one_cublas_launch_name() {
    let backend = require_device();
    let a = upload_f32(&backend, &[2, 3], &[1.0; 6]);
    let b = upload_f32(&backend, &[3, 4], &[1.0; 12]);
    let mut c = upload_f32(&backend, &[2, 4], &[0.0; 8]);
    backend.gemm(&a, &b, &mut c, 1.0, 0.0).unwrap();
    assert_eq!(backend.debug_launches(), vec!["gemm-f32-cublas"]);
}

#[cfg(feature = "cuda")]
#[test]
fn gelu_does_not_download() {
    let backend = require_device();
    let xs: Vec<f32> = (-4..=4).map(|v| v as f32).collect();
    let x = upload_f32(&backend, &[xs.len()], &xs);
    let mut y = backend.alloc(DType::F32, &[xs.len()]).unwrap();
    let before = backend.debug_downloads();
    backend.gelu(&x, &mut y).unwrap();
    assert_eq!(backend.debug_downloads(), before, "gelu must not download to the host");
    let got = f32_values(&backend.download(&y).unwrap());
    for (index, (&g, &xv)) in got.iter().zip(&xs).enumerate() {
        let expected = 0.5 * xv * (1.0 + erf_ref(xv / std::f32::consts::SQRT_2));
        assert!((g - expected).abs() <= 1e-5, "index {index}: got {g} expected {expected}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn repeated_gemm_is_bitwise_stable() {
    let backend = require_device();
    let a = metal_fixture(16, 16);
    let b = metal_fixture(16, 16);
    let a_t = upload_f32(&backend, &[16, 16], &a);
    let b_t = upload_f32(&backend, &[16, 16], &b);

    let mut c1 = upload_f32(&backend, &[16, 16], &[0.0; 256]);
    backend.gemm(&a_t, &b_t, &mut c1, 1.0, 0.0).unwrap();
    let d1 = backend.download(&c1).unwrap();

    let mut c2 = upload_f32(&backend, &[16, 16], &[0.0; 256]);
    backend.gemm(&a_t, &b_t, &mut c2, 1.0, 0.0).unwrap();
    let d2 = backend.download(&c2).unwrap();

    assert_eq!(d1, d2, "repeated gemm must be bitwise stable");
}

#[cfg(feature = "cuda")]
#[test]
fn gemm_alias_returns_alias_without_writing_c() {
    let backend = require_device();
    let a = upload_f32(&backend, &[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let mut c = upload_f32(&backend, &[2, 2], &[9.0, 9.0, 9.0, 9.0]);
    let before = backend.download(&c).unwrap();
    let result = backend.gemm(&a, &a, &mut c, 1.0, 0.0);
    assert_eq!(result, Err(Error::Alias));
    assert_eq!(backend.download(&c).unwrap(), before);
}

#[cfg(feature = "cuda")]
#[test]
fn gemm_unsupported_layout_does_not_write_c() {
    let backend = require_device();
    let a = upload_f32(&backend, &[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let b = upload_f32(&backend, &[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let mut c = upload_f32(&backend, &[2, 2], &[5.0, 5.0, 5.0, 5.0]);
    // Non-zero offset is unsupported.
    c.offset = 1;
    let before = {
        c.offset = 0;
        let bytes = backend.download(&c).unwrap();
        c.offset = 1;
        bytes
    };
    let result = backend.gemm(&a, &b, &mut c, 1.0, 0.0);
    assert_eq!(result, Err(Error::UnsupportedLayout));
    c.offset = 0;
    assert_eq!(backend.download(&c).unwrap(), before);
}

#[cfg(feature = "cuda")]
#[test]
fn gemm_beta_zero_ignores_nan_in_c() {
    let backend = require_device();
    let a = upload_f32(&backend, &[2, 2], &[1.0, 0.0, 0.0, 1.0]);
    let b = upload_f32(&backend, &[2, 2], &[2.0, 3.0, 4.0, 5.0]);
    let mut c = upload_f32(&backend, &[2, 2], &[f32::NAN, f32::NAN, f32::NAN, f32::NAN]);
    backend.gemm(&a, &b, &mut c, 1.0, 0.0).unwrap();
    let got = f32_values(&backend.download(&c).unwrap());
    assert_eq!(got, vec![2.0, 3.0, 4.0, 5.0]);
}

#[cfg(feature = "cuda")]
#[test]
fn gemm_k_zero_beta_zero_writes_zeros() {
    let backend = require_device();
    let a = upload_f32(&backend, &[3, 0], &[]);
    let b = upload_f32(&backend, &[0, 4], &[]);
    let mut c = upload_f32(&backend, &[3, 4], &[1.0; 12]);
    backend.gemm(&a, &b, &mut c, 1.0, 0.0).unwrap();
    let got = f32_values(&backend.download(&c).unwrap());
    assert_eq!(got, vec![0.0; 12]);
}

#[cfg(feature = "cuda")]
#[test]
fn storage_id_distinct_and_non_null_for_empty() {
    let backend = require_device();
    let a = backend.alloc(DType::F32, &[0]).unwrap();
    let b = backend.alloc(DType::F32, &[0]).unwrap();
    assert_ne!(a.storage.storage_id(), 0);
    assert_ne!(b.storage.storage_id(), 0);
    assert_ne!(a.storage.storage_id(), b.storage.storage_id());
}

#[cfg(feature = "cuda")]
#[test]
fn gelu_backward_within_1e_5() {
    let backend = require_device();
    let xs: Vec<f32> = (-4..=4).map(|v| v as f32 * 0.5).collect();
    let dys: Vec<f32> = xs.iter().map(|x| x * 0.25 + 1.0).collect();
    let pre = upload_f32(&backend, &[xs.len()], &xs);
    let dy = upload_f32(&backend, &[xs.len()], &dys);
    let mut dx = backend.alloc(DType::F32, &[xs.len()]).unwrap();
    backend.gelu_backward(&pre, &dy, &mut dx).unwrap();
    assert_eq!(backend.debug_launches().last().copied(), Some("gelu-backward-cuda"));
    let got = f32_values(&backend.download(&dx).unwrap());
    for (index, ((&g, &x), &dyv)) in got.iter().zip(&xs).zip(&dys).enumerate() {
        let expected = dyv * gelu_prime_ref(x);
        assert!((g - expected).abs() <= 1e-5, "index {index}: got {g} expected {expected}");
    }
}

#[cfg(feature = "cuda")]
fn erf_ref(x: f32) -> f32 {
    // Abramowitz and Stegun 7.1.26, same f32 spellings as the CPU backend.
    const ERF_A1: f32 = 0.254_829_6;
    const ERF_A2: f32 = 0.284_496_72;
    const ERF_A3: f32 = 1.421_413_8;
    const ERF_A4: f32 = 1.453_152_1;
    const ERF_A5: f32 = 1.061_405_4;
    let sign = if x.is_sign_negative() { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = (((((ERF_A5 * t - ERF_A4) * t) + ERF_A3) * t - ERF_A2) * t + ERF_A1) * t;
    sign * (1.0 - poly * (-x * x).exp())
}

#[cfg(feature = "cuda")]
fn gelu_prime_ref(x: f32) -> f32 {
    const INV_SQRT_2PI: f32 = 0.398_942_3;
    let phi = 0.5 * (1.0 + erf_ref(x * std::f32::consts::FRAC_1_SQRT_2));
    let pdf = (-0.5 * x * x).exp() * INV_SQRT_2PI;
    phi + x * pdf
}

#[cfg(feature = "cuda")]
#[test]
fn empty_tensor_row_major_strides() {
    let backend = require_device();
    let t = backend.alloc(DType::F32, &[2, 0]).unwrap();
    assert_eq!(t.strides, row_major_strides(&[2, 0]));
}
