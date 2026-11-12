use crate::{CpuBackend, CpuStore};
use half::{bf16, f16};
use linalg_core::{
    row_major_strides, transpose_matrix, Backend, DType, DeviceKind, Error, Identified, Tensor,
};

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for value in values {
        out.extend_from_slice(&value.to_ne_bytes());
    }
    out
}

fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap())).collect()
}

fn u16_values(bytes: &[u8]) -> Vec<u16> {
    bytes.chunks_exact(2).map(|chunk| u16::from_ne_bytes(chunk.try_into().unwrap())).collect()
}

fn upload_f32(backend: &CpuBackend, shape: &[usize], values: &[f32]) -> Tensor<CpuStore> {
    backend.upload(DType::F32, shape, &f32_bytes(values)).unwrap()
}

fn narrow_bytes(values: &[f32], bits_of: impl Fn(f32) -> u16) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &value in values {
        out.extend_from_slice(&bits_of(value).to_ne_bytes());
    }
    out
}

/// Spec accumulation: `acc` starts at 0 when `beta` is 0, otherwise at `beta * C`,
/// then `acc += alpha * A * B` in `k` order. Products associate left.
#[allow(clippy::too_many_arguments)]
fn sequential(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &[f32],
    alpha: f32,
    beta: f32,
) -> Vec<f32> {
    let mut out = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = if beta == 0.0 { 0.0 } else { beta * c[row * n + col] };
            if alpha != 0.0 {
                for inner in 0..k {
                    acc += alpha * a[row * k + inner] * b[inner * n + col];
                }
            }
            out[row * n + col] = acc;
        }
    }
    out
}

fn assert_bits(got: &[u8], expected: &[f32]) {
    let got = f32_values(got);
    assert_eq!(got.len(), expected.len());
    for (index, (got, expected)) in got.iter().zip(expected).enumerate() {
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "index {index}: got {got} expected {expected}"
        );
    }
}

fn transposed(
    backend: &CpuBackend,
    logical_rows: usize,
    logical_cols: usize,
    logical: &[f32],
) -> Tensor<CpuStore> {
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
    assert_eq!(view.dtype, DType::F32);
    assert_eq!(view.storage.storage_id(), uploaded.storage.storage_id());
    view
}

#[allow(clippy::too_many_arguments)]
fn gemm_layouts(
    a: &[f32],
    b: &[f32],
    c0: &[f32],
    m: usize,
    k: usize,
    n: usize,
    alpha: f32,
    beta: f32,
    expected: &[f32],
) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c0.len(), m * n);
    assert_eq!(expected.len(), m * n);

    let run = |a: &Tensor<CpuStore>, b: &Tensor<CpuStore>, backend: &CpuBackend| {
        let mut c = upload_f32(backend, &[m, n], c0);
        backend.gemm(a, b, &mut c, alpha, beta).unwrap();
        assert_bits(&backend.download(&c).unwrap(), expected);
    };

    let backend = CpuBackend::new();
    let a_rm = upload_f32(&backend, &[m, k], a);
    let b_rm = upload_f32(&backend, &[k, n], b);
    run(&a_rm, &b_rm, &backend);

    let backend = CpuBackend::new();
    let a_t = transposed(&backend, m, k, a);
    let b_rm = upload_f32(&backend, &[k, n], b);
    run(&a_t, &b_rm, &backend);

    let backend = CpuBackend::new();
    let a_rm = upload_f32(&backend, &[m, k], a);
    let b_t = transposed(&backend, k, n, b);
    run(&a_rm, &b_t, &backend);
}

#[test]
fn f32_gemm_matches_sequential_summation_bit_for_bit() {
    let backend = CpuBackend::new();
    assert_eq!(backend.kind(), DeviceKind::Cpu);

    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [5.0, 6.0, 7.0, 8.0];
    let c0 = [0.0; 4];
    let expected = sequential(2, 2, 2, &a, &b, &c0, 1.0, 0.0);
    assert_eq!(expected, vec![19.0, 22.0, 43.0, 50.0]);
    gemm_layouts(&a, &b, &c0, 2, 2, 2, 1.0, 0.0, &expected);

    let c0 = [1.0, -2.0, 0.5, 4.0];
    let alpha = -0.5;
    let beta = 0.25;
    let expected = sequential(2, 2, 2, &a, &b, &c0, alpha, beta);
    // Written out for C[0, 0] so the oracle loop is not the only statement of the sum.
    let mut acc = beta * c0[0];
    acc += alpha * a[0] * b[0];
    acc += alpha * a[1] * b[2];
    assert_eq!(expected[0].to_bits(), acc.to_bits());
    gemm_layouts(&a, &b, &c0, 2, 2, 2, alpha, beta, &expected);

    // 2^25 has f32 ulp 4, so the required k order loses the 1. Cancelling the large
    // terms first, or accumulating in f64, keeps it.
    let big = 33554432.0f32;
    let a = [big, 1.0, -big];
    let b = [1.0, 1.0, 1.0];
    let mut acc = 0.0f32;
    acc += 1.0 * a[0] * b[0];
    acc += 1.0 * a[1] * b[1];
    acc += 1.0 * a[2] * b[2];
    assert_eq!(acc.to_bits(), 0);
    let mut cancelled_first = 0.0f32;
    cancelled_first += big;
    cancelled_first += -big;
    cancelled_first += 1.0;
    assert_eq!(cancelled_first.to_bits(), 1.0f32.to_bits());
    let mut wide = 0.0f64;
    for value in a {
        wide += f64::from(value);
    }
    assert_eq!(wide, 1.0);
    gemm_layouts(&a, &b, &[0.0], 1, 3, 1, 1.0, 0.0, &[acc]);

    // `alpha * A * B` associates to the left. The other grouping is a different value.
    let alpha = 1e20_f32;
    let left = (alpha * 1e-20_f32) * 1e-20_f32;
    let right = alpha * (1e-20_f32 * 1e-20_f32);
    assert_ne!(left.to_bits(), right.to_bits());
    gemm_layouts(&[1e-20], &[1e-20], &[0.0], 1, 1, 1, alpha, 0.0, &[left]);
}

#[test]
fn beta_zero_ignores_nan_in_c_and_alpha_zero_ignores_nan_in_operands() {
    let nan = f32::from_bits(0x7fc0_1234);
    assert!(nan.is_nan());

    let backend = CpuBackend::new();
    let a = upload_f32(&backend, &[1, 2], &[2.0, 3.0]);
    let b = upload_f32(&backend, &[2, 1], &[4.0, 5.0]);
    let mut c = upload_f32(&backend, &[1, 1], &[nan]);
    backend.gemm(&a, &b, &mut c, 1.0, 0.0).unwrap();
    let got = f32_values(&backend.download(&c).unwrap());
    assert_eq!(got[0].to_bits(), 23.0f32.to_bits());

    let backend = CpuBackend::new();
    let a = upload_f32(&backend, &[1, 2], &[nan, f32::from_bits(0x7f80_0011)]);
    let b = upload_f32(&backend, &[2, 2], &[nan, nan, nan, nan]);
    let mut c = upload_f32(&backend, &[1, 2], &[3.0, -4.0]);
    backend.gemm(&a, &b, &mut c, 0.0, 2.0).unwrap();
    assert_bits(&backend.download(&c).unwrap(), &[6.0, -8.0]);

    // Both factors are zero, so neither the NaN operands nor the NaN output are read.
    let backend = CpuBackend::new();
    let a = upload_f32(&backend, &[2, 2], &[nan; 4]);
    let b = upload_f32(&backend, &[2, 2], &[nan; 4]);
    let mut c = upload_f32(&backend, &[2, 2], &[nan; 4]);
    backend.gemm(&a, &b, &mut c, 0.0, 0.0).unwrap();
    assert_bits(&backend.download(&c).unwrap(), &[0.0; 4]);
}

#[test]
fn k_zero_beta_zero_stores_zeros() {
    let backend = CpuBackend::new();
    let a = backend.alloc(DType::F32, &[2, 0]).unwrap();
    let b = backend.alloc(DType::F32, &[0, 3]).unwrap();
    assert_ne!(a.storage.storage_id(), b.storage.storage_id());
    assert_eq!(a.shape, vec![2, 0]);
    assert_eq!(a.strides, row_major_strides(&[2, 0]));
    assert_eq!(backend.download(&a).unwrap(), Vec::<u8>::new());

    let nan = f32::from_bits(0x7fa0_00ab);
    let mut c = upload_f32(&backend, &[2, 3], &[nan; 6]);
    let launches = backend.debug_launches().len();
    backend.gemm(&a, &b, &mut c, 1.5, 0.0).unwrap();
    assert_eq!(&backend.download(&c).unwrap(), &vec![0u8; 24]);
    assert_eq!(backend.debug_launches().len(), launches + 1);
    assert_eq!(backend.debug_launches().last().copied(), Some("gemm-f32"));

    // K = 0 with a non-zero beta scales C and does not invent zeros.
    let a = backend.alloc(DType::F32, &[1, 0]).unwrap();
    let b = backend.alloc(DType::F32, &[0, 2]).unwrap();
    let mut c = upload_f32(&backend, &[1, 2], &[1.5, -3.0]);
    backend.gemm(&a, &b, &mut c, 4.0, 2.0).unwrap();
    assert_bits(&backend.download(&c).unwrap(), &[3.0, -6.0]);
}

#[test]
fn unsupported_layout_and_alias_leave_c_unchanged() {
    let backend = CpuBackend::new();
    let sentinel = [1.25, -2.5, 3.5, -4.5];
    let bytes = f32_bytes(&sentinel);

    let mut c = upload_f32(&backend, &[2, 2], &sentinel);
    let a = upload_f32(&backend, &[2, 2], &[1.0, 0.0, 0.0, 1.0]);
    let b = upload_f32(&backend, &[2, 2], &[9.0, 8.0, 7.0, 6.0]);
    let mut a_bad = upload_f32(&backend, &[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    a_bad.strides = vec![1, 1];
    let launches = backend.debug_launches();
    assert_eq!(backend.gemm(&a_bad, &b, &mut c, 1.0, 0.0).err(), Some(Error::UnsupportedLayout));
    assert_eq!(backend.download(&c).unwrap(), bytes);
    assert_eq!(backend.debug_launches(), launches);

    let a_t = transposed(&backend, 2, 2, &[1.0, 2.0, 3.0, 4.0]);
    let b_t = transposed(&backend, 2, 2, &[5.0, 6.0, 7.0, 8.0]);
    assert_eq!(backend.gemm(&a_t, &b_t, &mut c, 1.0, 1.0).err(), Some(Error::UnsupportedLayout));
    assert_eq!(backend.download(&c).unwrap(), bytes);

    let mut c_view = transpose_matrix(&c).unwrap();
    assert_eq!(c_view.storage.storage_id(), c.storage.storage_id());
    assert_eq!(backend.gemm(&a, &b, &mut c_view, 1.0, 0.0).err(), Some(Error::UnsupportedLayout));
    assert_eq!(backend.download(&c).unwrap(), bytes);

    let mut shifted = upload_f32(&backend, &[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    shifted.offset = 1;
    assert_eq!(backend.gemm(&shifted, &b, &mut c, 1.0, 0.0).err(), Some(Error::UnsupportedLayout));
    assert_eq!(backend.download(&c).unwrap(), bytes);

    let alias_a = Tensor {
        storage: c.storage.clone(),
        dtype: DType::F32,
        shape: vec![2, 2],
        strides: row_major_strides(&[2, 2]),
        offset: 0,
    };
    assert_eq!(alias_a.storage.storage_id(), c.storage.storage_id());
    assert_eq!(backend.gemm(&alias_a, &b, &mut c, 1.0, 0.0).err(), Some(Error::Alias));
    assert_eq!(backend.download(&c).unwrap(), bytes);

    let alias_b = Tensor {
        storage: c.storage.clone(),
        dtype: DType::F32,
        shape: vec![2, 2],
        strides: row_major_strides(&[2, 2]),
        offset: 0,
    };
    assert_eq!(backend.gemm(&a, &alias_b, &mut c, 1.0, 0.0).err(), Some(Error::Alias));
    assert_eq!(backend.download(&c).unwrap(), bytes);

    let shared = upload_f32(&backend, &[2, 2], &[4.0, 3.0, 2.0, 1.0]);
    let alias_ab = Tensor {
        storage: shared.storage.clone(),
        dtype: DType::F32,
        shape: vec![2, 2],
        strides: row_major_strides(&[2, 2]),
        offset: 0,
    };
    assert_eq!(backend.gemm(&shared, &alias_ab, &mut c, 1.0, 0.0).err(), Some(Error::Alias));
    assert_eq!(backend.download(&c).unwrap(), bytes);
    assert_eq!(backend.debug_launches(), launches);
}

#[test]
fn transposed_gemm_alloc_count_is_flat() {
    let backend = CpuBackend::new();
    assert_eq!(backend.debug_allocs(), 0);
    assert_eq!(backend.debug_uploads(), 0);
    assert_eq!(backend.debug_downloads(), 0);

    let scalar = backend.alloc(DType::BF16, &[]).unwrap();
    assert_eq!(scalar.dtype, DType::BF16);
    assert_eq!(scalar.shape, Vec::<usize>::new());
    assert_eq!(scalar.strides, Vec::<usize>::new());
    assert_eq!(scalar.offset, 0);
    assert_eq!(backend.debug_allocs(), 1);
    assert_eq!(backend.download(&scalar).unwrap(), vec![0, 0]);
    assert_eq!(backend.debug_downloads(), 1);

    let matrix = backend.alloc(DType::F32, &[2, 3]).unwrap();
    assert_eq!(matrix.strides, vec![3, 1]);
    assert_eq!(backend.debug_allocs(), 2);
    assert_eq!(backend.download(&matrix).unwrap(), vec![0u8; 24]);
    assert_ne!(matrix.storage.storage_id(), scalar.storage.storage_id());
    let cloned = matrix.storage.clone();
    assert_eq!(cloned.storage_id(), matrix.storage.storage_id());
    assert_eq!(backend.debug_allocs(), 2);

    let uploads = backend.debug_uploads();
    let allocs = backend.debug_allocs();
    assert_eq!(backend.upload(DType::F32, &[2, 2], &[0u8; 4]).err(), Some(Error::Dtype));
    assert_eq!(backend.upload(DType::F16, &[2], &[0, 0]).err(), Some(Error::Dtype));
    assert_eq!(backend.upload(DType::BF16, &[1], &[0, 0, 0]).err(), Some(Error::Dtype));
    assert_eq!(backend.debug_uploads(), uploads);
    assert_eq!(backend.debug_allocs(), allocs);

    let f16_matrix = backend.upload(DType::F16, &[2], &[1, 2, 3, 4]).unwrap();
    assert_eq!(backend.debug_uploads(), uploads + 1);
    assert_eq!(backend.debug_allocs(), allocs);
    assert_eq!(backend.download(&f16_matrix).unwrap(), vec![1, 2, 3, 4]);

    let a_logical = [1.0, 2.0, 3.0, 4.0];
    let b_logical = [5.0, 6.0, 7.0, 8.0];
    let expected = [19.0, 22.0, 43.0, 50.0];

    let a = transposed(&backend, 2, 2, &a_logical);
    let b = upload_f32(&backend, &[2, 2], &b_logical);
    let mut c = backend.alloc(DType::F32, &[2, 2]).unwrap();
    let allocs = backend.debug_allocs();
    let uploads = backend.debug_uploads();
    let downloads = backend.debug_downloads();
    let launches = backend.debug_launches().len();
    assert!(allocs > 0);
    backend.gemm(&a, &b, &mut c, 1.0, 0.0).unwrap();
    assert_eq!(backend.debug_allocs(), allocs);
    assert_eq!(backend.debug_uploads(), uploads);
    assert_eq!(backend.debug_downloads(), downloads);
    assert_eq!(backend.debug_launches().len(), launches + 1);
    assert_bits(&backend.download(&c).unwrap(), &expected);

    let a = upload_f32(&backend, &[2, 2], &a_logical);
    let b = transposed(&backend, 2, 2, &b_logical);
    let mut c = backend.alloc(DType::F32, &[2, 2]).unwrap();
    let allocs = backend.debug_allocs();
    let uploads = backend.debug_uploads();
    let downloads = backend.debug_downloads();
    backend.gemm(&a, &b, &mut c, 1.0, 0.0).unwrap();
    assert_eq!(backend.debug_allocs(), allocs);
    assert_eq!(backend.debug_uploads(), uploads);
    assert_eq!(backend.debug_downloads(), downloads);
    assert_bits(&backend.download(&c).unwrap(), &expected);
}

#[test]
fn one_launch_for_a_10000_element_gemm_and_for_gelu() {
    let backend = CpuBackend::new();
    assert!(backend.debug_launches().is_empty());

    let a = upload_f32(&backend, &[100, 1], &[1.0; 100]);
    let b = upload_f32(&backend, &[1, 100], &[1.0; 100]);
    let mut c = upload_f32(&backend, &[100, 100], &[f32::NAN; 10_000]);
    assert_eq!(c.shape.iter().product::<usize>(), 10_000);
    backend.gemm(&a, &b, &mut c, 1.0, 0.0).unwrap();
    assert_eq!(backend.debug_launches(), vec!["gemm-f32"]);
    let got = f32_values(&backend.download(&c).unwrap());
    assert_eq!(got.len(), 10_000);
    assert!(got.iter().all(|value| value.to_bits() == 1.0f32.to_bits()));

    let x = upload_f32(&backend, &[10_000], &[1.0; 10_000]);
    let mut y = backend.alloc(DType::F32, &[10_000]).unwrap();
    backend.gelu(&x, &mut y).unwrap();
    assert_eq!(backend.debug_launches(), vec!["gemm-f32", "gelu"]);
    let gelu_one = f32_values(&backend.download(&y).unwrap());
    assert_eq!(gelu_one.len(), 10_000);
    assert!(gelu_one.iter().all(|value| *value != 0.0 && *value != 1.0));

    let empty_m_a = backend.alloc(DType::F32, &[0, 4]).unwrap();
    let empty_m_b = backend.alloc(DType::F32, &[4, 5]).unwrap();
    let mut empty_m_c = backend.alloc(DType::F32, &[0, 5]).unwrap();
    backend.gemm(&empty_m_a, &empty_m_b, &mut empty_m_c, 1.0, 0.0).unwrap();

    let empty_n_a = backend.alloc(DType::F32, &[3, 2]).unwrap();
    let empty_n_b = backend.alloc(DType::F32, &[2, 0]).unwrap();
    let mut empty_n_c = backend.alloc(DType::F32, &[3, 0]).unwrap();
    backend.gemm(&empty_n_a, &empty_n_b, &mut empty_n_c, 1.0, 1.0).unwrap();

    let empty_k_a = backend.alloc(DType::F32, &[2, 0]).unwrap();
    let empty_k_b = backend.alloc(DType::F32, &[0, 2]).unwrap();
    let mut empty_k_c = upload_f32(&backend, &[2, 2], &[1.0, 1.0, 1.0, 1.0]);
    backend.gemm(&empty_k_a, &empty_k_b, &mut empty_k_c, 1.0, 0.0).unwrap();
    assert_bits(&backend.download(&empty_k_c).unwrap(), &[0.0; 4]);

    assert_eq!(
        backend.debug_launches(),
        vec!["gemm-f32", "gelu", "gemm-f32", "gemm-f32", "gemm-f32"]
    );
}

/// `erf` from its integral, for an oracle that does not share the implementation's polynomial.
fn erf_integral(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    if x == 0.0 {
        return 0.0;
    }
    if x > 8.0 {
        return sign;
    }
    let intervals = 2048u32;
    let step = x / f64::from(intervals);
    let mut sum = 1.0 + (-x * x).exp();
    for index in 1..intervals {
        let t = f64::from(index) * step;
        let weight = if index % 2 == 0 { 2.0 } else { 4.0 };
        sum += weight * (-t * t).exp();
    }
    sign * (2.0 / std::f64::consts::PI.sqrt()) * sum * step / 3.0
}

fn gelu_ref(x: f64) -> f64 {
    0.5 * x * (1.0 + erf_integral(x / std::f64::consts::SQRT_2))
}

fn gelu_prime_ref(x: f64) -> f64 {
    let phi = 0.5 * (1.0 + erf_integral(x / std::f64::consts::SQRT_2));
    let pdf = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
    phi + x * pdf
}

fn assert_close(got: f32, expected: f64) {
    let diff = (f64::from(got) - expected).abs();
    assert!(diff <= 1e-6, "got {got} expected {expected} abs diff {diff}");
}

#[test]
fn gelu_and_backward_match_the_normal_cdf_definition_within_1e_6_absolute() {
    let mut samples = Vec::new();
    for index in -24..=24 {
        samples.push(index as f32 * 0.25);
    }
    samples.extend([4.75, -4.75, 8.0, -8.0, 1.0e-4, -1.0e-3]);

    let backend = CpuBackend::new();
    let x = upload_f32(&backend, &[samples.len()], &samples);
    let mut y = backend.alloc(DType::F32, &[samples.len()]).unwrap();
    let launches = backend.debug_launches().len();
    backend.gelu(&x, &mut y).unwrap();
    assert_eq!(backend.debug_launches().len(), launches + 1);
    assert_eq!(backend.debug_launches().last().copied(), Some("gelu"));
    let got = f32_values(&backend.download(&y).unwrap());
    for (got, sample) in got.iter().zip(&samples) {
        assert_close(*got, gelu_ref(f64::from(*sample)));
    }

    let matrix = upload_f32(&backend, &[2, 3], &samples[..6]);
    let mut matrix_y = backend.alloc(DType::F32, &[2, 3]).unwrap();
    backend.gelu(&matrix, &mut matrix_y).unwrap();
    let got = f32_values(&backend.download(&matrix_y).unwrap());
    for (got, sample) in got.iter().zip(samples.iter().take(6)) {
        assert_close(*got, gelu_ref(f64::from(*sample)));
    }

    let dy_values: Vec<f32> = samples.iter().map(|value| value * 0.5 - 0.25).collect();
    let pre = upload_f32(&backend, &[samples.len()], &samples);
    let dy = upload_f32(&backend, &[samples.len()], &dy_values);
    let mut dx = backend.alloc(DType::F32, &[samples.len()]).unwrap();
    backend.gelu_backward(&pre, &dy, &mut dx).unwrap();
    assert_eq!(backend.debug_launches().last().copied(), Some("gelu-backward"));
    let got = f32_values(&backend.download(&dx).unwrap());
    for ((got, sample), dy) in got.iter().zip(&samples).zip(&dy_values) {
        assert_close(*got, f64::from(*dy) * gelu_prime_ref(f64::from(*sample)));
    }

    let sentinel = [9.0, 8.0, 7.0, 6.0];
    let sentinel_bytes = f32_bytes(&sentinel);
    let mut y = upload_f32(&backend, &[4], &sentinel);
    let f16_x = backend.upload(DType::F16, &[4], &[0u8; 8]).unwrap();
    let launches = backend.debug_launches();
    assert_eq!(backend.gelu(&f16_x, &mut y).err(), Some(Error::Dtype));
    assert_eq!(backend.download(&y).unwrap(), sentinel_bytes);
    assert_eq!(backend.debug_launches(), launches);

    let mut bad = upload_f32(&backend, &[4], &[1.0, 2.0, 3.0, 4.0]);
    bad.strides = vec![2];
    assert_eq!(backend.gelu(&bad, &mut y).err(), Some(Error::UnsupportedLayout));
    assert_eq!(backend.download(&y).unwrap(), sentinel_bytes);
    assert_eq!(backend.debug_launches(), launches);

    let mut dx = upload_f32(&backend, &[4], &sentinel);
    let pre = upload_f32(&backend, &[4], &[1.0, 2.0, 3.0, 4.0]);
    let dy_f16 = backend.upload(DType::BF16, &[4], &[0u8; 8]).unwrap();
    assert_eq!(backend.gelu_backward(&pre, &dy_f16, &mut dx).err(), Some(Error::Dtype));
    assert_eq!(backend.download(&dx).unwrap(), sentinel_bytes);

    let mut dy_bad = upload_f32(&backend, &[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    dy_bad.strides = vec![1, 2];
    let pre = upload_f32(&backend, &[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let mut dx = upload_f32(&backend, &[2, 2], &sentinel);
    assert_eq!(backend.gelu_backward(&pre, &dy_bad, &mut dx).err(), Some(Error::UnsupportedLayout));
    assert_eq!(backend.download(&dx).unwrap(), sentinel_bytes);
    assert_eq!(backend.debug_launches(), launches);
}

#[test]
fn gelu_does_not_download_upload_or_allocate() {
    let backend = CpuBackend::new();
    let x = upload_f32(&backend, &[4], &[0.0, 0.5, -1.0, 2.0]);
    let mut y = backend.alloc(DType::F32, &[4]).unwrap();
    let pre = upload_f32(&backend, &[4], &[0.0, 0.5, -1.0, 2.0]);
    let dy = upload_f32(&backend, &[4], &[1.0, -1.0, 0.5, 2.0]);
    let mut dx = backend.alloc(DType::F32, &[4]).unwrap();

    let allocs = backend.debug_allocs();
    let uploads = backend.debug_uploads();
    let downloads = backend.debug_downloads();
    let launches = backend.debug_launches().len();
    backend.gelu(&x, &mut y).unwrap();
    backend.gelu_backward(&pre, &dy, &mut dx).unwrap();
    assert_eq!(backend.debug_allocs(), allocs);
    assert_eq!(backend.debug_uploads(), uploads);
    assert_eq!(backend.debug_downloads(), downloads);
    assert_eq!(&backend.debug_launches()[launches..], &["gelu", "gelu-backward"][..]);

    let got_y = f32_values(&backend.download(&y).unwrap());
    let got_dx = f32_values(&backend.download(&dx).unwrap());
    assert_ne!(got_y, vec![0.0; 4]);
    assert!(got_dx.iter().any(|value| *value != 0.0));
}

#[test]
fn f16_and_bf16_round_the_f32_sum_nearest_even() {
    // f16 around 2048 has ulp 2. 2049 and 2051 are ties; even mantissas are 2048 and 2052.
    // Summing in f16 would collapse 2048+1+1 to 2048. The f32 sum is 2050, which is exact.
    assert_eq!(f16::from_f32_const(2048.0).to_bits(), 0x6800);
    assert_eq!(f16::from_f32_const(2050.0).to_bits(), 0x6801);
    assert_eq!(f16::from_f32_const(2052.0).to_bits(), 0x6802);
    assert_eq!(f16::from_f32_const(2049.0).to_bits(), 0x6800);
    assert_eq!(f16::from_f32_const(2051.0).to_bits(), 0x6802);
    assert_eq!(bf16::from_f32_const(1.0).to_bits(), 0x3f80);
    assert_eq!(bf16::from_f32_const(1.0 + 2.0f32.powi(-7)).to_bits(), 0x3f81);
    assert_eq!(bf16::from_f32_const(1.0 + 2.0f32.powi(-6)).to_bits(), 0x3f82);

    struct Case {
        dtype: DType,
        a: &'static [f32],
        b: &'static [f32],
        c: &'static [f32],
        alpha: f32,
        beta: f32,
        expected: u16,
        launch: &'static str,
    }

    // 2^-8 is the exact halfway step at 1.0 for bf16 (ulp 2^-7).
    const TIE_DOWN: f32 = 1.0 / 256.0;
    const ONE_PLUS_ULP: f32 = 1.0 + 1.0 / 128.0;
    let cases = [
        Case {
            dtype: DType::F16,
            a: &[2048.0, 1.0],
            b: &[1.0, 1.0],
            c: &[0.0],
            alpha: 1.0,
            beta: 0.0,
            expected: 0x6800,
            launch: "gemm-f16",
        },
        Case {
            dtype: DType::F16,
            a: &[2050.0, 1.0],
            b: &[1.0, 1.0],
            c: &[0.0],
            alpha: 1.0,
            beta: 0.0,
            expected: 0x6802,
            launch: "gemm-f16",
        },
        Case {
            dtype: DType::F16,
            a: &[2048.0, 1.0, 1.0],
            b: &[1.0, 1.0, 1.0],
            c: &[0.0],
            alpha: 1.0,
            beta: 0.0,
            expected: 0x6801,
            launch: "gemm-f16",
        },
        Case {
            dtype: DType::F16,
            a: &[2048.0, 1.0, 0.5],
            b: &[1.0, 1.0, 1.0],
            c: &[0.0],
            alpha: 1.0,
            beta: 0.0,
            expected: 0x6801,
            launch: "gemm-f16",
        },
        Case {
            dtype: DType::F16,
            a: &[-2048.0, -1.0],
            b: &[1.0, 1.0],
            c: &[0.0],
            alpha: 1.0,
            beta: 0.0,
            expected: 0xe800,
            launch: "gemm-f16",
        },
        Case {
            dtype: DType::F16,
            a: &[2048.0, 1.0],
            b: &[1.0, 1.0],
            c: &[1.0],
            alpha: 1.0,
            beta: 1.0,
            expected: 0x6801,
            launch: "gemm-f16",
        },
        Case {
            dtype: DType::BF16,
            a: &[1.0, TIE_DOWN],
            b: &[1.0, 1.0],
            c: &[0.0],
            alpha: 1.0,
            beta: 0.0,
            expected: 0x3f80,
            launch: "gemm-bf16",
        },
        Case {
            dtype: DType::BF16,
            a: &[ONE_PLUS_ULP, TIE_DOWN],
            b: &[1.0, 1.0],
            c: &[0.0],
            alpha: 1.0,
            beta: 0.0,
            expected: 0x3f82,
            launch: "gemm-bf16",
        },
        Case {
            dtype: DType::BF16,
            a: &[1.0, TIE_DOWN, TIE_DOWN],
            b: &[1.0, 1.0, 1.0],
            c: &[0.0],
            alpha: 1.0,
            beta: 0.0,
            expected: 0x3f81,
            launch: "gemm-bf16",
        },
        Case {
            dtype: DType::BF16,
            a: &[-1.0, -TIE_DOWN],
            b: &[1.0, 1.0],
            c: &[0.0],
            alpha: 1.0,
            beta: 0.0,
            expected: 0xbf80,
            launch: "gemm-bf16",
        },
        Case {
            dtype: DType::BF16,
            a: &[1.0, TIE_DOWN],
            b: &[1.0, 1.0],
            c: &[TIE_DOWN],
            alpha: 1.0,
            beta: 1.0,
            expected: 0x3f81,
            launch: "gemm-bf16",
        },
    ];

    let bits_of = |dtype: DType, value: f32| match dtype {
        DType::F16 => f16::from_f32_const(value).to_bits(),
        DType::BF16 => bf16::from_f32_const(value).to_bits(),
        DType::F32 => unreachable!(),
    };

    for case in cases {
        for &value in case.a.iter().chain(case.b).chain(case.c) {
            let roundtrip = match case.dtype {
                DType::F16 => f16::from_bits(bits_of(case.dtype, value)).to_f32(),
                DType::BF16 => bf16::from_bits(bits_of(case.dtype, value)).to_f32(),
                DType::F32 => unreachable!(),
            };
            assert_eq!(
                roundtrip.to_bits(),
                value.to_bits(),
                "input {value} is not exact in {:?}",
                case.dtype
            );
        }
        let k = case.a.len();
        assert_eq!(case.b.len(), k);
        assert_eq!(case.c.len(), 1);
        let backend = CpuBackend::new();
        let pack = |values: &[f32]| narrow_bytes(values, |value| bits_of(case.dtype, value));
        let a = backend.upload(case.dtype, &[1, k], &pack(case.a)).unwrap();
        let b = backend.upload(case.dtype, &[k, 1], &pack(case.b)).unwrap();
        let mut c = backend.upload(case.dtype, &[1, 1], &pack(case.c)).unwrap();
        let launches = backend.debug_launches().len();
        backend.gemm(&a, &b, &mut c, case.alpha, case.beta).unwrap();
        assert_eq!(backend.debug_launches().len(), launches + 1);
        assert_eq!(backend.debug_launches().last().copied(), Some(case.launch));
        assert_eq!(u16_values(&backend.download(&c).unwrap()), vec![case.expected]);
    }

    // A NaN payload in an unread f16 operand must not reach C when alpha is 0.
    let backend = CpuBackend::new();
    let nan_bits = f16::NAN.to_bits();
    let a = backend.upload(DType::F16, &[1, 1], &nan_bits.to_ne_bytes()).unwrap();
    let b = backend.upload(DType::F16, &[1, 1], &nan_bits.to_ne_bytes()).unwrap();
    let two = f16::from_f32_const(2.0).to_bits().to_ne_bytes();
    let mut c = backend.upload(DType::F16, &[1, 1], &two).unwrap();
    backend.gemm(&a, &b, &mut c, 0.0, 1.0).unwrap();
    assert_eq!(
        u16_values(&backend.download(&c).unwrap()),
        vec![f16::from_f32_const(2.0).to_bits()]
    );
}
