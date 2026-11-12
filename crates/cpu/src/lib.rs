//! CPU backend for the device-independent linalg contract.
//!
//! `gemm`, `gelu`, and `gelu_backward` share one math path in test and release.
//! Test builds record counters. Release builds of those three functions take no
//! mutex and perform no atomic.

use std::sync::Arc;

use half::{bf16, f16};
use linalg_core::{row_major_strides, Backend, DType, DeviceKind, Error, Identified, Tensor};

/// Owns one byte buffer. The id is the address of this allocation, not the data
/// pointer: a zero-length buffer has no unique data address, and [`Clone`] only
/// bumps the refcount.
///
/// The data pointer and length are captured once at construction. `len` and
/// `as_mut_ptr` copy those fields. Neither forms a `&mut Vec<u8>` or a
/// `&mut [u8]` from `&self`.
struct Allocation {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: `ptr` and `len` uniquely own one `Box<[u8]>` (an empty box keeps its
// dangling pointer). Moving `Allocation` moves that owner. No method on
// `&Allocation` forms a reference to the bytes, so a move does not invalidate
// an outstanding alias.
unsafe impl Send for Allocation {}

// SAFETY: `&Allocation` only copies the captured pointer and length. Byte reads
// and writes go through that pointer, not through a reference held by this
// type. Sharing is sound because no thread may read or write a storage id while
// another call writes that same storage id.
unsafe impl Sync for Allocation {}

impl Allocation {
    fn new(bytes: Vec<u8>) -> Self {
        let boxed = bytes.into_boxed_slice();
        let len = boxed.len();
        let ptr = Box::into_raw(boxed).cast::<u8>();
        Self { ptr, len }
    }

    fn len(&self) -> usize {
        self.len
    }

    /// Captured at construction. The buffer is never reallocated, so the pointer
    /// stays valid for the life of this allocation.
    fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        // SAFETY: `ptr` and `len` came from `Box::<[u8]>::into_raw` and this
        // drop is the unique owner of that box.
        unsafe {
            drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(self.ptr, self.len)));
        }
    }
}

#[derive(Clone)]
pub struct CpuStore {
    allocation: Arc<Allocation>,
}

impl Identified for CpuStore {
    fn storage_id(&self) -> u64 {
        Arc::as_ptr(&self.allocation) as u64
    }
}

#[cfg(test)]
struct Stats {
    allocs: u64,
    downloads: u64,
    uploads: u64,
    launches: Vec<&'static str>,
}

pub struct CpuBackend {
    #[cfg(test)]
    stats: std::sync::Mutex<Stats>,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            #[cfg(test)]
            stats: std::sync::Mutex::new(Stats {
                allocs: 0,
                downloads: 0,
                uploads: 0,
                launches: Vec::new(),
            }),
        }
    }

    fn bump_allocs(&self) {
        #[cfg(test)]
        {
            self.stats.lock().expect("cpu stats").allocs += 1;
        }
        #[cfg(not(test))]
        {
            let _ = self;
        }
    }

    fn bump_uploads(&self) {
        #[cfg(test)]
        {
            self.stats.lock().expect("cpu stats").uploads += 1;
        }
        #[cfg(not(test))]
        {
            let _ = self;
        }
    }

    fn bump_downloads(&self) {
        #[cfg(test)]
        {
            self.stats.lock().expect("cpu stats").downloads += 1;
        }
        #[cfg(not(test))]
        {
            let _ = self;
        }
    }

    fn record(&self, name: &'static str) {
        #[cfg(test)]
        {
            self.stats.lock().expect("cpu stats").launches.push(name);
        }
        #[cfg(not(test))]
        {
            let _ = name;
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    type Store = CpuStore;

    fn kind(&self) -> DeviceKind {
        DeviceKind::Cpu
    }

    fn alloc(&self, dtype: DType, shape: &[usize]) -> Result<Tensor<Self::Store>, Error> {
        let tensor = owned_tensor(dtype, shape, vec![0u8; nbytes(dtype, shape)?]);
        self.bump_allocs();
        Ok(tensor)
    }

    fn upload(
        &self,
        dtype: DType,
        shape: &[usize],
        row_major: &[u8],
    ) -> Result<Tensor<Self::Store>, Error> {
        if row_major.len() != nbytes(dtype, shape)? {
            return Err(Error::Dtype);
        }
        let tensor = owned_tensor(dtype, shape, row_major.to_vec());
        self.bump_uploads();
        Ok(tensor)
    }

    fn download(&self, tensor: &Tensor<Self::Store>) -> Result<Vec<u8>, Error> {
        if tensor.strides != row_major_strides(&tensor.shape) {
            return Err(Error::UnsupportedLayout);
        }
        let size = tensor.dtype.size_of();
        let count = numel(&tensor.shape)?;
        let start = tensor.offset.checked_mul(size).ok_or(Error::Shape)?;
        let len = count.checked_mul(size).ok_or(Error::Shape)?;
        let end = start.checked_add(len).ok_or(Error::Shape)?;
        if end > tensor.storage.allocation.len() {
            return Err(Error::Shape);
        }
        let out = unsafe {
            let ptr = tensor.storage.allocation.as_mut_ptr().add(start);
            std::slice::from_raw_parts(ptr, len).to_vec()
        };
        self.bump_downloads();
        Ok(out)
    }

    fn gemm(
        &self,
        a: &Tensor<Self::Store>,
        b: &Tensor<Self::Store>,
        c: &mut Tensor<Self::Store>,
        alpha: f32,
        beta: f32,
    ) -> Result<(), Error> {
        let (m, k) = matrix_dims(a)?;
        let (k_b, n) = matrix_dims(b)?;
        let (m_c, n_c) = matrix_dims(c)?;
        if k != k_b || m != m_c || n != n_c {
            return Err(Error::Shape);
        }
        if a.dtype != b.dtype || a.dtype != c.dtype {
            return Err(Error::Dtype);
        }
        let launch = match a.dtype {
            DType::F32 => "gemm-f32",
            DType::F16 => "gemm-f16",
            DType::BF16 => "gemm-bf16",
        };
        let a_id = a.storage.storage_id();
        let b_id = b.storage.storage_id();
        let c_id = c.storage.storage_id();
        if a_id == b_id || a_id == c_id || b_id == c_id {
            return Err(Error::Alias);
        }
        let a_layout = matrix_layout(a)?;
        let b_layout = matrix_layout(b)?;
        let c_layout = matrix_layout(c)?;
        let supported = matches!(
            (a_layout, b_layout, c_layout),
            (MatrixLayout::RowMajor, MatrixLayout::RowMajor, MatrixLayout::RowMajor)
                | (MatrixLayout::Transposed, MatrixLayout::RowMajor, MatrixLayout::RowMajor)
                | (MatrixLayout::RowMajor, MatrixLayout::Transposed, MatrixLayout::RowMajor)
        );
        if !supported {
            return Err(Error::UnsupportedLayout);
        }
        let size = a.dtype.size_of();
        if m > 0 && n > 0 {
            require_range(c, m, n, size)?;
            if alpha != 0.0 && k > 0 {
                require_range(a, m, k, size)?;
                require_range(b, k, n, size)?;
            }
        }

        let a_ptr = a.storage.allocation.as_mut_ptr();
        let b_ptr = b.storage.allocation.as_mut_ptr();
        let c_ptr = c.storage.allocation.as_mut_ptr();
        let a_index = indexer(a);
        let b_index = indexer(b);
        let c_index = indexer(c);
        match a.dtype {
            DType::F32 => accumulate(
                m,
                n,
                k,
                alpha,
                beta,
                |row, col| unsafe { load_f32(a_ptr, a_index(row, col)) },
                |row, col| unsafe { load_f32(b_ptr, b_index(row, col)) },
                |row, col| unsafe { load_f32(c_ptr, c_index(row, col)) },
                |row, col, value| unsafe { store_f32(c_ptr, c_index(row, col), value) },
            ),
            DType::F16 => accumulate(
                m,
                n,
                k,
                alpha,
                beta,
                |row, col| unsafe { load_f16(a_ptr, a_index(row, col)) },
                |row, col| unsafe { load_f16(b_ptr, b_index(row, col)) },
                |row, col| unsafe { load_f16(c_ptr, c_index(row, col)) },
                |row, col, value| unsafe { store_f16(c_ptr, c_index(row, col), value) },
            ),
            DType::BF16 => accumulate(
                m,
                n,
                k,
                alpha,
                beta,
                |row, col| unsafe { load_bf16(a_ptr, a_index(row, col)) },
                |row, col| unsafe { load_bf16(b_ptr, b_index(row, col)) },
                |row, col| unsafe { load_bf16(c_ptr, c_index(row, col)) },
                |row, col, value| unsafe { store_bf16(c_ptr, c_index(row, col), value) },
            ),
        }
        self.record(launch);
        Ok(())
    }

    fn gelu(&self, x: &Tensor<Self::Store>, y: &mut Tensor<Self::Store>) -> Result<(), Error> {
        let n = prepare_f32_row_major(&[x, y])?;
        let x_ptr = x.storage.allocation.as_mut_ptr();
        let y_ptr = y.storage.allocation.as_mut_ptr();
        for index in 0..n {
            let value = unsafe { load_f32(x_ptr, index) };
            unsafe { store_f32(y_ptr, index, gelu_f32(value)) };
        }
        self.record("gelu");
        Ok(())
    }

    fn gelu_backward(
        &self,
        pre: &Tensor<Self::Store>,
        dy: &Tensor<Self::Store>,
        dx: &mut Tensor<Self::Store>,
    ) -> Result<(), Error> {
        let n = prepare_f32_row_major(&[pre, dy, dx])?;
        let pre_ptr = pre.storage.allocation.as_mut_ptr();
        let dy_ptr = dy.storage.allocation.as_mut_ptr();
        let dx_ptr = dx.storage.allocation.as_mut_ptr();
        for index in 0..n {
            let pre_value = unsafe { load_f32(pre_ptr, index) };
            let dy_value = unsafe { load_f32(dy_ptr, index) };
            unsafe { store_f32(dx_ptr, index, dy_value * gelu_prime_f32(pre_value)) };
        }
        self.record("gelu-backward");
        Ok(())
    }

    fn debug_allocs(&self) -> u64 {
        #[cfg(test)]
        {
            self.stats.lock().expect("cpu stats").allocs
        }
        #[cfg(not(test))]
        {
            let _ = self;
            0
        }
    }

    fn debug_launches(&self) -> Vec<&'static str> {
        #[cfg(test)]
        {
            self.stats.lock().expect("cpu stats").launches.clone()
        }
        #[cfg(not(test))]
        {
            let _ = self;
            Vec::new()
        }
    }

    fn debug_downloads(&self) -> u64 {
        #[cfg(test)]
        {
            self.stats.lock().expect("cpu stats").downloads
        }
        #[cfg(not(test))]
        {
            let _ = self;
            0
        }
    }

    fn debug_uploads(&self) -> u64 {
        #[cfg(test)]
        {
            self.stats.lock().expect("cpu stats").uploads
        }
        #[cfg(not(test))]
        {
            let _ = self;
            0
        }
    }
}

fn numel(shape: &[usize]) -> Result<usize, Error> {
    let mut count = 1usize;
    for &extent in shape {
        count = count.checked_mul(extent).ok_or(Error::Shape)?;
    }
    Ok(count)
}

fn nbytes(dtype: DType, shape: &[usize]) -> Result<usize, Error> {
    numel(shape)?.checked_mul(dtype.size_of()).ok_or(Error::Shape)
}

fn owned_tensor(dtype: DType, shape: &[usize], bytes: Vec<u8>) -> Tensor<CpuStore> {
    Tensor {
        storage: CpuStore { allocation: Arc::new(Allocation::new(bytes)) },
        dtype,
        shape: shape.to_vec(),
        strides: row_major_strides(shape),
        offset: 0,
    }
}

fn matrix_dims(tensor: &Tensor<CpuStore>) -> Result<(usize, usize), Error> {
    if tensor.shape.len() != 2 || tensor.strides.len() != 2 {
        return Err(Error::Shape);
    }
    Ok((tensor.shape[0], tensor.shape[1]))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MatrixLayout {
    RowMajor,
    Transposed,
}

fn matrix_layout(tensor: &Tensor<CpuStore>) -> Result<MatrixLayout, Error> {
    if tensor.offset != 0 {
        return Err(Error::UnsupportedLayout);
    }
    if tensor.strides == row_major_strides(&tensor.shape) {
        return Ok(MatrixLayout::RowMajor);
    }
    // `transpose_matrix` of row-major `[cols, rows]` yields strides `[1, rows]`.
    if tensor.strides[0] == 1 && tensor.strides[1] == tensor.shape[0] {
        return Ok(MatrixLayout::Transposed);
    }
    Err(Error::UnsupportedLayout)
}

fn corner(rows: usize, cols: usize, strides: &[usize], offset: usize) -> Result<usize, Error> {
    if rows == 0 || cols == 0 {
        return Ok(0);
    }
    let row_term = (rows - 1).checked_mul(strides[0]).ok_or(Error::Shape)?;
    let col_term = (cols - 1).checked_mul(strides[1]).ok_or(Error::Shape)?;
    offset.checked_add(row_term).and_then(|sum| sum.checked_add(col_term)).ok_or(Error::Shape)
}

fn require_range(
    tensor: &Tensor<CpuStore>,
    rows: usize,
    cols: usize,
    size: usize,
) -> Result<(), Error> {
    let last = corner(rows, cols, &tensor.strides, tensor.offset)?;
    let end = last.checked_add(1).ok_or(Error::Shape)?.checked_mul(size).ok_or(Error::Shape)?;
    if end > tensor.storage.allocation.len() {
        return Err(Error::Shape);
    }
    Ok(())
}

fn indexer(tensor: &Tensor<CpuStore>) -> impl Fn(usize, usize) -> usize {
    let stride_row = tensor.strides[0];
    let stride_col = tensor.strides[1];
    let offset = tensor.offset;
    move |row, col| offset + row * stride_row + col * stride_col
}

#[allow(clippy::too_many_arguments)]
fn accumulate<A, B, C, W>(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
    mut load_a: A,
    mut load_b: B,
    mut load_c: C,
    mut store_c: W,
) where
    A: FnMut(usize, usize) -> f32,
    B: FnMut(usize, usize) -> f32,
    C: FnMut(usize, usize) -> f32,
    W: FnMut(usize, usize, f32),
{
    for row in 0..m {
        for col in 0..n {
            let mut acc = if beta == 0.0 { 0.0 } else { beta * load_c(row, col) };
            if alpha != 0.0 {
                for inner in 0..k {
                    acc += alpha * load_a(row, inner) * load_b(inner, col);
                }
            }
            store_c(row, col, acc);
        }
    }
}

unsafe fn load_f32(ptr: *const u8, index: usize) -> f32 {
    ptr.add(index * 4).cast::<f32>().read_unaligned()
}

unsafe fn store_f32(ptr: *mut u8, index: usize, value: f32) {
    ptr.add(index * 4).cast::<f32>().write_unaligned(value);
}

// `*_const` widens exactly and rounds ties to even. The hardware entry points
// take an atomic feature-detect load, which the math path does not.
unsafe fn load_f16(ptr: *const u8, index: usize) -> f32 {
    let bits = ptr.add(index * 2).cast::<u16>().read_unaligned();
    f16::from_bits(bits).to_f32_const()
}

unsafe fn store_f16(ptr: *mut u8, index: usize, value: f32) {
    let bits = f16::from_f32_const(value).to_bits();
    ptr.add(index * 2).cast::<u16>().write_unaligned(bits);
}

unsafe fn load_bf16(ptr: *const u8, index: usize) -> f32 {
    let bits = ptr.add(index * 2).cast::<u16>().read_unaligned();
    bf16::from_bits(bits).to_f32_const()
}

unsafe fn store_bf16(ptr: *mut u8, index: usize, value: f32) {
    let bits = bf16::from_f32_const(value).to_bits();
    ptr.add(index * 2).cast::<u16>().write_unaligned(bits);
}

fn prepare_f32_row_major(tensors: &[&Tensor<CpuStore>]) -> Result<usize, Error> {
    for tensor in tensors {
        if tensor.dtype != DType::F32 {
            return Err(Error::Dtype);
        }
    }
    let shape = &tensors[0].shape;
    for tensor in tensors.iter().skip(1) {
        if tensor.shape != *shape {
            return Err(Error::Shape);
        }
    }
    let packed = row_major_strides(shape);
    for tensor in tensors {
        if tensor.offset != 0 || tensor.strides != packed {
            return Err(Error::UnsupportedLayout);
        }
    }
    let count = numel(shape)?;
    let bytes = count.checked_mul(DType::F32.size_of()).ok_or(Error::Shape)?;
    for tensor in tensors {
        if tensor.storage.allocation.len() < bytes {
            return Err(Error::Shape);
        }
    }
    Ok(count)
}

/// Abramowitz and Stegun 7.1.26. Evaluated in `f32`, absolute error under 1e-6
/// on `gelu` across the tested domain.
fn erf_f32(x: f32) -> f32 {
    let sign = if x.is_sign_negative() { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = (((((ERF_A5 * t - ERF_A4) * t) + ERF_A3) * t - ERF_A2) * t + ERF_A1) * t;
    sign * (1.0 - poly * (-x * x).exp())
}

fn gelu_f32(x: f32) -> f32 {
    let inner = x * std::f32::consts::FRAC_1_SQRT_2;
    0.5 * x * (1.0 + erf_f32(inner))
}

fn gelu_prime_f32(x: f32) -> f32 {
    let phi = 0.5 * (1.0 + erf_f32(x * std::f32::consts::FRAC_1_SQRT_2));
    let pdf = (-0.5 * x * x).exp() * INV_SQRT_2PI;
    phi + x * pdf
}

// `f32` spellings of 0.254829592, 0.284496736, 1.421413741, 1.453152027,
// 1.061405429, and 0.3989422804014327. Assertions pin those `to_bits` values.
const ERF_A1: f32 = 0.254_829_6;
const ERF_A2: f32 = 0.284_496_72;
const ERF_A3: f32 = 1.421_413_8;
const ERF_A4: f32 = 1.453_152_1;
const ERF_A5: f32 = 1.061_405_4;
const INV_SQRT_2PI: f32 = 0.398_942_3;

/// `f32::to_bits` is not `const` on the 1.82 MSRV.
#[allow(unnecessary_transmutes)]
const fn f32_bits(value: f32) -> u32 {
    // SAFETY: `f32` and `u32` have the same size and alignment.
    unsafe { std::mem::transmute::<f32, u32>(value) }
}

const _: () = {
    assert!(f32_bits(ERF_A1) == 0x3e82_7906);
    assert!(f32_bits(ERF_A2) == 0x3e91_a98e);
    assert!(f32_bits(ERF_A3) == 0x3fb5_f0e3);
    assert!(f32_bits(ERF_A4) == 0x3fba_00e3);
    assert!(f32_bits(ERF_A5) == 0x3f87_dc22);
    assert!(f32_bits(INV_SQRT_2PI) == 0x3ecc_422a);
};

#[cfg(test)]
mod invariants;
