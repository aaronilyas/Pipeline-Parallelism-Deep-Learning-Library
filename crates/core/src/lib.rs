//! Device-independent tensor contract shared by every backend.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
}

impl DType {
    pub const fn size_of(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
        }
    }

    const fn tag(self) -> u32 {
        match self {
            DType::F32 => 1,
            DType::F16 => 2,
            DType::BF16 => 3,
        }
    }

    fn from_tag(tag: u32) -> Result<Self, Error> {
        match tag {
            1 => Ok(DType::F32),
            2 => Ok(DType::F16),
            3 => Ok(DType::BF16),
            _ => Err(Error::Wire),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Metal,
    Cuda,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Shape,
    Dtype,
    UnsupportedLayout,
    UnsupportedDevice,
    Alias,
    Transport,
    Wire,
    Schedule,
}

pub struct Tensor<S> {
    pub storage: S,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
}

pub trait Identified {
    fn storage_id(&self) -> u64;
}

/// One device's execution of the tensor contract.
///
/// No thread may read or write a `storage_id` while another call writes that
/// same `storage_id`.
pub trait Backend: Send + Sync {
    type Store: Clone + Send + Sync + Identified;

    fn kind(&self) -> DeviceKind;

    fn alloc(&self, dtype: DType, shape: &[usize]) -> Result<Tensor<Self::Store>, Error>;

    fn upload(
        &self,
        dtype: DType,
        shape: &[usize],
        row_major: &[u8],
    ) -> Result<Tensor<Self::Store>, Error>;

    fn download(&self, tensor: &Tensor<Self::Store>) -> Result<Vec<u8>, Error>;

    /// `a` is `[M, K]`, `b` is `[K, N]`, and `c` is `[M, N]`. `alpha` and `beta` are `f32`.
    ///
    /// The stored result is `beta * C + alpha * A * B` when `beta != 0`. When `beta` is 0, `c` is
    /// not read. When `alpha` is 0, `a` and `b` are not read. When `K` is 0 and `beta` is 0, `c`
    /// becomes zeros.
    ///
    /// Three layouts are supported. Every tensor has offset 0, and all three have the same dtype:
    /// all three row-major contiguous; `a` is a [`transpose_matrix`] of a row-major contiguous
    /// matrix and `b` and `c` are row-major contiguous; or `b` is a [`transpose_matrix`] of a
    /// row-major contiguous matrix and `a` and `c` are row-major contiguous.
    ///
    /// Any other layout returns [`Error::UnsupportedLayout`]. `a`, `b`, and `c` have distinct
    /// [`Identified::storage_id`] values. Otherwise the result is [`Error::Alias`] and `c` is
    /// unchanged.
    fn gemm(
        &self,
        a: &Tensor<Self::Store>,
        b: &Tensor<Self::Store>,
        c: &mut Tensor<Self::Store>,
        alpha: f32,
        beta: f32,
    ) -> Result<(), Error>;

    /// Writes `GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))` into `y`.
    ///
    /// Pipeline layers use `f32`. This crate states the definition and does not evaluate it.
    fn gelu(&self, x: &Tensor<Self::Store>, y: &mut Tensor<Self::Store>) -> Result<(), Error>;

    /// Writes `dy * GELU'(pre)` into `dx`, where `GELU'(x) = Φ(x) + x * φ(x)`, `Φ` is the
    /// standard normal cdf, and `φ` is the standard normal pdf.
    ///
    /// Pipeline layers use `f32`. This crate states the definition and does not evaluate it.
    fn gelu_backward(
        &self,
        pre: &Tensor<Self::Store>,
        dy: &Tensor<Self::Store>,
        dx: &mut Tensor<Self::Store>,
    ) -> Result<(), Error>;

    fn debug_allocs(&self) -> u64;

    fn debug_launches(&self) -> Vec<&'static str>;

    fn debug_downloads(&self) -> u64;

    fn debug_uploads(&self) -> u64;
}

pub trait Transport: Send {
    fn rank(&self) -> usize;

    fn world(&self) -> usize;

    fn send(&self, dest: usize, bytes: &[u8]) -> Result<(), Error>;

    fn recv_timeout(&self, src: usize, timeout: std::time::Duration) -> Result<Vec<u8>, Error>;
}

pub fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut stride = 1usize;
    for index in (0..shape.len()).rev() {
        strides[index] = stride;
        stride *= shape[index];
    }
    strides
}

pub fn transpose_matrix<S: Clone>(tensor: &Tensor<S>) -> Result<Tensor<S>, Error> {
    if tensor.shape.len() != 2 || tensor.strides.len() != 2 {
        return Err(Error::Shape);
    }
    Ok(Tensor {
        storage: tensor.storage.clone(),
        dtype: tensor.dtype,
        shape: vec![tensor.shape[1], tensor.shape[0]],
        strides: vec![tensor.strides[1], tensor.strides[0]],
        offset: tensor.offset,
    })
}

pub const WIRE_MAGIC: [u8; 4] = *b"LNGL";
pub const WIRE_VERSION: u32 = 1;

/// One allocation. `f16` and `bf16` bits are copied raw.
pub fn encode_wire(dtype: DType, shape: &[usize], row_major: &[u8]) -> Result<Vec<u8>, Error> {
    if shape.len() > 8 {
        return Err(Error::Wire);
    }
    let nbytes = checked_nbytes(dtype, shape)?;
    if row_major.len() != nbytes {
        return Err(Error::Wire);
    }
    let total = (16 + 4 * shape.len()).checked_add(nbytes).ok_or(Error::Shape)?;
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&WIRE_MAGIC);
    out.extend_from_slice(&WIRE_VERSION.to_le_bytes());
    out.extend_from_slice(&dtype.tag().to_le_bytes());
    out.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    for &extent in shape {
        out.extend_from_slice(&(extent as u32).to_le_bytes());
    }
    out.extend_from_slice(row_major);
    Ok(out)
}

pub fn decode_wire(bytes: &[u8]) -> Result<(DType, Vec<usize>, Vec<u8>), Error> {
    if bytes.len() < 16 || bytes[0..4] != WIRE_MAGIC {
        return Err(Error::Wire);
    }
    if read_u32(bytes, 4)? != WIRE_VERSION {
        return Err(Error::Wire);
    }
    let dtype = DType::from_tag(read_u32(bytes, 8)?)?;
    let ndim_u32 = read_u32(bytes, 12)?;
    if ndim_u32 > 8 {
        return Err(Error::Wire);
    }
    let ndim = ndim_u32 as usize;
    let header = 16 + 4 * ndim;
    if bytes.len() < header {
        return Err(Error::Wire);
    }
    let mut shape = Vec::with_capacity(ndim);
    let mut numel = 1usize;
    for chunk in bytes[16..header].chunks_exact(4) {
        let extent = read_u32(chunk, 0)? as usize;
        numel = numel.checked_mul(extent).ok_or(Error::Shape)?;
        shape.push(extent);
    }
    let nbytes = numel.checked_mul(dtype.size_of()).ok_or(Error::Shape)?;
    let payload = &bytes[header..];
    if payload.len() != nbytes {
        return Err(Error::Wire);
    }
    Ok((dtype, shape, payload.to_vec()))
}

fn checked_nbytes(dtype: DType, shape: &[usize]) -> Result<usize, Error> {
    let mut numel = 1usize;
    for &extent in shape {
        if extent > u32::MAX as usize {
            return Err(Error::Shape);
        }
        numel = numel.checked_mul(extent).ok_or(Error::Shape)?;
    }
    numel.checked_mul(dtype.size_of()).ok_or(Error::Shape)
}

fn read_u32(bytes: &[u8], at: usize) -> Result<u32, Error> {
    let chunk = bytes.get(at..at + 4).ok_or(Error::Wire)?;
    let mut buf = [0u8; 4];
    buf.copy_from_slice(chunk);
    Ok(u32::from_le_bytes(buf))
}
