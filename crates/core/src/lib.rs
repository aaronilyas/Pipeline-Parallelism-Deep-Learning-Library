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
///
/// # Ordering
///
/// Launch order is execution order: every call takes effect after every call made before it on
/// the same backend and before every call made after it. [`gemm`](Backend::gemm),
/// [`gelu`](Backend::gelu), [`gelu_backward`](Backend::gelu_backward),
/// [`add_bias`](Backend::add_bias), [`bias_grad`](Backend::bias_grad),
/// [`sgd_step`](Backend::sgd_step), and [`convert`](Backend::convert) may return before the
/// device finishes them. [`download`](Backend::download),
/// [`download_into`](Backend::download_into), [`sync`](Backend::sync), and
/// [`mse_grad`](Backend::mse_grad) return only after every earlier call has completed, so the
/// bytes or loss they hand back are final. A call that returns an error launches nothing.
///
/// # Training ops
///
/// [`add_bias`](Backend::add_bias), [`bias_grad`](Backend::bias_grad),
/// [`mse_grad`](Backend::mse_grad), [`sgd_step`](Backend::sgd_step), and
/// [`convert`](Backend::convert) are each exactly one launch and append exactly one entry to
/// [`debug_launches`](Backend::debug_launches). Each checks its operands in this order and
/// returns the first failure, leaving every output unchanged:
///
/// 1. [`Error::Shape`]: extents that disagree with the op's stated shapes.
/// 2. [`Error::Dtype`]: any operand not of the op's stated dtype.
/// 3. [`Error::Alias`]: an output that shares a [`Identified::storage_id`] with another operand,
///    except where the op allows it.
/// 4. [`Error::UnsupportedLayout`]: any operand that is not row-major contiguous with offset 0.
///
/// Arithmetic is IEEE-754 `f32` with round-to-nearest-even and no flush of subnormals. Each
/// written formula is evaluated as written, one rounding per operation, with no fused
/// multiply-add. Where an op sums, the order of summation is the backend's choice, but it is
/// fixed: the same inputs give the same bits on the same backend. This crate states the math and
/// does not evaluate it.
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

    /// Writes `tensor`'s elements into `out` as row-major bytes, the same bytes
    /// [`download`](Backend::download) returns, without allocating a host buffer.
    ///
    /// `out.len()` must equal the tensor's element count times its dtype size; otherwise the
    /// result is [`Error::Shape`] and `out` is unchanged. Returns after every earlier call has
    /// completed.
    fn download_into(&self, tensor: &Tensor<Self::Store>, out: &mut [u8]) -> Result<(), Error>;

    /// Overwrites every element of the existing `tensor` with row-major `bytes`. Allocates
    /// nothing: no tensor, no device buffer, no host buffer.
    ///
    /// `bytes.len()` must equal the tensor's element count times its dtype size; otherwise the
    /// result is [`Error::Shape`] and `tensor` is unchanged. A tensor that is not row-major
    /// contiguous is [`Error::UnsupportedLayout`]. The write takes effect after every earlier
    /// call and before every later one; `bytes` may be reused as soon as this returns.
    fn upload_into(&self, bytes: &[u8], tensor: &mut Tensor<Self::Store>) -> Result<(), Error>;

    /// Returns after every earlier call has completed on the device.
    ///
    /// The default suits a backend whose every call completes before it returns.
    fn sync(&self) -> Result<(), Error> {
        Ok(())
    }

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

    /// `x` is `[B, N]`, `b` is `[N]`, and `y` is `[B, N]`, all `f32`.
    ///
    /// Writes `y[i][j] = x[i][j] + b[j]`. `y` may be `x` itself (the same storage, so the add is
    /// in place); `b` has a storage distinct from `x` and `y`.
    fn add_bias(
        &self,
        x: &Tensor<Self::Store>,
        b: &Tensor<Self::Store>,
        y: &mut Tensor<Self::Store>,
    ) -> Result<(), Error>;

    /// `dpre` is `[B, N]` and `db` is `[N]`, both `f32`, with distinct storage.
    ///
    /// Accumulates the column sums of `dpre`: `db[j] = db[j] + s[j]`, where `s[j]` is the sum
    /// over `i` of `dpre[i][j]`. When `B` is 0, `db` is unchanged.
    fn bias_grad(
        &self,
        dpre: &Tensor<Self::Store>,
        db: &mut Tensor<Self::Store>,
    ) -> Result<(), Error>;

    /// `y`, `target`, and `dy` have one shape, all `f32`, with `n` elements; `n` of 0 is
    /// [`Error::Shape`].
    ///
    /// With `d = y - target` elementwise and `n` rounded to `f32`, writes `dy = (2 * d) / n`
    /// elementwise and sets `loss` to the mean squared error, `(sum of d * d) / n`. `dy` may be
    /// `y` itself (in place); `dy` and `target` have distinct storage. `y` and `target` may
    /// share storage. Returns after every earlier call and this one have completed, so `loss`
    /// is final; on error `loss` is unchanged.
    fn mse_grad(
        &self,
        y: &Tensor<Self::Store>,
        target: &Tensor<Self::Store>,
        dy: &mut Tensor<Self::Store>,
        loss: &mut f32,
    ) -> Result<(), Error>;

    /// `w` and `dw` have one shape, both `f32`, with distinct storage.
    ///
    /// Writes `w = w - (lr * dw)` elementwise, then `dw = +0.0` elementwise.
    fn sgd_step(
        &self,
        w: &mut Tensor<Self::Store>,
        dw: &mut Tensor<Self::Store>,
        lr: f32,
    ) -> Result<(), Error>;

    /// `x` and `y` have one shape and distinct storage. `(x, y)` dtypes are `(F32, BF16)` or
    /// `(BF16, F32)`; any other pair is [`Error::Dtype`].
    ///
    /// `F32` to `BF16` gives the `bf16` value nearest `x`, ties to even: finite values too large
    /// for `bf16` become infinities of the same sign, subnormals are kept, and NaN becomes a
    /// quiet NaN of the same sign. `BF16` to `F32` is exact: the `bf16` bits become the upper
    /// 16 bits of the `f32` and the lower 16 bits are zero.
    fn convert(&self, x: &Tensor<Self::Store>, y: &mut Tensor<Self::Store>) -> Result<(), Error>;

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

    /// Sends one message to `dest` whose bytes are the concatenation of `parts`, in order,
    /// without building the concatenation. The receiver cannot tell it from
    /// [`send`](Transport::send) of the concatenated bytes. Empty `parts` sends an empty message.
    fn send_parts(&self, dest: usize, parts: &[&[u8]]) -> Result<(), Error>;

    /// Receives the next message from `src` into `buf`, as [`recv_timeout`] would return it:
    /// afterward `buf` holds exactly the message. `buf`'s allocation is reused, so nothing is
    /// allocated when its capacity already covers the message. On error the contents of `buf`
    /// are unspecified and its allocation is kept.
    ///
    /// [`recv_timeout`]: Transport::recv_timeout
    fn recv_into(
        &self,
        src: usize,
        buf: &mut Vec<u8>,
        timeout: std::time::Duration,
    ) -> Result<(), Error>;
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

const WIRE_MAX_NDIM: usize = 8;

/// The longest wire header: 16 fixed bytes plus 4 per extent at the maximum rank of 8.
pub const WIRE_HEADER_MAX: usize = 16 + 4 * WIRE_MAX_NDIM;

/// One allocation. `f16` and `bf16` bits are copied raw.
///
/// The bytes are the header [`write_wire_header`] writes followed by `row_major`.
pub fn encode_wire(dtype: DType, shape: &[usize], row_major: &[u8]) -> Result<Vec<u8>, Error> {
    let mut header = [0u8; WIRE_HEADER_MAX];
    let (len, nbytes) = write_header(dtype, shape, &mut header)?;
    if row_major.len() != nbytes {
        return Err(Error::Wire);
    }
    let total = len.checked_add(nbytes).ok_or(Error::Shape)?;
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&header[..len]);
    out.extend_from_slice(row_major);
    Ok(out)
}

/// Writes the wire header for a `dtype` tensor of `shape` into `out` and returns its length,
/// `16 + 4 * shape.len()`. Followed by the row-major payload, these are the bytes
/// [`encode_wire`] produces. Allocates nothing.
///
/// More than 8 extents is [`Error::Wire`]; an extent above `u32::MAX`, or a byte count that
/// overflows `usize`, is [`Error::Shape`]. On error `out` is unchanged.
pub fn write_wire_header(
    dtype: DType,
    shape: &[usize],
    out: &mut [u8; WIRE_HEADER_MAX],
) -> Result<usize, Error> {
    write_header(dtype, shape, out).map(|(len, _)| len)
}

/// Returns `(header length, payload byte count)`.
fn write_header(
    dtype: DType,
    shape: &[usize],
    out: &mut [u8; WIRE_HEADER_MAX],
) -> Result<(usize, usize), Error> {
    if shape.len() > WIRE_MAX_NDIM {
        return Err(Error::Wire);
    }
    let nbytes = checked_nbytes(dtype, shape)?;
    out[0..4].copy_from_slice(&WIRE_MAGIC);
    out[4..8].copy_from_slice(&WIRE_VERSION.to_le_bytes());
    out[8..12].copy_from_slice(&dtype.tag().to_le_bytes());
    out[12..16].copy_from_slice(&(shape.len() as u32).to_le_bytes());
    for (slot, &extent) in out[16..].chunks_exact_mut(4).zip(shape) {
        slot.copy_from_slice(&(extent as u32).to_le_bytes());
    }
    Ok((16 + 4 * shape.len(), nbytes))
}

/// Reads a whole wire message's header and returns `(dtype, shape, payload_offset)`. The
/// payload is `&bytes[payload_offset..]`, and its length has been checked against the shape.
///
/// Rejects exactly what [`decode_wire`] rejects, with the same error. Allocates nothing on the
/// heap except the returned shape, and nothing at all on error.
pub fn parse_wire_header(bytes: &[u8]) -> Result<(DType, Vec<usize>, usize), Error> {
    if bytes.len() < 16 || bytes[0..4] != WIRE_MAGIC {
        return Err(Error::Wire);
    }
    if read_u32(bytes, 4)? != WIRE_VERSION {
        return Err(Error::Wire);
    }
    let dtype = DType::from_tag(read_u32(bytes, 8)?)?;
    let ndim_u32 = read_u32(bytes, 12)?;
    if ndim_u32 > WIRE_MAX_NDIM as u32 {
        return Err(Error::Wire);
    }
    let ndim = ndim_u32 as usize;
    let header = 16 + 4 * ndim;
    let extents = bytes.get(16..header).ok_or(Error::Wire)?;
    let mut numel = 1usize;
    for chunk in extents.chunks_exact(4) {
        numel = numel.checked_mul(read_u32(chunk, 0)? as usize).ok_or(Error::Shape)?;
    }
    let nbytes = numel.checked_mul(dtype.size_of()).ok_or(Error::Shape)?;
    if bytes.len() - header != nbytes {
        return Err(Error::Wire);
    }
    let mut shape = Vec::with_capacity(ndim);
    for chunk in extents.chunks_exact(4) {
        shape.push(read_u32(chunk, 0)? as usize);
    }
    Ok((dtype, shape, header))
}

pub fn decode_wire(bytes: &[u8]) -> Result<(DType, Vec<usize>, Vec<u8>), Error> {
    let (dtype, shape, offset) = parse_wire_header(bytes)?;
    Ok((dtype, shape, bytes[offset..].to_vec()))
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
