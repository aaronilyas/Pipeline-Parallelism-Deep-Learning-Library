//! Locks the zero-copy additions to the core contract: in-place transfers, `sync`, the five
//! training ops, scatter/reuse transport calls, and the split wire header.

use std::cell::RefCell;
use std::time::Duration;

use linalg_core::{
    decode_wire, encode_wire, parse_wire_header, write_wire_header, Backend, DType, DeviceKind,
    Error, Identified, Tensor, Transport, WIRE_HEADER_MAX,
};

// ------------------------------------------------------------------ wire header

/// `(dtype, shape, payload)` cases spanning every dtype, rank 0, rank 8, and zero extents.
fn wire_cases() -> Vec<(DType, Vec<usize>, Vec<u8>)> {
    vec![
        (DType::F32, vec![1], vec![0x11, 0x22, 0x33, 0x44]),
        (DType::F32, vec![], vec![1, 2, 3, 4]),
        (DType::F32, vec![2, 3], (0..24).collect()),
        (DType::F32, vec![0, 5], vec![]),
        (DType::F16, vec![2, 1], vec![0xAB, 0xCD, 0x11, 0x22]),
        (DType::F16, vec![2, 1, 1, 1, 1, 1, 1, 1], vec![9, 8, 7, 6]),
        (DType::F16, vec![2, 1, 2], vec![0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]),
        (DType::BF16, vec![], vec![0xEE, 0xFF]),
        (DType::BF16, vec![3], vec![1, 2, 3, 4, 5, 6]),
        (DType::BF16, vec![4, 0, 7], vec![]),
    ]
}

#[test]
fn header_max_fits_rank_8() {
    assert_eq!(WIRE_HEADER_MAX, 48);
    assert_eq!(WIRE_HEADER_MAX, 16 + 4 * 8);
}

#[test]
fn written_header_plus_payload_equals_encode_wire_for_every_dtype() {
    for (dtype, shape, payload) in wire_cases() {
        let mut header = [0xA5u8; WIRE_HEADER_MAX];
        let len = write_wire_header(dtype, &shape, &mut header);
        assert_eq!(len, Ok(16 + 4 * shape.len()), "{dtype:?} {shape:?}");
        let len = len.unwrap();
        let mut joined = header[..len].to_vec();
        joined.extend_from_slice(&payload);
        assert_eq!(Ok(joined), encode_wire(dtype, &shape, &payload), "{dtype:?} {shape:?}");
        assert!(header[len..].iter().all(|&b| b == 0xA5), "bytes past the header untouched");
    }
}

#[test]
fn header_round_trips_through_parse() {
    for (dtype, shape, payload) in wire_cases() {
        let message = encode_wire(dtype, &shape, &payload).unwrap();
        let parsed = parse_wire_header(&message);
        assert_eq!(parsed, Ok((dtype, shape.clone(), 16 + 4 * shape.len())));
        let (_, _, offset) = parsed.unwrap();
        assert_eq!(&message[offset..], &payload[..]);

        let mut header = [0u8; WIRE_HEADER_MAX];
        let len = write_wire_header(dtype, &shape, &mut header).unwrap();
        let mut joined = header[..len].to_vec();
        joined.extend_from_slice(&payload);
        assert_eq!(parse_wire_header(&joined), Ok((dtype, shape.clone(), len)));
        assert_eq!(decode_wire(&joined), Ok((dtype, shape, payload)));
    }
}

#[test]
fn write_header_rejects_what_encode_rejects_for_the_header() {
    let mut out = [0u8; WIRE_HEADER_MAX];
    let above = u32::MAX as usize + 1;
    let cases: Vec<(DType, Vec<usize>, Error)> = vec![
        (DType::F32, vec![1; 9], Error::Wire),
        (DType::F32, vec![1; 12], Error::Wire),
        (DType::F32, vec![above], Error::Shape),
        (DType::F16, vec![above + 5], Error::Shape),
        (DType::BF16, vec![1, above + 5, 2], Error::Shape),
        (DType::F32, vec![1 << 16, 1 << 16, 1 << 16, 1 << 16], Error::Shape),
        (DType::F32, vec![1 << 31, 1 << 31], Error::Shape),
        (DType::F16, vec![1 << 31, 1 << 31, 2], Error::Shape),
        (DType::BF16, vec![1 << 31, 1 << 31, 2], Error::Shape),
    ];
    for (dtype, shape, want) in cases {
        out.fill(0x5A);
        assert_eq!(write_wire_header(dtype, &shape, &mut out), Err(want), "{dtype:?} {shape:?}");
        assert_eq!(encode_wire(dtype, &shape, &[]), Err(want), "{dtype:?} {shape:?}");
        assert!(out.iter().all(|&b| b == 0x5A), "rejected header leaves out unchanged");
    }
    // A payload mismatch is the one rejection the header alone cannot see.
    let len = write_wire_header(DType::F32, &[u32::MAX as usize], &mut out);
    assert_eq!(len, Ok(20));
    assert_eq!(encode_wire(DType::F32, &[u32::MAX as usize], &[]), Err(Error::Wire));
}

fn message(dtype_tag: u32, extents: &[u32], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"LNGL");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&dtype_tag.to_le_bytes());
    out.extend_from_slice(&(extents.len() as u32).to_le_bytes());
    for extent in extents {
        out.extend_from_slice(&extent.to_le_bytes());
    }
    out.extend_from_slice(payload);
    out
}

#[test]
fn parse_rejects_everything_decode_rejects_with_the_same_error() {
    let good = message(1, &[1], &[0x11, 0x22, 0x33, 0x44]);
    let mut bad = Vec::new();

    let mut m = good.clone();
    m[0] = b'X';
    bad.push(("bad magic", m, Error::Wire));
    for version in [0u32, 2] {
        let mut m = good.clone();
        m[4..8].copy_from_slice(&version.to_le_bytes());
        bad.push(("bad version", m, Error::Wire));
    }
    bad.push(("empty", vec![], Error::Wire));
    bad.push(("15 zero bytes", vec![0; 15], Error::Wire));
    bad.push(("header only", good[..16].to_vec(), Error::Wire));
    bad.push(("short payload", good[..good.len() - 1].to_vec(), Error::Wire));
    let mut m = good.clone();
    m.push(0xFF);
    bad.push(("trailing byte", m, Error::Wire));
    bad.push(("ndim 9", message(1, &[1; 9], &[0, 0, 0, 0]), Error::Wire));
    bad.push((
        "ndim u32::MAX",
        {
            let mut m = good.clone();
            m[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
            m
        },
        Error::Wire,
    ));
    bad.push(("dtype 0", message(0, &[1], &[0, 0, 0, 0]), Error::Wire));
    bad.push(("dtype 4", message(4, &[1], &[0, 0, 0, 0]), Error::Wire));
    let mut m = message(1, &[1, 1], &[0, 0, 0, 0]);
    m.truncate(18);
    bad.push(("mid extent", m, Error::Wire));
    bad.push(("numel overflow", message(1, &[65536; 4], &[]), Error::Shape));
    bad.push(("numel overflow 2", message(1, &[1 << 31, 1 << 31], &[]), Error::Shape));
    bad.push(("bytes overflow", message(2, &[1 << 31, 1 << 31, 2], &[]), Error::Shape));
    bad.push(("bf16 bytes overflow", message(3, &[1 << 31, 1 << 31, 2], &[]), Error::Shape));

    for (name, bytes, want) in bad {
        assert_eq!(decode_wire(&bytes).err(), Some(want), "decode: {name}");
        assert_eq!(parse_wire_header(&bytes).err(), Some(want), "parse: {name}");
    }
}

// ------------------------------------------------------------------ backend surface

#[derive(Clone)]
struct Id(u64);

impl Identified for Id {
    fn storage_id(&self) -> u64 {
        self.0
    }
}

/// Implements every required method and records which were called. It does no math: the
/// point is that the trait has exactly these signatures, and that `sync` defaults to `Ok`.
#[derive(Default)]
struct Surface {
    calls: RefCell<Vec<&'static str>>,
}

// `Backend: Sync` is satisfied for the test by asserting single-threaded use.
unsafe impl Sync for Surface {}

impl Surface {
    fn call(&self, name: &'static str) -> Result<(), Error> {
        self.calls.borrow_mut().push(name);
        Ok(())
    }
}

fn tensor(id: u64, dtype: DType, shape: &[usize]) -> Tensor<Id> {
    Tensor {
        storage: Id(id),
        dtype,
        shape: shape.to_vec(),
        strides: linalg_core::row_major_strides(shape),
        offset: 0,
    }
}

impl Backend for Surface {
    type Store = Id;

    fn kind(&self) -> DeviceKind {
        DeviceKind::Cpu
    }

    fn alloc(&self, dtype: DType, shape: &[usize]) -> Result<Tensor<Id>, Error> {
        Ok(tensor(0, dtype, shape))
    }

    fn upload(
        &self,
        dtype: DType,
        shape: &[usize],
        _row_major: &[u8],
    ) -> Result<Tensor<Id>, Error> {
        Ok(tensor(0, dtype, shape))
    }

    fn download(&self, _tensor: &Tensor<Id>) -> Result<Vec<u8>, Error> {
        Ok(Vec::new())
    }

    fn download_into(&self, _tensor: &Tensor<Id>, out: &mut [u8]) -> Result<(), Error> {
        out.fill(7);
        self.call("download_into")
    }

    fn upload_into(&self, _bytes: &[u8], _tensor: &mut Tensor<Id>) -> Result<(), Error> {
        self.call("upload_into")
    }

    fn gemm(
        &self,
        _a: &Tensor<Id>,
        _b: &Tensor<Id>,
        _c: &mut Tensor<Id>,
        _alpha: f32,
        _beta: f32,
    ) -> Result<(), Error> {
        self.call("gemm")
    }

    fn gelu(&self, _x: &Tensor<Id>, _y: &mut Tensor<Id>) -> Result<(), Error> {
        self.call("gelu")
    }

    fn gelu_backward(
        &self,
        _pre: &Tensor<Id>,
        _dy: &Tensor<Id>,
        _dx: &mut Tensor<Id>,
    ) -> Result<(), Error> {
        self.call("gelu_backward")
    }

    fn add_bias(&self, _x: &Tensor<Id>, _b: &Tensor<Id>, _y: &mut Tensor<Id>) -> Result<(), Error> {
        self.call("add_bias")
    }

    fn bias_grad(&self, _dpre: &Tensor<Id>, _db: &mut Tensor<Id>) -> Result<(), Error> {
        self.call("bias_grad")
    }

    fn mse_grad(
        &self,
        _y: &Tensor<Id>,
        _target: &Tensor<Id>,
        _dy: &mut Tensor<Id>,
        loss: &mut f32,
    ) -> Result<(), Error> {
        *loss = 0.25;
        self.call("mse_grad")
    }

    fn sgd_step(&self, _w: &mut Tensor<Id>, _dw: &mut Tensor<Id>, _lr: f32) -> Result<(), Error> {
        self.call("sgd_step")
    }

    fn convert(&self, _x: &Tensor<Id>, _y: &mut Tensor<Id>) -> Result<(), Error> {
        self.call("convert")
    }

    fn debug_allocs(&self) -> u64 {
        0
    }

    fn debug_launches(&self) -> Vec<&'static str> {
        Vec::new()
    }

    fn debug_downloads(&self) -> u64 {
        0
    }

    fn debug_uploads(&self) -> u64 {
        0
    }
}

/// Drives every addition through a generic `B: Backend`, as the pipeline will.
fn one_step<B: Backend>(be: &B, loss: &mut f32) -> Result<Vec<u8>, Error> {
    let x = be.alloc(DType::F32, &[2, 3])?;
    let b = be.alloc(DType::F32, &[3])?;
    let mut w = be.alloc(DType::F32, &[3, 3])?;
    let mut dw = be.alloc(DType::F32, &[3, 3])?;
    let mut pre = be.alloc(DType::F32, &[2, 3])?;
    let mut db = be.alloc(DType::F32, &[3])?;
    let mut dy = be.alloc(DType::F32, &[2, 3])?;
    let mut half = be.alloc(DType::BF16, &[2, 3])?;
    be.upload_into(&[0u8; 24], &mut pre)?;
    be.add_bias(&x, &b, &mut pre)?;
    be.mse_grad(&pre, &x, &mut dy, loss)?;
    be.bias_grad(&dy, &mut db)?;
    be.sgd_step(&mut w, &mut dw, 1e-3)?;
    be.convert(&pre, &mut half)?;
    be.sync()?;
    let mut out = vec![0u8; 24];
    be.download_into(&pre, &mut out)?;
    Ok(out)
}

#[test]
fn backend_has_the_additions_and_sync_defaults_to_ok() {
    let be = Surface::default();
    assert_eq!(be.sync(), Ok(()));
    let mut loss = f32::NAN;
    assert_eq!(one_step(&be, &mut loss), Ok(vec![7; 24]));
    assert_eq!(loss, 0.25);
    assert_eq!(
        *be.calls.borrow(),
        vec![
            "upload_into",
            "add_bias",
            "mse_grad",
            "bias_grad",
            "sgd_step",
            "convert",
            "download_into"
        ]
    );
}

// ------------------------------------------------------------------ transport surface

/// Two-party pipe in one process. `send_parts` writes each part straight into the queued
/// message; `recv_into` moves bytes into the caller's buffer.
#[derive(Default)]
struct Pipe {
    queue: RefCell<Vec<Vec<u8>>>,
}

impl Transport for Pipe {
    fn rank(&self) -> usize {
        0
    }

    fn world(&self) -> usize {
        2
    }

    fn send(&self, dest: usize, bytes: &[u8]) -> Result<(), Error> {
        self.send_parts(dest, &[bytes])
    }

    fn recv_timeout(&self, src: usize, timeout: Duration) -> Result<Vec<u8>, Error> {
        let mut buf = Vec::new();
        self.recv_into(src, &mut buf, timeout)?;
        Ok(buf)
    }

    fn send_parts(&self, dest: usize, parts: &[&[u8]]) -> Result<(), Error> {
        if dest != 1 {
            return Err(Error::Transport);
        }
        let mut message = Vec::with_capacity(parts.iter().map(|p| p.len()).sum());
        for part in parts {
            message.extend_from_slice(part);
        }
        self.queue.borrow_mut().push(message);
        Ok(())
    }

    fn recv_into(&self, src: usize, buf: &mut Vec<u8>, _timeout: Duration) -> Result<(), Error> {
        if src != 1 || self.queue.borrow().is_empty() {
            return Err(Error::Transport);
        }
        let message = self.queue.borrow_mut().remove(0);
        buf.clear();
        buf.extend_from_slice(&message);
        Ok(())
    }
}

#[test]
fn transport_has_send_parts_and_recv_into() {
    let pipe = Pipe::default();
    let t: &dyn Transport = &pipe;
    let mut header = [0u8; WIRE_HEADER_MAX];
    let payload = [1u8, 2, 3, 4, 5, 6, 7, 8];
    let len = write_wire_header(DType::F32, &[2], &mut header).unwrap();
    assert_eq!(t.send_parts(1, &[&header[..len], &payload]), Ok(()));
    assert_eq!(t.send_parts(1, &[]), Ok(()));

    let mut buf = Vec::with_capacity(64);
    let capacity = buf.capacity();
    let ptr = buf.as_ptr();
    assert_eq!(t.recv_into(1, &mut buf, Duration::from_millis(1)), Ok(()));
    assert_eq!(buf.as_ptr(), ptr);
    assert_eq!(buf.capacity(), capacity);
    assert_eq!(Ok(buf.clone()), encode_wire(DType::F32, &[2], &payload));
    assert_eq!(parse_wire_header(&buf), Ok((DType::F32, vec![2], len)));

    assert_eq!(t.recv_into(1, &mut buf, Duration::from_millis(1)), Ok(()));
    assert!(buf.is_empty());
    assert_eq!(t.recv_into(1, &mut buf, Duration::from_millis(1)), Err(Error::Transport));
}
