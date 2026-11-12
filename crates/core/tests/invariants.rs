use linalg_core::{
    decode_wire, encode_wire, row_major_strides, transpose_matrix, DType, Error, Identified,
    Tensor, WIRE_MAGIC, WIRE_VERSION,
};

/// F32, shape `[1]`, payload `11 22 33 44`.
const F32_SCALAR: &[u8] = &[
    b'L', b'N', b'G', b'L', 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x11, 0x22, 0x33, 0x44,
];

/// F16, shape `[2, 1]`, payload `AB CD 11 22`.
const F16_MATRIX: &[u8] = &[
    b'L', b'N', b'G', b'L', 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
    0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xAB, 0xCD, 0x11, 0x22,
];

/// BF16, shape `[]`, payload `EE FF`. An empty shape has numel 1.
const BF16_SCALAR: &[u8] = &[
    b'L', b'N', b'G', b'L', 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xEE, 0xFF,
];

/// F32, shape `[0, 5]`, empty payload. A zero extent makes numel 0.
const F32_ZERO_EXTENT: &[u8] = &[
    b'L', b'N', b'G', b'L', 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
];

#[derive(Clone)]
struct ArcStore(std::sync::Arc<()>);

impl Identified for ArcStore {
    fn storage_id(&self) -> u64 {
        std::sync::Arc::as_ptr(&self.0) as u64
    }
}

fn rank_tensor(shape: Vec<usize>) -> Tensor<()> {
    Tensor { storage: (), dtype: DType::F32, shape, strides: Vec::new(), offset: 0 }
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
fn strides_of_row_major_matrix_and_empty_shape() {
    assert_eq!(row_major_strides(&[]), Vec::<usize>::new());
    assert_eq!(row_major_strides(&[3, 5]), vec![5, 1]);
    assert_eq!(row_major_strides(&[2, 3, 4]), vec![12, 4, 1]);
    assert_eq!(row_major_strides(&[7]), vec![1]);
    assert_eq!(row_major_strides(&[4, 0]), vec![0, 1]);
    assert_eq!(row_major_strides(&[0, 4]), vec![4, 1]);
}

#[test]
fn transpose_swaps_axes_and_keeps_storage_id() {
    let storage = ArcStore(std::sync::Arc::new(()));
    let id = storage.storage_id();
    let tensor =
        Tensor { storage, dtype: DType::BF16, shape: vec![2, 3], strides: vec![3, 1], offset: 4 };

    let got = transpose_matrix(&tensor);
    assert_eq!(got.as_ref().map(|out| out.shape.clone()), Ok(vec![3, 2]));
    assert_eq!(got.as_ref().map(|out| out.strides.clone()), Ok(vec![1, 3]));
    assert_eq!(got.as_ref().map(|out| out.dtype), Ok(DType::BF16));
    assert_eq!(got.as_ref().map(|out| out.offset), Ok(4));
    assert_eq!(got.as_ref().map(|out| out.storage.storage_id()), Ok(id));
    assert_eq!(tensor.storage.storage_id(), id);
    assert_eq!(tensor.shape, vec![2, 3]);
    assert_eq!(tensor.strides, vec![3, 1]);

    let zeros = Tensor {
        storage: ArcStore(std::sync::Arc::new(())),
        dtype: DType::F32,
        shape: vec![0, 4],
        strides: vec![8, 2],
        offset: 1,
    };
    let zero_id = zeros.storage.storage_id();
    let got = transpose_matrix(&zeros);
    assert_eq!(got.as_ref().map(|out| out.shape.clone()), Ok(vec![4, 0]));
    assert_eq!(got.as_ref().map(|out| out.strides.clone()), Ok(vec![2, 8]));
    assert_eq!(got.as_ref().map(|out| out.offset), Ok(1));
    assert_eq!(got.as_ref().map(|out| out.storage.storage_id()), Ok(zero_id));
}

fn assert_shape<S>(result: Result<Tensor<S>, Error>) {
    assert_eq!(result.err(), Some(Error::Shape));
}

#[test]
fn transpose_rejects_rank_other_than_2() {
    assert_shape(transpose_matrix(&rank_tensor(vec![])));
    assert_shape(transpose_matrix(&rank_tensor(vec![7])));
    assert_shape(transpose_matrix(&rank_tensor(vec![2, 3, 4])));
    assert_shape(transpose_matrix(&rank_tensor(vec![2, 3, 4, 5])));
}

#[test]
fn wire_roundtrip_f32_f16_and_bf16() {
    assert_eq!(WIRE_MAGIC, *b"LNGL");
    assert_eq!(WIRE_VERSION, 1);
    assert_eq!(DType::F32.size_of(), 4);
    assert_eq!(DType::F16.size_of(), 2);
    assert_eq!(DType::BF16.size_of(), 2);

    assert_eq!(encode_wire(DType::F32, &[1], &[0x11, 0x22, 0x33, 0x44]), Ok(F32_SCALAR.to_vec()));
    assert_eq!(decode_wire(F32_SCALAR), Ok((DType::F32, vec![1], vec![0x11, 0x22, 0x33, 0x44])));

    assert_eq!(
        encode_wire(DType::F16, &[2, 1], &[0xAB, 0xCD, 0x11, 0x22]),
        Ok(F16_MATRIX.to_vec())
    );
    assert_eq!(decode_wire(F16_MATRIX), Ok((DType::F16, vec![2, 1], vec![0xAB, 0xCD, 0x11, 0x22])));

    assert_eq!(encode_wire(DType::BF16, &[], &[0xEE, 0xFF]), Ok(BF16_SCALAR.to_vec()));
    assert_eq!(decode_wire(BF16_SCALAR), Ok((DType::BF16, vec![], vec![0xEE, 0xFF])));

    assert_eq!(encode_wire(DType::F32, &[0, 5], &[]), Ok(F32_ZERO_EXTENT.to_vec()));
    assert_eq!(decode_wire(F32_ZERO_EXTENT), Ok((DType::F32, vec![0, 5], vec![])));

    let payload = vec![0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE];
    let encoded = encode_wire(DType::F16, &[2, 1, 2], &payload);
    let round = encoded.as_ref().map(|bytes| decode_wire(bytes));
    assert_eq!(round, Ok(Ok((DType::F16, vec![2, 1, 2], payload))));

    assert_eq!(encode_wire(DType::F32, &[1], &[0x11, 0x22]), Err(Error::Wire));
    assert_eq!(encode_wire(DType::F32, &[1], &[0x11, 0x22, 0x33, 0x44, 0x55]), Err(Error::Wire));
    assert_eq!(encode_wire(DType::F16, &[], &[0xEE, 0xFF, 0x00, 0x00]), Err(Error::Wire));
    assert_eq!(encode_wire(DType::BF16, &[2], &[0xEE, 0xFF]), Err(Error::Wire));
    assert_eq!(encode_wire(DType::F32, &[0, 5], &[0x00]), Err(Error::Wire));
}

#[test]
fn wire_length_is_16_plus_4_times_ndim_plus_payload() {
    let scalar = encode_wire(DType::F32, &[], &[1, 2, 3, 4]);
    assert_eq!(scalar.as_ref().map(Vec::len), Ok(20));
    assert_eq!(scalar.as_ref().map(|bytes| bytes[..4].to_vec()), Ok(b"LNGL".to_vec()));
    assert_eq!(
        scalar.as_ref().map(|bytes| bytes[bytes.len() - 4..].to_vec()),
        Ok(vec![1, 2, 3, 4])
    );

    let matrix = encode_wire(DType::F32, &[2, 3], &[0xAB; 24]);
    assert_eq!(matrix.as_ref().map(Vec::len), Ok(48));
    assert_eq!(matrix.as_ref().map(|bytes| bytes[bytes.len() - 24..].to_vec()), Ok(vec![0xAB; 24]));

    let rank8 = encode_wire(DType::F16, &[2, 1, 1, 1, 1, 1, 1, 1], &[9, 8, 7, 6]);
    assert_eq!(rank8.as_ref().map(Vec::len), Ok(52));
    assert_eq!(
        rank8.as_ref().map(|bytes| u32::from_le_bytes(bytes[12..16].try_into().unwrap())),
        Ok(8)
    );
    assert_eq!(rank8.as_ref().map(|bytes| bytes[bytes.len() - 4..].to_vec()), Ok(vec![9, 8, 7, 6]));
}

#[test]
fn wire_rejects_bad_magic_version_ndim_truncation_and_trailing_bytes() {
    let mut bad_magic = F32_SCALAR.to_vec();
    bad_magic[0] = b'X';
    assert_eq!(decode_wire(&bad_magic), Err(Error::Wire));

    let mut bad_version = F32_SCALAR.to_vec();
    bad_version[4..8].copy_from_slice(&2u32.to_le_bytes());
    assert_eq!(decode_wire(&bad_version), Err(Error::Wire));

    let mut version_zero = F32_SCALAR.to_vec();
    version_zero[4..8].copy_from_slice(&0u32.to_le_bytes());
    assert_eq!(decode_wire(&version_zero), Err(Error::Wire));

    assert_eq!(decode_wire(&[]), Err(Error::Wire));
    assert_eq!(decode_wire(&[0u8; 15]), Err(Error::Wire));
    assert_eq!(decode_wire(&F32_SCALAR[..16]), Err(Error::Wire));
    assert_eq!(decode_wire(&F32_SCALAR[..F32_SCALAR.len() - 1]), Err(Error::Wire));

    let mut trailing = F32_SCALAR.to_vec();
    trailing.push(0xFF);
    assert_eq!(decode_wire(&trailing), Err(Error::Wire));

    let nine = message(1, &[1; 9], &[0, 0, 0, 0]);
    assert_eq!(decode_wire(&nine), Err(Error::Wire));
    assert_eq!(
        encode_wire(DType::F32, &[1, 1, 1, 1, 1, 1, 1, 1, 1], &[0, 0, 0, 0]),
        Err(Error::Wire)
    );

    assert_eq!(decode_wire(&message(0, &[1], &[0, 0, 0, 0])), Err(Error::Wire));
    assert_eq!(decode_wire(&message(4, &[1], &[0, 0, 0, 0])), Err(Error::Wire));

    let mut mid_extent = message(1, &[1, 1], &[0, 0, 0, 0]);
    mid_extent.truncate(18);
    assert_eq!(decode_wire(&mid_extent), Err(Error::Wire));
}

#[test]
fn wire_rejects_dimension_above_u32_and_overflowing_numel() {
    let above = (u32::MAX as usize).checked_add(1);
    assert!(above.is_some());
    let above = above.unwrap();
    assert_eq!(encode_wire(DType::F32, &[above], &[]), Err(Error::Shape));

    let disguised = above + 5;
    assert_eq!(encode_wire(DType::F16, &[disguised], &[]), Err(Error::Shape));
    assert_eq!(encode_wire(DType::BF16, &[1, disguised, 2], &[0, 0]), Err(Error::Shape));

    assert_eq!(encode_wire(DType::F32, &[u32::MAX as usize], &[]), Err(Error::Wire));

    assert_eq!(
        encode_wire(DType::F32, &[1 << 16, 1 << 16, 1 << 16, 1 << 16], &[]),
        Err(Error::Shape)
    );
    assert_eq!(encode_wire(DType::F32, &[1 << 31, 1 << 31], &[]), Err(Error::Shape));
    assert_eq!(encode_wire(DType::F16, &[1 << 31, 1 << 31, 2], &[]), Err(Error::Shape));
    assert_eq!(encode_wire(DType::BF16, &[1 << 31, 1 << 31, 2], &[]), Err(Error::Shape));

    assert_eq!(decode_wire(&message(1, &[65536, 65536, 65536, 65536], &[])), Err(Error::Shape));
    assert_eq!(decode_wire(&message(1, &[1 << 31, 1 << 31], &[])), Err(Error::Shape));
    assert_eq!(decode_wire(&message(2, &[1 << 31, 1 << 31, 2], &[])), Err(Error::Shape));
}
