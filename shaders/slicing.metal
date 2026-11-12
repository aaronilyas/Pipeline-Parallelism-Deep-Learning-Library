// Slicing primitive: bit-exact pack (strided copy) and concatenation.
//
// Elements move as raw integers — `ushort` (2 bytes), `uint` (4), `ulong` (8), or `ulong2`
// (16) — so NaN payloads and -0.0 survive. The host binds every buffer at byte offset 0 and
// passes element offsets, shapes, and strides in the params struct via `setBytes`. Concat
// also reads a part table from a device buffer, because its size grows with the destination
// extent along the concat axis and can exceed the 4 KB `setBytes` limit.
//
// Pack has two kernels. The scalar kernel addresses an arbitrary stride. The vector kernel
// runs only when the last axis of both views has stride 1 (or the whole view is one contiguous
// run) and the two row addresses share a 16-byte alignment. One dispatch then copies, in
// order, a scalar head, `ulong4` body (32 bytes), `ulong2` body (16 bytes), and a scalar tail.
//
// Worked example, f32, row byte-misalignment 4 (element offset ≡ 1 mod 4), inner length 16:
//   head = 3 elements (12 bytes) to reach a 16-byte boundary
//   one ulong4 covers the next 8 elements
//   one ulong2 covers the next 4
//   tail = 1 element
//   3 + 8 + 4 + 1 = 16, and every row of the dispatch uses that same split.
//
// All index arithmetic is 32-bit. The host proves every reachable index fits before launch.
// Concatenation is one dispatch over the destination: each thread decodes its logical index,
// finds the source part along `axis`, and copies one element.

#include <metal_stdlib>
using namespace metal;

/// Maximum rank. Mirrors `MAX_RANK` in slicing.rs.
constant constexpr uint MAX_RANK = 8;
/// Maximum parts in one concat dispatch. Mirrors `MAX_CONCAT_PARTS` in slicing.rs.
constant constexpr uint MAX_CONCAT_PARTS = 16;

/// Parameters of the scalar pack kernels. Mirrors `PackParams` in slicing.rs.
struct PackParams {
    /// Number of logical elements.
    uint count;
    /// Rank after a rank-0 view has been expanded to a single element. At least 1.
    uint rank;
    /// Source offset in elements.
    uint src_offset;
    /// Destination offset in elements.
    uint dst_offset;
    /// Extents, outermost axis first.
    uint shape[MAX_RANK];
    /// Source strides in elements.
    uint src_strides[MAX_RANK];
    /// Destination strides in elements.
    uint dst_strides[MAX_RANK];
};

/// Parameters of the vector pack kernels. Mirrors `VecParams` in slicing.rs.
///
/// Thread `(slot, row)` owns one piece of row `row`: a scalar element, one `ulong4`, or one
/// `ulong2`. `threads_per_row = head + n32 + n16 + tail`.
struct VecParams {
    /// Number of rows. A fully contiguous pair is one row of `inner` elements.
    uint n_rows;
    /// Elements in each row.
    uint inner;
    /// Element index of row 0's first element. Outer axes are added per row.
    uint src_offset;
    /// Destination element index of row 0.
    uint dst_offset;
    /// Number of axes outside the packed row. 0 when `n_rows` is 1 and the row is the whole view.
    uint outer_rank;
    /// Slots in one row: scalar head, then `ulong4` chunks, then `ulong2` chunks, then scalar tail.
    uint threads_per_row;
    /// Leading scalar elements that bring the row to a 16-byte boundary. May consume the row.
    uint head;
    /// `ulong4` (32-byte) chunks after `head`.
    uint n32;
    /// `ulong2` (16-byte) chunks after the `ulong4`s.
    uint n16;
    /// Trailing scalar elements.
    uint tail;
    /// Extents of the outer axes, outermost first. Unused when `outer_rank` is 0.
    uint outer_shape[MAX_RANK];
    /// Source strides of the outer axes, in elements.
    uint src_outer[MAX_RANK];
    /// Destination strides of the outer axes, in elements.
    uint dst_outer[MAX_RANK];
};

/// Parameters of the concat kernels. Mirrors `ConcatParams` in slicing.rs.
///
/// The per-coordinate part id lives in the device table at buffer 18, not in this struct,
/// because the table grows with the destination extent and can exceed the `setBytes` limit.
struct ConcatParams {
    /// Number of logical destination elements.
    uint count;
    /// Rank. At least 1.
    uint rank;
    /// Axis along which parts are laid out, in order.
    uint axis;
    /// Number of parts, including empty ones.
    uint n_parts;
    /// Destination offset in elements.
    uint dst_offset;
    /// Destination extents, outermost axis first.
    uint dst_shape[MAX_RANK];
    /// Destination strides in elements.
    uint dst_strides[MAX_RANK];
    /// Element offset of each part.
    uint part_offset[MAX_CONCAT_PARTS];
    /// Strides of each part, `part_strides[part][axis]`, matching the Rust array layout.
    uint part_strides[MAX_CONCAT_PARTS][MAX_RANK];
};

/// Copies one element.
template <typename E>
inline void copy_one(device const E* src, device E* dst, uint si, uint di) {
    dst[di] = src[si];
}

/// Copies 32 bytes (`ulong4`) starting at an element whose address is 16-byte aligned.
template <typename E>
inline void copy32(device const E* src, device E* dst, uint si, uint di) {
    device const ulong4* sp = reinterpret_cast<device const ulong4*>(src + si);
    device ulong4* dp = reinterpret_cast<device ulong4*>(dst + di);
    *dp = *sp;
}

/// Copies 16 bytes (`ulong2`) starting at an element whose address is 16-byte aligned.
template <typename E>
inline void copy16(device const E* src, device E* dst, uint si, uint di) {
    device const ulong2* sp = reinterpret_cast<device const ulong2*>(src + si);
    device ulong2* dp = reinterpret_cast<device ulong2*>(dst + di);
    *dp = *sp;
}

/// Scalar pack: one thread per logical element, in row-major order.
template <typename E>
kernel void slicing_pack_impl(device const E* src [[buffer(0)]],
                              device E* dst [[buffer(1)]],
                              constant PackParams& p [[buffer(2)]],
                              uint i [[thread_position_in_grid]]) {
    if (i >= p.count) {
        return;
    }
    uint rem = i;
    uint s = p.src_offset;
    uint d = p.dst_offset;
    for (uint ax = p.rank - 1; ax > 0; --ax) {
        uint q = rem / p.shape[ax];
        uint c = rem - q * p.shape[ax];
        s += c * p.src_strides[ax];
        d += c * p.dst_strides[ax];
        rem = q;
    }
    s += rem * p.src_strides[0];
    d += rem * p.dst_strides[0];
    dst[d] = src[s];
}

/// Vector pack. `gid.y` selects the row and `gid.x` the slot inside the head / vector / tail split.
template <typename E>
kernel void slicing_pack_vec_impl(device const E* src [[buffer(0)]],
                                  device E* dst [[buffer(1)]],
                                  constant VecParams& p [[buffer(2)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    constexpr uint PER32 = sizeof(ulong4) / sizeof(E);
    constexpr uint PER16 = sizeof(ulong2) / sizeof(E);

    uint row = gid.y;
    uint slot = gid.x;
    if (row >= p.n_rows || slot >= p.threads_per_row) {
        return;
    }

    // Row start. `outer_rank <= 1` is the streaming case (a contiguous run, or a matrix with a
    // constant leading dimension) and skips the general outer-index decode.
    uint src_base;
    uint dst_base;
    if (p.outer_rank == 0) {
        src_base = p.src_offset;
        dst_base = p.dst_offset;
    } else if (p.outer_rank == 1) {
        src_base = p.src_offset + row * p.src_outer[0];
        dst_base = p.dst_offset + row * p.dst_outer[0];
    } else {
        src_base = p.src_offset;
        dst_base = p.dst_offset;
        uint rem = row;
        for (uint d = p.outer_rank; d-- > 0;) {
            uint q = rem / p.outer_shape[d];
            uint c = rem - q * p.outer_shape[d];
            src_base += c * p.src_outer[d];
            dst_base += c * p.dst_outer[d];
            rem = q;
        }
    }

    uint head = p.head;
    uint n32 = p.n32;
    uint n16 = p.n16;
    if (slot < head) {
        copy_one(src, dst, src_base + slot, dst_base + slot);
        return;
    }
    uint past_head = slot - head;
    if (past_head < n32) {
        uint elem = head + past_head * PER32;
        copy32(src, dst, src_base + elem, dst_base + elem);
        return;
    }
    uint past_32 = past_head - n32;
    if (past_32 < n16) {
        uint elem = head + n32 * PER32 + past_32 * PER16;
        copy16(src, dst, src_base + elem, dst_base + elem);
        return;
    }
    uint tail_i = past_32 - n16;
    uint elem = head + n32 * PER32 + n16 * PER16 + tail_i;
    copy_one(src, dst, src_base + elem, dst_base + elem);
}

/// Selects part `part`. Only indices below `n_parts` are read; the rest are bound but unused.
template <typename E>
inline device const E* part_buffer(uint part,
                                   device const E* b0,
                                   device const E* b1,
                                   device const E* b2,
                                   device const E* b3,
                                   device const E* b4,
                                   device const E* b5,
                                   device const E* b6,
                                   device const E* b7,
                                   device const E* b8,
                                   device const E* b9,
                                   device const E* b10,
                                   device const E* b11,
                                   device const E* b12,
                                   device const E* b13,
                                   device const E* b14,
                                   device const E* b15) {
    switch (part) {
    case 0:
        return b0;
    case 1:
        return b1;
    case 2:
        return b2;
    case 3:
        return b3;
    case 4:
        return b4;
    case 5:
        return b5;
    case 6:
        return b6;
    case 7:
        return b7;
    case 8:
        return b8;
    case 9:
        return b9;
    case 10:
        return b10;
    case 11:
        return b11;
    case 12:
        return b12;
    case 13:
        return b13;
    case 14:
        return b14;
    default:
        return b15;
    }
}

/// Concatenation: one thread per destination element, in row-major logical order.
///
/// `p` is the mirrored `ConcatParams`, bound with `setBytes` at buffer 17. `table` holds
/// `(part, base)` pairs indexed by the coordinate along `axis`; `base` is that part's
/// exclusive prefix. Coordinates stay in scalars.
template <typename E>
kernel void slicing_concat_impl(device const E* b0 [[buffer(0)]],
                                device const E* b1 [[buffer(1)]],
                                device const E* b2 [[buffer(2)]],
                                device const E* b3 [[buffer(3)]],
                                device const E* b4 [[buffer(4)]],
                                device const E* b5 [[buffer(5)]],
                                device const E* b6 [[buffer(6)]],
                                device const E* b7 [[buffer(7)]],
                                device const E* b8 [[buffer(8)]],
                                device const E* b9 [[buffer(9)]],
                                device const E* b10 [[buffer(10)]],
                                device const E* b11 [[buffer(11)]],
                                device const E* b12 [[buffer(12)]],
                                device const E* b13 [[buffer(13)]],
                                device const E* b14 [[buffer(14)]],
                                device const E* b15 [[buffer(15)]],
                                device E* dst [[buffer(16)]],
                                constant ConcatParams& p [[buffer(17)]],
                                device const uint* table [[buffer(18)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= p.count) {
        return;
    }
    uint rank = p.rank;
    uint axis = p.axis;

    // Destination uses full coordinates. The source uses the same coordinates except on
    // `axis`, where the part-local index replaces the destination coordinate.
    uint rem = i;
    uint along = 0;
    uint d = p.dst_offset;
    for (uint ax = rank - 1; ax > 0; --ax) {
        uint ext = p.dst_shape[ax];
        uint q = rem / ext;
        uint c = rem - q * ext;
        if (ax == axis) {
            along = c;
        }
        d += c * p.dst_strides[ax];
        rem = q;
    }
    if (axis == 0) {
        along = rem;
    }
    d += rem * p.dst_strides[0];

    ulong slot = (ulong)along * 2ul;
    uint part = table[slot];
    uint local = along - table[slot + 1ul];

    rem = i;
    uint s = p.part_offset[part];
    for (uint ax = rank - 1; ax > 0; --ax) {
        uint ext = p.dst_shape[ax];
        uint q = rem / ext;
        uint c = rem - q * ext;
        if (ax == axis) {
            c = local;
        }
        s += c * p.part_strides[part][ax];
        rem = q;
    }
    if (axis == 0) {
        rem = local;
    }
    s += rem * p.part_strides[part][0];

    device const E* src = part_buffer(part, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12,
                                      b13, b14, b15);
    dst[d] = src[s];
}

/// Scalar pack of 2-byte elements.
template [[host_name("slicing_pack_e2")]] kernel void
slicing_pack_impl<ushort>(device const ushort*, device ushort*, constant PackParams&, uint);
/// Scalar pack of 4-byte elements.
template [[host_name("slicing_pack_e4")]] kernel void
slicing_pack_impl<uint>(device const uint*, device uint*, constant PackParams&, uint);
/// Scalar pack of 8-byte elements.
template [[host_name("slicing_pack_e8")]] kernel void
slicing_pack_impl<ulong>(device const ulong*, device ulong*, constant PackParams&, uint);
/// Scalar pack of 16-byte elements.
template [[host_name("slicing_pack_e16")]] kernel void
slicing_pack_impl<ulong2>(device const ulong2*, device ulong2*, constant PackParams&, uint);

/// Vector pack of 2-byte elements (`ushort` head and tail, `ulong4` / `ulong2` body).
template [[host_name("slicing_pack_vec_e2")]] kernel void
slicing_pack_vec_impl<ushort>(device const ushort*, device ushort*, constant VecParams&, uint2);
/// Vector pack of 4-byte elements.
template [[host_name("slicing_pack_vec_e4")]] kernel void
slicing_pack_vec_impl<uint>(device const uint*, device uint*, constant VecParams&, uint2);
/// Vector pack of 8-byte elements.
template [[host_name("slicing_pack_vec_e8")]] kernel void
slicing_pack_vec_impl<ulong>(device const ulong*, device ulong*, constant VecParams&, uint2);
/// Vector pack of 16-byte elements.
template [[host_name("slicing_pack_vec_e16")]] kernel void
slicing_pack_vec_impl<ulong2>(device const ulong2*, device ulong2*, constant VecParams&, uint2);

/// Concatenation of 2-byte elements. Sixteen part buffers, then the destination.
template [[host_name("slicing_concat_e2")]] kernel void
slicing_concat_impl<ushort>(device const ushort*, device const ushort*, device const ushort*,
                            device const ushort*, device const ushort*, device const ushort*,
                            device const ushort*, device const ushort*, device const ushort*,
                            device const ushort*, device const ushort*, device const ushort*,
                            device const ushort*, device const ushort*, device const ushort*,
                            device const ushort*, device ushort*, constant ConcatParams&, device const uint*, uint);
/// Concatenation of 4-byte elements. Sixteen part buffers, then the destination.
template [[host_name("slicing_concat_e4")]] kernel void
slicing_concat_impl<uint>(device const uint*, device const uint*, device const uint*,
                          device const uint*, device const uint*, device const uint*,
                          device const uint*, device const uint*, device const uint*,
                          device const uint*, device const uint*, device const uint*,
                          device const uint*, device const uint*, device const uint*,
                          device const uint*, device uint*, constant ConcatParams&, device const uint*, uint);
/// Concatenation of 8-byte elements. Sixteen part buffers, then the destination.
template [[host_name("slicing_concat_e8")]] kernel void
slicing_concat_impl<ulong>(device const ulong*, device const ulong*, device const ulong*,
                           device const ulong*, device const ulong*, device const ulong*,
                           device const ulong*, device const ulong*, device const ulong*,
                           device const ulong*, device const ulong*, device const ulong*,
                           device const ulong*, device const ulong*, device const ulong*,
                           device const ulong*, device ulong*, constant ConcatParams&, device const uint*, uint);
/// Concatenation of 16-byte elements. Sixteen part buffers, then the destination.
template [[host_name("slicing_concat_e16")]] kernel void
slicing_concat_impl<ulong2>(device const ulong2*, device const ulong2*, device const ulong2*,
                            device const ulong2*, device const ulong2*, device const ulong2*,
                            device const ulong2*, device const ulong2*, device const ulong2*,
                            device const ulong2*, device const ulong2*, device const ulong2*,
                            device const ulong2*, device const ulong2*, device const ulong2*,
                            device const ulong2*, device ulong2*, constant ConcatParams&, device const uint*, uint);
