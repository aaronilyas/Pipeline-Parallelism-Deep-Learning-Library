// Permutation primitive: strided-to-contiguous materialization with optional complex
// conjugation. Elements move as raw bit patterns: `uint` for 4-byte types (f32, u32, i32) and
// `uint2` for complex f32 (x = re, y = im).
//
// The destination is always contiguous row-major with the (coalesced) shape of the source
// view; the source is addressed through arbitrary element strides. All index arithmetic is
// 32-bit; the host proves every reachable index fits before launching.

#include <metal_stdlib>
using namespace metal;

/// Maximum number of axes after the host merges adjacent contiguous axes. Mirrors MAX_RANK in
/// permutation.rs.
constant constexpr uint MAX_RANK = 8;
/// Edge of the square tile staged through threadgroup memory.
constant constexpr uint TILE = 32;
/// Threadgroup height; each thread moves TILE / TILE_ROWS = 4 tile rows.
constant constexpr uint TILE_ROWS = 8;
/// Sign bit of an f32 bit pattern.
constant constexpr uint SIGN_BIT = 0x80000000u;

/// Parameters of the coalesced strided-copy kernels. Mirrors `CopyParams` in permutation.rs.
struct CopyParams {
    /// Number of elements to move.
    uint count;
    /// Number of axes in use (at least 1).
    uint rank;
    /// Source offset in elements.
    uint src_offset;
    /// Destination offset in elements.
    uint dst_offset;
    /// Extents, outermost axis first; the destination is row-major over them.
    uint shape[MAX_RANK];
    /// Source strides in elements, one per axis.
    uint src_strides[MAX_RANK];
};

/// Parameters of the tiled transpose kernels. Mirrors `TileParams` in permutation.rs.
///
/// Axis A is the source's unit-stride axis, axis B the destination's unit-stride axis. All
/// other axes ("outer") are enumerated by the z coordinate of the threadgroup grid.
struct TileParams {
    /// Source offset in elements.
    uint src_offset;
    /// Destination offset in elements.
    uint dst_offset;
    /// Extent of axis A.
    uint extent_a;
    /// Extent of axis B.
    uint extent_b;
    /// Source stride of axis B (axis A has source stride 1).
    uint src_stride_b;
    /// Destination stride of axis A (axis B has destination stride 1).
    uint dst_stride_a;
    /// Number of outer axes.
    uint outer_rank;
    /// Extents of the outer axes, outermost first.
    uint outer_shape[MAX_RANK];
    /// Source strides of the outer axes.
    uint outer_src[MAX_RANK];
    /// Destination strides of the outer axes.
    uint outer_dst[MAX_RANK];
};

/// Conjugation of a 4-byte element is the identity; the `CONJ` kernels are only instantiated
/// for complex elements.
inline uint flip_imag(uint v) { return v; }

/// Conjugates a complex element by flipping the sign bit of the imaginary word, which keeps
/// -0.0, infinities and NaN payloads intact and makes conjugation an exact involution.
inline uint2 flip_imag(uint2 v) { return uint2(v.x, v.y ^ SIGN_BIT); }

/// Coalesced strided copy: each thread gathers one element through the source strides and
/// writes it at its linear destination index, so destination writes are fully coalesced.
template <typename V, bool CONJ>
kernel void permutation_copy_impl(device const V* src [[buffer(0)]],
                                  device V* dst [[buffer(1)]],
                                  constant CopyParams& p [[buffer(2)]],
                                  uint i [[thread_position_in_grid]]) {
    if (i >= p.count) {
        return;
    }
    uint rem = i;
    uint s = p.src_offset;
    for (uint d = p.rank - 1; d > 0; --d) {
        uint q = rem / p.shape[d];
        s += (rem - q * p.shape[d]) * p.src_strides[d];
        rem = q;
    }
    s += rem * p.src_strides[0];
    V v = src[s];
    if (CONJ) {
        v = flip_imag(v);
    }
    dst[p.dst_offset + i] = v;
}

/// Tiled transpose of axes A and B for one outer index. A 32x32 tile is read with
/// coalesced loads along A, staged in threadgroup memory padded to 33 columns (no bank
/// conflicts on either access direction), then written with coalesced stores along B.
/// The 32x8 threadgroup moves four tile rows per thread; partial tiles are guarded.
template <typename V, bool CONJ>
kernel void permutation_tile_impl(device const V* src [[buffer(0)]],
                                  device V* dst [[buffer(1)]],
                                  constant TileParams& p [[buffer(2)]],
                                  uint3 group [[threadgroup_position_in_grid]],
                                  uint3 lane [[thread_position_in_threadgroup]]) {
    threadgroup V tile[TILE][TILE + 1];

    uint src_base = p.src_offset;
    uint dst_base = p.dst_offset;
    uint rem = group.z;
    for (uint d = p.outer_rank; d-- > 0;) {
        uint q = rem / p.outer_shape[d];
        uint c = rem - q * p.outer_shape[d];
        src_base += c * p.outer_src[d];
        dst_base += c * p.outer_dst[d];
        rem = q;
    }

    uint b0 = group.x * TILE;
    uint a0 = group.y * TILE;

    uint a = a0 + lane.x;
    for (uint r = 0; r < TILE; r += TILE_ROWS) {
        uint b = b0 + lane.y + r;
        if (a < p.extent_a && b < p.extent_b) {
            tile[lane.y + r][lane.x] = src[src_base + a + b * p.src_stride_b];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint b = b0 + lane.x;
    for (uint r = 0; r < TILE; r += TILE_ROWS) {
        uint a_out = a0 + lane.y + r;
        if (a_out < p.extent_a && b < p.extent_b) {
            V v = tile[lane.x][lane.y + r];
            if (CONJ) {
                v = flip_imag(v);
            }
            dst[dst_base + a_out * p.dst_stride_a + b] = v;
        }
    }
}

/// Strided copy of 4-byte elements.
template [[host_name("permutation_copy_w1")]] kernel void
permutation_copy_impl<uint, false>(device const uint*, device uint*, constant CopyParams&, uint);
/// Strided copy of 8-byte elements.
template [[host_name("permutation_copy_w2")]] kernel void
permutation_copy_impl<uint2, false>(device const uint2*, device uint2*, constant CopyParams&, uint);
/// Strided copy of complex f32 elements with conjugation.
template [[host_name("permutation_copy_conj_w2")]] kernel void
permutation_copy_impl<uint2, true>(device const uint2*, device uint2*, constant CopyParams&, uint);

/// Tiled transpose of 4-byte elements.
template [[host_name("permutation_tile_w1")]] kernel void
permutation_tile_impl<uint, false>(device const uint*, device uint*, constant TileParams&, uint3, uint3);
/// Tiled transpose of 8-byte elements.
template [[host_name("permutation_tile_w2")]] kernel void
permutation_tile_impl<uint2, false>(device const uint2*, device uint2*, constant TileParams&, uint3, uint3);
/// Tiled transpose of complex f32 elements with conjugation.
template [[host_name("permutation_tile_conj_w2")]] kernel void
permutation_tile_impl<uint2, true>(device const uint2*, device uint2*, constant TileParams&, uint3, uint3);
