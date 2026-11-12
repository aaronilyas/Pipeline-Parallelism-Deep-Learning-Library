#include <metal_stdlib>
using namespace metal;

/// Host-side mirror of `SqrtParams` in `src/ops/sqrt.rs`. Field order is the `#[repr(C)]` layout.
/// Indexes are in elements. The host launches this kernel only after proving that `count` and
/// every reachable index fit in 32 bits.
struct SqrtParams {
    uint src_offset;
    uint dst_offset;
    uint rank;
    uint count;
    uint shape[8];
    uint src_strides[8];
    uint dst_strides[8];
};

/// Correctly rounded `f32` square root, including subnormals.
///
/// `precise::sqrt` on Apple GPUs flushes a subnormal input to zero (pinned by
/// `probe_subnormal_flush_is_pinned`). A positive subnormal is scaled by 2^48 with integer
/// arithmetic, square-rooted, then scaled by 2^-24:
/// `sqrt(x) = sqrt(x * 2^48) * 2^-24`. Both the scaled input and the scaled root are normal,
/// and the power-of-two scale is exact, so this matches `(x as f64).sqrt() as f32`.
/// A negative subnormal would flush to `-0` and come back as `-0`; it is negative, so the
/// result is a canonical NaN.
inline float sqrt_exact(uint bits) {
    uint exp = (bits >> 23) & 0xffu;
    uint mant = bits & 0x7fffffu;
    if (exp == 0u && mant != 0u) {
        if ((bits & 0x80000000u) != 0u) {
            return as_type<float>(0x7fc00000u);
        }
        uint shift = clz(mant) - 8u;
        uint frac = (mant << shift) & 0x7fffffu;
        uint biased = 49u - shift;
        float scaled = as_type<float>((biased << 23) | frac);
        return precise::sqrt(scaled) * 0x1p-24f;
    }
    return precise::sqrt(as_type<float>(bits));
}

/// Elementwise `f32` square root. One thread owns one logical element.
///
/// Logical index `tid` is unraveled in row-major order into `shape` and then applied to
/// `src_strides` and `dst_strides`. Rank 1 skips the division. Rank 0 is the single element
/// at the two offsets. Buffers are `uint` so a subnormal's bits survive the load; stores of
/// `-0` and NaN keep the bits `sqrt_exact` produced.
kernel void sqrt_f32 [[host_name("sqrt_f32")]] (
    device uint* src [[buffer(0)]],
    device uint* dst [[buffer(1)]],
    constant SqrtParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.count) {
        return;
    }
    uint src_i = params.src_offset;
    uint dst_i = params.dst_offset;
    if (params.rank == 1u) {
        src_i += tid * params.src_strides[0];
        dst_i += tid * params.dst_strides[0];
    } else if (params.rank > 1u) {
        uint remainder = tid;
        for (int axis = int(params.rank) - 1; axis >= 0; --axis) {
            uint extent = params.shape[axis];
            uint coord = remainder % extent;
            remainder /= extent;
            src_i += coord * params.src_strides[axis];
            dst_i += coord * params.dst_strides[axis];
        }
    }
    dst[dst_i] = as_type<uint>(sqrt_exact(src[src_i]));
}
