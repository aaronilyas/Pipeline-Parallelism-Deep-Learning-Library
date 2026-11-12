#include <metal_stdlib>
using namespace metal;

// Smith's soft-float add is a separate multiply and add. Contraction into an fma would
// disagree with that rounding, so keep every product and sum unfused.
#pragma clang fp contract(off)

/// Host-side mirror of `RecipParams` in `src/ops/reciprocal.rs`. Field order is the
/// `#[repr(C)]` layout. Indexes are in elements. The host launches a kernel only after
/// proving that `count` and every reachable index fit in 32 bits.
struct RecipParams {
    uint src_offset;
    uint dst_offset;
    uint rank;
    uint count;
    uint shape[3];
    uint src_strides[3];
    uint dst_strides[3];
};

/// Round-to-nearest-even integer quotient `numer / sig`. `sig` is nonzero.
inline ulong rne_quot(ulong numer, ulong sig) {
    ulong q = numer / sig;
    ulong rem = numer % sig;
    if ((rem << 1) > sig || ((rem << 1) == sig && (q & 1UL) == 1UL)) {
        q += 1UL;
    }
    return q;
}

/// Exact power of two `±2^exp_unbiased`, including overflow to infinity and subnormals.
inline uint pack_pow2(uint sign, int exp_unbiased) {
    if (exp_unbiased >= 128) {
        return sign | 0x7f800000u;
    }
    // 2^(-150) is the tie between 0 and the smallest subnormal; ties round to even zero.
    if (exp_unbiased <= -150) {
        return sign;
    }
    if (exp_unbiased >= -126) {
        uint biased = uint(exp_unbiased + 127);
        return sign | (biased << 23);
    }
    uint shift = uint(exp_unbiased + 149);
    return sign | (1u << shift);
}

/// Correctly rounded reciprocal whose normalized significand is in (1, 2) at `result_exp`.
/// `sig` is the 24-bit input significand in (2^23, 2^24).
inline uint pack_normal(uint sign, int result_exp, uint sig) {
    ulong q = rne_quot(1UL << 47, ulong(sig));
    if (q == (1UL << 24)) {
        int biased = result_exp + 1 + 127;
        if (biased >= 255) {
            return sign | 0x7f800000u;
        }
        return sign | (uint(biased) << 23);
    }
    int biased = result_exp + 127;
    if (biased >= 255) {
        return sign | 0x7f800000u;
    }
    return sign | (uint(biased) << 23) | (uint(q) & 0x007fffffu);
}

/// Correctly rounded reciprocal that lands in the subnormal range, or rounds up to the
/// smallest normal. The unit in the last place is 2^(-149) throughout this range.
inline uint pack_subnormal(uint sign, int result_exp, uint sig) {
    int p = 173 + result_exp;
    uint m = uint(rne_quot(1UL << uint(p), ulong(sig)));
    if (m == 0u) {
        return sign;
    }
    if (m < (1u << 23)) {
        return sign | m;
    }
    uint frac = m - (1u << 23);
    if (frac < (1u << 23)) {
        return sign | (1u << 23) | frac;
    }
    return sign | (2u << 23) | (frac - (1u << 23));
}

/// Correctly rounded `f32` reciprocal, bit pattern in and bit pattern out.
///
/// This GPU flushes subnormal operands to zero and subnormal results to a signed zero
/// (pinned by `probe_subnormal_flush_is_pinned`). Every special, subnormal input, and
/// subnormal result is therefore classified and packed with integer arithmetic. The
/// significand division is a 64-bit round-to-nearest-even quotient, which matches
/// `(1.0 / x as f64) as f32` on every one of the 2^32 inputs, so no Newton step is applied.
/// `precise::divide` is 0 ULP for normal quotients but flushes a subnormal quotient, so it
/// is not used here.
inline uint recip_f32_bits(uint bits) {
    uint sign = bits & 0x80000000u;
    uint exp = (bits >> 23) & 0xffu;
    uint frac = bits & 0x007fffffu;

    if (exp == 0xffu) {
        if (frac == 0u) {
            return sign;
        }
        // Quiet a signaling NaN. The sign and the rest of the payload stay.
        return bits | 0x00400000u;
    }
    if ((bits & 0x7fffffffu) == 0u) {
        return sign | 0x7f800000u;
    }

    uint sig;
    int unbiased;
    if (exp == 0u) {
        uint shift = clz(frac) - 8u;
        sig = frac << shift;
        unbiased = -126 - int(shift);
    } else {
        sig = frac | 0x00800000u;
        unbiased = int(exp) - 127;
    }

    if (sig == 0x00800000u) {
        return pack_pow2(sign, -unbiased);
    }

    int result_exp = -unbiased - 1;
    if (result_exp >= 128) {
        return sign | 0x7f800000u;
    }
    if (result_exp <= -127) {
        return pack_subnormal(sign, result_exp, sig);
    }
    return pack_normal(sign, result_exp, sig);
}

/// A finite float as a normal significand in [1, 2) (or a signed zero) times `2^e`.
/// Subnormal inputs are normalized in integer arithmetic, so the float unit never sees them.
struct Soft {
    float m;
    int e;
};

inline Soft soft_from_bits(uint bits) {
    uint exp = (bits >> 23) & 0xffu;
    uint frac = bits & 0x007fffffu;
    uint sign = bits & 0x80000000u;
    if (exp == 0u) {
        uint shift = clz(frac) - 8u;
        uint sig = frac << shift;
        uint m_bits = (sig & 0x007fffffu) | (127u << 23) | sign;
        return Soft{as_type<float>(m_bits), -126 - int(shift)};
    }
    uint m_bits = (bits & 0x807fffffu) | (127u << 23);
    return Soft{as_type<float>(m_bits), int(exp) - 127};
}

/// `mant` is a 24-bit significand. Shift it right by `rshift` and round to nearest even.
inline uint round_right_shift(uint mant, uint rshift) {
    if (rshift == 0u) {
        return mant;
    }
    if (rshift >= 25u) {
        return 0u;
    }
    uint shifted = mant >> rshift;
    uint rem = mant & ((1u << rshift) - 1u);
    uint midpoint = 1u << (rshift - 1u);
    if (rem > midpoint || (rem == midpoint && (shifted & 1u) == 1u)) {
        return shifted + 1u;
    }
    return shifted;
}

/// Pack a soft float to an `f32` bit pattern, including subnormals, signed zeros, and
/// overflow. The significand never passes through a subnormal float register.
inline uint soft_to_bits(Soft s) {
    uint raw = as_type<uint>(s.m);
    uint sign = raw & 0x80000000u;
    if ((raw & 0x7fffffffu) == 0u) {
        return sign;
    }
    if (s.e >= 128) {
        return sign | 0x7f800000u;
    }
    if (s.e >= -126) {
        uint biased = uint(s.e + 127);
        return (raw & 0x807fffffu) | (biased << 23);
    }
    uint rshift = uint(-s.e - 126);
    uint mant = (raw & 0x007fffffu) | 0x00800000u;
    uint m = round_right_shift(mant, rshift);
    if (m == 0u) {
        return sign;
    }
    if (m < (1u << 23)) {
        return sign | m;
    }
    uint frac = m - (1u << 23);
    if (frac < (1u << 23)) {
        return sign | (1u << 23) | frac;
    }
    return sign | (2u << 23) | (frac - (1u << 23));
}

inline Soft soft_neg(Soft a) {
    return Soft{as_type<float>(as_type<uint>(a.m) ^ 0x80000000u), a.e};
}

/// Product of two soft floats. A zero factor produces a signed zero. Both significands are
/// normal, and contraction is disabled for the whole shader.
inline Soft soft_mul(Soft a, Soft b) {
    if (a.m == 0.0f || b.m == 0.0f) {
        float z = a.m * b.m;
        return Soft{z, 0};
    }
    float p = a.m * b.m;
    if (fabs(p) >= 2.0f) {
        return Soft{p * 0.5f, a.e + b.e + 1};
    }
    return Soft{p, a.e + b.e};
}

/// Reciprocal of a nonzero soft float, delegated to the correctly rounded real reciprocal.
inline Soft soft_recip(Soft a) {
    float inv = as_type<float>(recip_f32_bits(as_type<uint>(a.m)));
    if (fabs(inv) < 1.0f) {
        return Soft{inv * 2.0f, -a.e - 1};
    }
    return Soft{inv, -a.e};
}

inline Soft soft_div(Soft a, Soft b) {
    return soft_mul(a, soft_recip(b));
}

/// Sum of two same-sign soft floats (Smith's denominator does not cancel). The larger
/// magnitude dominates; a gap of 25 bits or more drops the smaller term.
inline Soft soft_add(Soft a, Soft b) {
    if (b.m == 0.0f) {
        return a;
    }
    if (a.m == 0.0f) {
        return b;
    }
    if (a.e < b.e) {
        Soft swap = a;
        a = b;
        b = swap;
    }
    int diff = a.e - b.e;
    if (diff >= 25) {
        return a;
    }
    float scaled = b.m * as_type<float>(uint(127 - diff) << 23);
    float s = a.m + scaled;
    if (fabs(s) >= 2.0f) {
        return Soft{s * 0.5f, a.e + 1};
    }
    return Soft{s, a.e};
}

inline bool abs_ge(Soft a, Soft b) {
    bool az = (a.m == 0.0f);
    bool bz = (b.m == 0.0f);
    if (az) {
        return bz;
    }
    if (bz) {
        return true;
    }
    if (a.e != b.e) {
        return a.e > b.e;
    }
    return fabs(a.m) >= fabs(b.m);
}

/// Complex reciprocal in the style of C99 Annex G, then Smith's algorithm.
///
/// - An infinite part, even beside a NaN, makes `z` an infinity. The result is the zero
///   `(copysign(0, re), copysign(0, -im))`.
/// - Otherwise a NaN part returns a canonical quiet `(NaN, NaN)`.
/// - `1/(±0 ± 0i) = (copysign(∞, re), copysign(0, -im))`, and the same sign rule applies
///   when only one part is zero, so `1/(x + 0i)` matches the real reciprocal with an
///   imaginary part of `-0`.
/// - Otherwise, with `|a| >= |b|`: `r = b/a`, `d = a + b·r`, `z⁻¹ = (1/d, -r/d)`.
///   With `|b| > |a|`: `r = a/b`, `d = a·r + b`, `z⁻¹ = (r/d, -1/d)`.
///   The arithmetic runs on soft floats whose significands stay normal, so flush-to-zero
///   cannot erase a tiny part. Each division is the correctly rounded real reciprocal.
inline uint2 recip_c32_bits(uint2 z) {
    uint ab = z.x;
    uint bb = z.y;
    uint a_exp = (ab >> 23) & 0xffu;
    uint b_exp = (bb >> 23) & 0xffu;
    uint a_frac = ab & 0x007fffffu;
    uint b_frac = bb & 0x007fffffu;
    bool a_inf = (a_exp == 0xffu) && (a_frac == 0u);
    bool b_inf = (b_exp == 0xffu) && (b_frac == 0u);
    bool a_nan = (a_exp == 0xffu) && (a_frac != 0u);
    bool b_nan = (b_exp == 0xffu) && (b_frac != 0u);
    if (a_inf || b_inf) {
        return uint2(ab & 0x80000000u, (bb ^ 0x80000000u) & 0x80000000u);
    }
    if (a_nan || b_nan) {
        return uint2(0x7fc00000u, 0x7fc00000u);
    }
    // A zero part (including both zero) follows the real reciprocal and the `-0` rule.
    if ((bb & 0x7fffffffu) == 0u) {
        return uint2(recip_f32_bits(ab), (bb ^ 0x80000000u) & 0x80000000u);
    }
    if ((ab & 0x7fffffffu) == 0u) {
        return uint2(ab & 0x80000000u, recip_f32_bits(bb) ^ 0x80000000u);
    }

    Soft sa = soft_from_bits(ab);
    Soft sb = soft_from_bits(bb);
    if (abs_ge(sa, sb)) {
        Soft r = soft_div(sb, sa);
        Soft d = soft_add(sa, soft_mul(sb, r));
        Soft inv_d = soft_recip(d);
        uint re = soft_to_bits(inv_d);
        uint im = soft_to_bits(soft_mul(soft_neg(r), inv_d));
        return uint2(re, im);
    }
    Soft r = soft_div(sa, sb);
    Soft d = soft_add(sb, soft_mul(sa, r));
    Soft inv_d = soft_recip(d);
    uint re = soft_to_bits(soft_mul(r, inv_d));
    uint im = soft_to_bits(soft_neg(inv_d));
    return uint2(re, im);
}

/// Element index of linear id `tid` in row-major order over `shape`.
inline void locate(constant RecipParams& params, uint tid, thread uint& src_i, thread uint& dst_i) {
    src_i = params.src_offset;
    dst_i = params.dst_offset;
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
}

/// Elementwise real reciprocal. One thread owns one logical element.
///
/// Logical index `tid` is unraveled in row-major order into `shape` and then applied to
/// `src_strides` and `dst_strides`. Rank 0 is the single element at the two offsets.
/// Buffers are `uint` so a subnormal bit pattern survives the load and the store.
kernel void reciprocal_f32 [[host_name("reciprocal_f32")]] (
    device const uint* src [[buffer(0)]],
    device uint* dst [[buffer(1)]],
    constant RecipParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.count) {
        return;
    }
    uint src_i;
    uint dst_i;
    locate(params, tid, src_i, dst_i);
    dst[dst_i] = recip_f32_bits(src[src_i]);
}

/// Elementwise complex reciprocal. One thread owns one `float2` element.
///
/// Indexing matches `reciprocal_f32`. The buffer is read and written as `uint2` so each
/// component's bits, including subnormals and signed zeros, bypass flush-to-zero.
kernel void reciprocal_c32 [[host_name("reciprocal_c32")]] (
    device const uint2* src [[buffer(0)]],
    device uint2* dst [[buffer(1)]],
    constant RecipParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.count) {
        return;
    }
    uint src_i;
    uint dst_i;
    locate(params, tid, src_i, dst_i);
    dst[dst_i] = recip_c32_bits(src[src_i]);
}
