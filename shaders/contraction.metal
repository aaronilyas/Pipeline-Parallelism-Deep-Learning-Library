// Tensor contraction: C = α·A·B + β·C.
//
// Axes arrive already grouped into batch / M / N / K. A group is either one stride
// (the tiled kernels) or up to four strides, fastest axis last in `Axes` (the general
// kernels). Both element types accumulate in `simdgroup_matrix<float, 8, 8>`: f16 operands
// are widened exactly to f32 while they are staged into threadgroup memory, and the result
// is rounded to f16 once, at the store.
//
// Tiled kernels: each threadgroup owns a 64×64 output tile and streams K in chunks of
// 32 through padded threadgroup memory, then `simdgroup_multiply_accumulate`. Tiles that
// hang off M, N, or K are zero-filled on the way in and predicate the store. β = 0 does
// not load C.
//
// Split-K (small M·N, long K): several threadgroups split K at fixed chunk boundaries
// and write raw f32 partials. A later kernel adds those partials in split order and
// applies α and β once. No atomics touch the sum, so it is bit-stable.
//
// Subnormals. This GPU treats a subnormal float operand as zero and flushes a subnormal
// float result to zero, in scalar arithmetic and inside `simdgroup_multiply_accumulate`
// alike (pinned by `probe_subnormal_flush_is_pinned` in the tests). Loads, stores,
// selects, and integer operations keep the bits. Two mechanisms keep that out of C:
//
// 1. Epilogue. `α·acc + β·C` and `β·C` are evaluated by `combine_f32`. Its fast path is
//    plain float arithmetic and is taken only when bit tests prove that no operand,
//    intermediate, or result was subnormal. Otherwise the operands are split into a
//    significand in [1, 2) and an integer exponent, combined where every float is normal,
//    and packed (with round-to-nearest-even into the subnormal range) by integer shifts.
// 2. Accumulation (f32 only; f16 products are never below 2⁻⁴⁸). `contraction_scan_f32`
//    first reads A and B once and records the largest and the smallest nonzero magnitude
//    of each. If an operand is subnormal, or the smallest possible product is below
//    2⁻¹⁰³, a flush could exceed the accuracy gate. The accumulating kernels then stage
//    A and B scaled up by exact powers of two (as far as the largest product allows
//    without overflow) and the epilogue removes the scale in integer arithmetic.
//    Products that are still below 2⁻¹²⁶ after that scaling, which needs magnitudes
//    spanning more than about 2²⁰⁰ within one contraction, are flushed.
//
// The scan is a separate pass because the tiled kernel is bound by its device reads:
// tracking magnitudes inside its staging loop costs about 40% of its throughput, and
// re-reading a tile's operands costs more.

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

// Must match the host dispatch in `src/ops/contraction.rs`.
constexpr constant uint BM = 64;
constexpr constant uint BN = 64;
constexpr constant uint BK = 32;
constexpr constant uint WM = 4; // simdgroups along M
constexpr constant uint WN = 4; // simdgroups along N
constexpr constant uint TM = 2; // 8×8 fragments a simdgroup owns along M
constexpr constant uint TN = 2; // 8×8 fragments a simdgroup owns along N
constexpr constant uint THREADS = WM * WN * 32;
constexpr constant uint SA_STRIDE = BK + 4; // pad so 8-wide rows miss threadgroup banks
constexpr constant uint SB_STRIDE = BN + 4;
constexpr constant uint SA_ELEMS = BM * SA_STRIDE;
constexpr constant uint SB_ELEMS = BK * SB_STRIDE;

// Right-aligned mixed radix: slot 3 is the fastest axis, unused slots have extent 1.
struct Axes {
    uint ext[4];
    uint stride[4];
};

struct TiledParams {
    uint a_offset;
    uint b_offset;
    uint c_offset;
    uint m;
    uint n;
    uint k;
    uint batch;
    uint a_batch_stride;
    uint a_m_stride;
    uint a_k_stride;
    uint b_batch_stride;
    uint b_n_stride;
    uint b_k_stride;
    uint c_batch_stride;
    uint c_m_stride;
    uint c_n_stride;
    float alpha;
    float beta;
    uint k_chunk;
    uint splits;
    uint vector_ok;
};

struct GeneralParams {
    uint a_offset;
    uint b_offset;
    uint c_offset;
    uint m;
    uint n;
    uint k;
    uint batch;
    float alpha;
    float beta;
    Axes a_batch;
    Axes a_m;
    Axes a_k;
    Axes b_batch;
    Axes b_n;
    Axes b_k;
    Axes c_batch;
    Axes c_m;
    Axes c_n;
};

struct ScaleParams {
    uint c_offset;
    uint count;
    uint m;
    uint n;
    uint batch;
    float beta;
    Axes c_batch;
    Axes c_m;
    Axes c_n;
};

// One operand's elements as `outer × inner`: `outer` is batch × mid, and `inner` is the
// group the scan walks. `slot` is the operand's first index in the magnitude record.
struct ScanParams {
    uint offset;
    uint slot;
    uint outer_count;
    uint mid_count;
    uint inner_count;
    Axes batch;
    Axes mid;
    Axes inner;
};

struct ReduceParams {
    uint c_offset;
    uint count;
    uint m;
    uint n;
    uint batch;
    uint splits;
    uint k;
    float alpha;
    float beta;
    Axes c_batch;
    Axes c_m;
    Axes c_n;
};

// Linear index → element offset. Extents are never zero (unused axes are extent 1).
inline uint decode(uint linear, Axes ax) {
    uint t = linear;
    uint i3 = t % ax.ext[3];
    t /= ax.ext[3];
    uint i2 = t % ax.ext[2];
    t /= ax.ext[2];
    uint i1 = t % ax.ext[1];
    t /= ax.ext[1];
    uint i0 = t % ax.ext[0];
    return i0 * ax.stride[0] + i1 * ax.stride[1] + i2 * ax.stride[2] + i3 * ax.stride[3];
}

inline void unpack_mn(uint tid, uint m, uint n, thread uint& b, thread uint& row, thread uint& col) {
    col = tid % n;
    uint t = tid / n;
    row = t % m;
    b = t / m;
}

// --- Subnormal-safe scalar helpers -------------------------------------------------------

// Exact 2^e as a float, for e in [-126, 127].
inline float pow2(int e) {
    return as_type<float>(uint(e + 127) << 23);
}

// Magnitude bits. Integer tests on these never see the float unit's subnormal-as-zero rule.
inline uint mag_bits(float x) {
    return as_type<uint>(x) & 0x7fffffffu;
}

inline bool is_zero_bits(float x) {
    return mag_bits(x) == 0u;
}

// True for ±0 and for normal, infinite, and NaN values: everything but a subnormal.
inline bool not_subnormal(float x) {
    uint mag = mag_bits(x);
    return mag == 0u || mag >= 0x00800000u;
}

// floor(log2 |x|) from the magnitude bits of a nonzero value. Infinity and NaN give 128.
inline int exponent_of(uint mag) {
    uint e = mag >> 23;
    if (e == 0u) {
        return int(31u - clz(mag)) - 149;
    }
    return int(e) - 127;
}

// x·2^s for s in [0, 127], exact whenever the result is normal. A subnormal x is
// normalized with integer arithmetic first, because a float multiply would read it as zero.
inline float scale_up(float x, int s) {
    uint bits = as_type<uint>(x);
    uint mag = bits & 0x7fffffffu;
    if (mag == 0u || mag >= 0x00800000u) {
        return x * pow2(s);
    }
    // |x| = mag·2^-149. Two steps keep the intermediate normal.
    int e = s - 149;
    int first = e / 2;
    float scaled = (float(mag) * pow2(first)) * pow2(e - first);
    return (bits >> 31) != 0u ? -scaled : scaled;
}

// Four elements of `scale_up`. The multiply is exact for normal and zero lanes; subnormal
// lanes are rare and redo the vector element by element.
inline float4 scale_up4(float4 x, int s) {
    uint4 mag = as_type<uint4>(x) & 0x7fffffffu;
    // The wrapping subtraction sends zero above the subnormal range.
    if (any(mag - 1u < uint4(0x007fffffu))) {
        return float4(scale_up(x.x, s), scale_up(x.y, s), scale_up(x.z, s), scale_up(x.w, s));
    }
    return x * pow2(s);
}

// Error-free transform: s + e = a + b exactly (no overflow).
inline void two_sum(float a, float b, thread float& s, thread float& e) {
    s = a + b;
    float v = s - a;
    e = (a - (s - v)) + (b - v);
}

// p + e = a * b exactly, while p and e are normal. Fast math is off, so `fma` is a real
// fused multiply-add.
inline void two_prod(float a, float b, thread float& p, thread float& e) {
    p = a * b;
    e = fma(a, b, -p);
}

// A finite float as `m·2^e` with |m| in [1, 2), or m = ±0. A subnormal is normalized with
// integer arithmetic, so the float unit only ever sees `m`.
struct Soft {
    float m;
    int e;
};

inline Soft soft_from(float x) {
    uint bits = as_type<uint>(x);
    uint sign = bits & 0x80000000u;
    uint mag = bits & 0x7fffffffu;
    Soft out;
    if (mag == 0u) {
        out.m = x;
        out.e = 0;
        return out;
    }
    int exp = int(mag >> 23);
    if (exp == 0) {
        uint shift = clz(mag) - 8u;
        mag <<= shift;
        exp = 1 - int(shift);
    }
    out.m = as_type<float>(sign | 0x3f800000u | (mag & 0x007fffffu));
    out.e = exp - 127;
    return out;
}

// Bits of round-to-nearest-even((hi + lo)·2^e). `hi` is zero or normal and is already the
// rounding of `hi + lo` to 24 bits. Above the subnormal range that is the answer. Below
// it the significand is shifted in integer arithmetic and `lo` breaks the tie it would
// otherwise hide, so the value is rounded once.
inline uint soft_pack(float hi, float lo, int e) {
    uint bits = as_type<uint>(hi);
    uint sign = bits & 0x80000000u;
    uint mag = bits & 0x7fffffffu;
    if (mag == 0u) {
        return sign;
    }
    int exp = int(mag >> 23) - 127 + e;
    if (exp > 127) {
        return sign | 0x7f800000u;
    }
    if (exp >= -126) {
        return sign | (uint(exp + 127) << 23) | (mag & 0x007fffffu);
    }
    int shift = -126 - exp;
    // Below half the smallest subnormal.
    if (shift > 25) {
        return sign;
    }
    uint sig = (mag & 0x007fffffu) | 0x00800000u;
    uint kept = sig >> uint(shift);
    uint dropped = sig & ((1u << uint(shift)) - 1u);
    uint halfway = 1u << uint(shift - 1);
    uint lo_bits = as_type<uint>(lo);
    bool lo_nonzero = (lo_bits & 0x7fffffffu) != 0u;
    bool lo_away = lo_nonzero && (lo_bits & 0x80000000u) == sign;
    bool lo_toward = lo_nonzero && !lo_away;
    bool up = dropped > halfway ||
        (dropped == halfway && (lo_away || (!lo_toward && (kept & 1u) != 0u)));
    return sign | (kept + (up ? 1u : 0u));
}

// Bits of `α·(acc·2^-scale) + β·c` (or of `α·(acc·2^-scale)` when `use_c` is false) with
// every float operation kept in the normal range. A lone product is rounded once. The sum
// rounds `β·c`, then the fused sum, like the float path; a subnormal result is then
// rounded once more by `soft_pack`, which costs at most half the smallest subnormal.
inline uint combine_soft(float alpha, float acc, int scale, float beta, float c, bool use_c) {
    bool finite = isfinite(alpha) && isfinite(acc) && (!use_c || (isfinite(beta) && isfinite(c)));
    if (!finite) {
        // Infinity and NaN propagate through plain float arithmetic.
        float out = alpha * acc;
        if (use_c) {
            out += beta * c;
        }
        return as_type<uint>(out);
    }
    Soft a = soft_from(alpha);
    Soft x = soft_from(acc);
    float p1, l1;
    two_prod(a.m, x.m, p1, l1);
    int e1 = a.e + x.e - scale;
    if (!use_c) {
        return soft_pack(p1, l1, e1);
    }
    Soft b = soft_from(beta);
    Soft y = soft_from(c);
    float p2, l2;
    two_prod(b.m, y.m, p2, l2);
    int e2 = b.e + y.e;
    if (is_zero_bits(p1)) {
        return is_zero_bits(p2) ? as_type<uint>(p1 + p2) : soft_pack(p2, l2, e2);
    }
    if (is_zero_bits(p2)) {
        return soft_pack(p1, l1, e1);
    }
    // Align both terms to the larger exponent. A term more than 2^100 smaller cannot
    // change the rounded sum, so its shift is clamped to stay normal.
    int top = max(e1, e2);
    float t1 = a.m * pow2(max(e1 - top, -100));
    float t2 = p2 * pow2(max(e2 - top, -100));
    return soft_pack(fma(t1, x.m, t2), 0.0f, top);
}

// Bits of the f32 epilogue `α·(acc·2^-scale) + β·c`. The float fast path is used only
// when bit tests show that nothing in it was subnormal: the operands, `β·c`, and the
// result are each normal, non-finite, or a zero that follows from a zero operand.
inline uint combine_f32(float alpha, float acc, int scale, float beta, float c, bool use_c) {
    if (scale == 0 && not_subnormal(alpha) && not_subnormal(acc)) {
        float t2 = 0.0f;
        bool t2_exact = true;
        if (use_c) {
            t2 = beta * c;
            t2_exact = not_subnormal(beta) &&
                (mag_bits(t2) >= 0x00800000u || is_zero_bits(c));
        }
        // Without C the product keeps its own sign of zero.
        float out = use_c ? fma(alpha, acc, t2) : alpha * acc;
        bool out_exact = mag_bits(out) >= 0x00800000u ||
            ((is_zero_bits(acc) || is_zero_bits(alpha)) && is_zero_bits(t2));
        if (t2_exact && out_exact) {
            return as_type<uint>(out);
        }
    }
    return combine_soft(alpha, acc, scale, beta, c, use_c);
}

// Round-to-nearest-even f32 → f16 bits, including subnormals and the overflow tie.
// Kept as bits: MSL `exp2` may miss a power of two by one ulp, and that miss
// turns an exact f16 tie into the odd neighbor.
inline ushort f32_to_f16_bits(float value) {
    uint bits = as_type<uint>(value);
    uint sign = (bits >> 16) & 0x8000u;
    uint rest = bits & 0x7fffffffu;
    if (rest > 0x7f800000u) {
        return ushort(sign | 0x7e00u);
    }
    if (rest == 0u) {
        return ushort(sign);
    }
    int unbiased = int((bits >> 23) & 0xffu) - 127;
    uint mant = bits & 0x7fffffu;
    if (unbiased > 15) {
        return ushort(sign | 0x7c00u);
    }
    if (unbiased < -14) {
        // Smaller than half the tiniest subnormal: underflow to zero.
        if (unbiased < -25) {
            return ushort(sign);
        }
        uint sig = mant | 0x800000u;
        int shift = -unbiased - 1;
        uint kept = sig >> uint(shift);
        uint dropped = sig & ((1u << uint(shift)) - 1u);
        uint halfway = 1u << uint(shift - 1);
        if (dropped > halfway || (dropped == halfway && (kept & 1u) != 0u)) {
            kept += 1u;
        }
        return ushort(sign | kept);
    }
    int exp16 = unbiased + 15;
    uint kept = mant >> 13;
    uint dropped = mant & 0x1fffu;
    if (dropped > 0x1000u || (dropped == 0x1000u && (kept & 1u) != 0u)) {
        kept += 1u;
        if (kept == 0x400u) {
            kept = 0u;
            exp16 += 1;
            if (exp16 >= 31) {
                return ushort(sign | 0x7c00u);
            }
        }
    }
    return ushort(sign | (uint(exp16) << 10) | kept);
}

// Exact f16 → f32. Every finite half fits in an f32.
inline float f16_bits_to_f32(ushort bits) {
    uint sign = uint(bits & 0x8000u) << 16;
    uint exp = (uint(bits) >> 10) & 0x1fu;
    uint mant = uint(bits) & 0x3ffu;
    if (exp == 0u) {
        float mag = float(mant) * as_type<float>(0x33800000u); // 2^-24
        return as_type<float>(sign | as_type<uint>(mag));
    }
    if (exp == 31u) {
        return as_type<float>(sign | 0x7f800000u | (mant << 13));
    }
    uint fexp = exp + 112u; // 127 - 15
    return as_type<float>(sign | (fexp << 23) | (mant << 13));
}

// ulp of a finite f16, built as an exact power of two.
inline float f16_ulp_bits(ushort bits) {
    uint mag = uint(bits) & 0x7fffu;
    if (mag < 0x0400u) {
        return as_type<float>(0x33800000u); // 2^-24
    }
    int ulp_unb = int(mag >> 10) - 25;
    return as_type<float>(uint(ulp_unb + 127) << 23);
}

inline ushort f16_step_bits(ushort b, bool toward_pos_inf) {
    bool neg = (b & 0x8000u) != 0u;
    bool inc = toward_pos_inf != neg;
    if (inc) {
        b += 1u;
    } else if ((b & 0x7fffu) == 0u) {
        b = toward_pos_inf ? ushort(1) : ushort(0x8001u);
    } else {
        b -= 1u;
    }
    return b;
}

// Round hi+lo to f16. Collapsing to one f32 first drops the bits that decide a
// near-tie. A normal power of two (not the minimum normal) has a half-size ulp
// toward zero; every other finite half has symmetric neighbors.
inline half dd_to_half(float hi, float lo) {
    if (!isfinite(hi) || !isfinite(lo)) {
        return half(hi + lo);
    }
    ushort zbits = f32_to_f16_bits(hi);
    if ((zbits & 0x7c00u) == 0x7c00u) {
        return as_type<half>(zbits);
    }
    float zf = f16_bits_to_f32(zbits);
    float rhi, rlo;
    two_sum(hi - zf, lo, rhi, rlo);
    float ulp = f16_ulp_bits(zbits);
    float half_away = ulp * 0.5f;
    uint exp = (uint(zbits) >> 10) & 0x1fu;
    bool pow2 = (zbits & 0x03ffu) == 0u && exp >= 2u;
    float half_toward = pow2 ? ulp * 0.25f : half_away;
    bool neg = (zbits & 0x8000u) != 0u;
    float up_lim = neg ? half_toward : half_away;
    float down_lim = neg ? half_away : half_toward;
    bool even = (zbits & 1u) == 0u;
    bool up_tie = rhi == up_lim && rlo == 0.0f;
    bool down_tie = rhi == -down_lim && rlo == 0.0f;
    bool past = rhi > up_lim || (rhi == up_lim && rlo > 0.0f);
    bool before = rhi < -down_lim || (rhi == -down_lim && rlo < 0.0f);
    if (past || (up_tie && !even)) {
        zbits = f16_step_bits(zbits, true);
    } else if (before || (down_tie && !even)) {
        zbits = f16_step_bits(zbits, false);
    }
    return as_type<half>(zbits);
}

// C[index] = α·(acc·2^-scale) + β·C[index] for f32, through the subnormal-safe epilogue.
// β = 0 (tested on bits) does not load C.
inline void store_result(
    device float* C, uint index, float alpha, float acc, int scale, float beta) {
    bool use_c = !is_zero_bits(beta);
    float c = 0.0f;
    if (use_c) {
        c = C[index];
    }
    C[index] = as_type<float>(combine_f32(alpha, acc, scale, beta, c, use_c));
}

// C[index] = α·acc + β·C[index] for f16: evaluated in f32, rounded to half once. `scale`
// is always 0 here. Every f16 value is at least 2⁻²⁴ as a float, so an f32 flush is below
// 2⁻¹²⁶ and cannot move the f16 result. β = 0 does not load C.
inline void store_result(
    device half* C, uint index, float alpha, float acc, int scale, float beta) {
    float out = alpha * acc;
    if (!is_zero_bits(beta)) {
        out = fma(alpha, acc, beta * float(C[index]));
    }
    C[index] = as_type<half>(f32_to_f16_bits(out));
}

// --- Operand magnitudes and the rescale decision -----------------------------------------

// A product of nonzero operands is at least 2^(SAFE_PRODUCT_EXP) when the exponents of the
// smallest operands add up to this. A flush then loses less than 2⁻¹²⁶, which is under
// 3·2⁻²⁴ of one product and therefore inside the (K+2)·u·Σ|a||b| accuracy gate.
constexpr constant int SAFE_PRODUCT_EXP = -103;

// Slots of the magnitude record that `contraction_scan_f32` fills. A `MIN` slot holds
// the smallest nonzero magnitude minus one, or 0xffffffff when every element is zero.
// The host initializes the record to {0, 0xffffffff, 0, 0xffffffff}.
constexpr constant uint STAT_A_MAX = 0;
constexpr constant uint STAT_A_MIN = 1;
constexpr constant uint STAT_B_MAX = 2;
constexpr constant uint STAT_B_MIN = 3;

// Elements of the inner group one scan thread reads.
constexpr constant uint SCAN_CHUNK = 256;

inline uint max4(uint4 v) {
    return max(max(v.x, v.y), max(v.z, v.w));
}

inline uint min4(uint4 v) {
    return min(min(v.x, v.y), min(v.z, v.w));
}

// Power-of-two scales (sa, sb) for A and B, or (0, 0) when no operand is subnormal and
// no product can be flushed harmfully. With |a| < 2^(ea+1) and |b| < 2^(eb+1), a sum of
// `k` products is below 2^(ea + eb + 2 + ⌈log2 k⌉), so the total scale
// 125 - ⌈log2 k⌉ - ea - eb keeps every scaled partial sum, and every sum of Split-K
// partials, below 2^127. Each scale is non-negative and keeps its own operand finite, so
// scaling never overflows or underflows an element. Every thread of a contraction
// derives the same scales from the same record.
inline int2 plan_scale(device const uint* stats, uint k) {
    uint a_min = stats[STAT_A_MIN];
    uint b_min = stats[STAT_B_MIN];
    if (a_min == 0xffffffffu || b_min == 0xffffffffu) {
        return int2(0);
    }
    bool subnormal_operand = a_min < 0x007fffffu || b_min < 0x007fffffu;
    bool safe_products =
        exponent_of(a_min + 1u) + exponent_of(b_min + 1u) >= SAFE_PRODUCT_EXP;
    if (safe_products && !subnormal_operand) {
        return int2(0);
    }
    int ea = exponent_of(stats[STAT_A_MAX]);
    int eb = exponent_of(stats[STAT_B_MAX]);
    int log_k = k > 1u ? int(32u - clz(k - 1u)) : 0;
    int total = 125 - log_k - ea - eb;
    int sa = clamp(min(total, 127 - ea), 0, 127);
    int sb = clamp(min(total - sa, 127 - eb), 0, 127);
    return int2(sa, sb);
}

// --- Kernel bodies -----------------------------------------------------------------------

// The accumulators of one simdgroup: TM×TN fragments of 8×8.
struct TileAcc {
    simdgroup_matrix<float, 8, 8> frag[TM][TN];
};

// Where one threadgroup of the tiled kernels works: its tile origin, K slice, operand
// bases, and this thread's place in it.
struct TileSpan {
    uint m0;
    uint n0;
    uint k_begin;
    uint k_end;
    uint a_base;
    uint b_base;
    uint tid;
    uint sg_row;
    uint sg_col;
};

// acc = A·B over the K slice of one 64×64 tile. Device operands stay in T. Threadgroup
// tiles are float, so the 8-wide dot accumulates in f32 for both f32 and f16 inputs (every
// f16 value is exact in f32). `Scaled` stages A·2^scale.x and B·2^scale.y instead.
template <typename T, bool Scaled>
inline void tiled_accumulate(
    device const T* A,
    device const T* B,
    constant TiledParams& p,
    threadgroup float* sA,
    threadgroup float* sB,
    TileSpan t,
    int2 scale,
    thread TileAcc& acc) {
    for (uint i = 0; i < TM; ++i) {
        for (uint j = 0; j < TN; ++j) {
            acc.frag[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    for (uint k_block = t.k_begin; k_block < t.k_end; k_block += BK) {
        const bool full = p.vector_ok != 0 && (t.m0 + BM <= p.m) && (t.n0 + BN <= p.n) &&
            (k_block + BK <= t.k_end);
        if (full) {
            // Contiguous, aligned inner edges: one vec4 per four elements.
            constexpr uint VEC_A = (BM * BK) / (THREADS * 4);
            for (uint i = 0; i < VEC_A; ++i) {
                const uint idx = i * THREADS + t.tid;
                const uint elem = idx * 4;
                const uint row = elem / BK;
                const uint col = elem % BK;
                const device T* ptr =
                    A + (t.a_base + (t.m0 + row) * p.a_m_stride + k_block + col);
                vec<T, 4> v = *reinterpret_cast<const device vec<T, 4>*>(ptr);
                float4 f = float4(float(v[0]), float(v[1]), float(v[2]), float(v[3]));
                if (Scaled) {
                    f = scale_up4(f, scale.x);
                }
                *reinterpret_cast<threadgroup float4*>(sA + row * SA_STRIDE + col) = f;
            }
            constexpr uint VEC_B = (BK * BN) / (THREADS * 4);
            for (uint i = 0; i < VEC_B; ++i) {
                const uint idx = i * THREADS + t.tid;
                const uint elem = idx * 4;
                const uint row = elem / BN;
                const uint col = elem % BN;
                const device T* ptr = B +
                    (t.b_base + (k_block + row) * p.b_k_stride + (t.n0 + col) * p.b_n_stride);
                vec<T, 4> v = *reinterpret_cast<const device vec<T, 4>*>(ptr);
                float4 f = float4(float(v[0]), float(v[1]), float(v[2]), float(v[3]));
                if (Scaled) {
                    f = scale_up4(f, scale.y);
                }
                *reinterpret_cast<threadgroup float4*>(sB + row * SB_STRIDE + col) = f;
            }
        } else {
            constexpr uint EPT_A = (BM * BK) / THREADS;
            for (uint i = 0; i < EPT_A; ++i) {
                const uint idx = i * THREADS + t.tid;
                const uint row = idx / BK;
                const uint col = idx % BK;
                const uint gm = t.m0 + row;
                const uint gk = k_block + col;
                float v = 0.0f;
                if (gm < p.m && gk < t.k_end) {
                    v = float(A[t.a_base + gm * p.a_m_stride + gk * p.a_k_stride]);
                }
                if (Scaled) {
                    v = scale_up(v, scale.x);
                }
                sA[row * SA_STRIDE + col] = v;
            }
            constexpr uint EPT_B = (BK * BN) / THREADS;
            for (uint i = 0; i < EPT_B; ++i) {
                const uint idx = i * THREADS + t.tid;
                const uint row = idx / BN;
                const uint col = idx % BN;
                const uint gk = k_block + row;
                const uint gn = t.n0 + col;
                float v = 0.0f;
                if (gk < t.k_end && gn < p.n) {
                    v = float(B[t.b_base + gn * p.b_n_stride + gk * p.b_k_stride]);
                }
                if (Scaled) {
                    v = scale_up(v, scale.y);
                }
                sB[row * SB_STRIDE + col] = v;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Fragment predicates are uniform inside a simdgroup (they depend on `sgid`).
        // A full simdgroup tile loads each A and B fragment once per K step.
        const bool sg_full = (t.m0 + (t.sg_row + 1) * (TM * 8)) <= p.m &&
            (t.n0 + (t.sg_col + 1) * (TN * 8)) <= p.n;
        if (sg_full) {
            for (uint kk = 0; kk < BK; kk += 8) {
                simdgroup_matrix<float, 8, 8> a_frag[TM];
                simdgroup_matrix<float, 8, 8> b_frag[TN];
                for (uint i = 0; i < TM; ++i) {
                    const uint row = t.sg_row * (TM * 8) + i * 8;
                    simdgroup_load(a_frag[i], sA + row * SA_STRIDE + kk, SA_STRIDE);
                }
                for (uint j = 0; j < TN; ++j) {
                    const uint col = t.sg_col * (TN * 8) + j * 8;
                    simdgroup_load(b_frag[j], sB + kk * SB_STRIDE + col, SB_STRIDE);
                }
                for (uint i = 0; i < TM; ++i) {
                    for (uint j = 0; j < TN; ++j) {
                        simdgroup_multiply_accumulate(
                            acc.frag[i][j], a_frag[i], b_frag[j], acc.frag[i][j]);
                    }
                }
            }
        } else if ((t.m0 + t.sg_row * (TM * 8)) < p.m && (t.n0 + t.sg_col * (TN * 8)) < p.n) {
            for (uint kk = 0; kk < BK; kk += 8) {
                for (uint i = 0; i < TM; ++i) {
                    for (uint j = 0; j < TN; ++j) {
                        const uint row = t.sg_row * (TM * 8) + i * 8;
                        const uint col = t.sg_col * (TN * 8) + j * 8;
                        if (t.m0 + row < p.m && t.n0 + col < p.n) {
                            simdgroup_matrix<float, 8, 8> a_frag;
                            simdgroup_matrix<float, 8, 8> b_frag;
                            simdgroup_load(a_frag, sA + row * SA_STRIDE + kk, SA_STRIDE);
                            simdgroup_load(b_frag, sB + kk * SB_STRIDE + col, SB_STRIDE);
                            simdgroup_multiply_accumulate(
                                acc.frag[i][j], a_frag, b_frag, acc.frag[i][j]);
                        }
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// One threadgroup of the tiled kernels: accumulate a 64×64 tile, then either store
// `α·acc + β·C` or (Split-K) write the raw partial.
//
// `Rescale` (f32) reads the scales from the magnitude record. They are the same for every
// thread of the contraction, so the branch on them is uniform and every Split-K partial
// is at the same scale. The unscaled loop is kept free of any magnitude bookkeeping.
template <typename T, bool WritePartial, bool Rescale>
inline void contraction_tiled_body(
    device const T* A,
    device const T* B,
    device T* C,
    device float* partial,
    device const uint* stats,
    constant TiledParams& p,
    threadgroup float* sA,
    threadgroup float* sB,
    threadgroup float* spill,
    uint3 tgid,
    uint tid,
    uint sgid,
    uint lane) {
    uint batch_id;
    uint split;
    TileSpan t;
    if (WritePartial) {
        split = tgid.z / p.batch;
        batch_id = tgid.z - split * p.batch;
        t.k_begin = split * p.k_chunk;
        t.k_end = min(t.k_begin + p.k_chunk, p.k);
    } else {
        split = 0;
        batch_id = tgid.z;
        t.k_begin = 0;
        t.k_end = p.k;
    }
    // `tgid` is uniform across the threadgroup, so this return is uniform.
    if (t.k_begin >= t.k_end) {
        return;
    }

    t.m0 = tgid.y * BM;
    t.n0 = tgid.x * BN;
    t.tid = tid;
    t.sg_row = sgid / WN;
    t.sg_col = sgid % WN;
    t.a_base = p.a_offset + batch_id * p.a_batch_stride;
    t.b_base = p.b_offset + batch_id * p.b_batch_stride;
    const uint c_base = p.c_offset + batch_id * p.c_batch_stride;

    int2 scale = int2(0);
    if (Rescale) {
        scale = plan_scale(stats, p.k);
    }
    const int total_scale = scale.x + scale.y;
    TileAcc acc;
    if (total_scale != 0) {
        tiled_accumulate<T, true>(A, B, p, sA, sB, t, scale, acc);
    } else {
        tiled_accumulate<T, false>(A, B, p, sA, sB, t, scale, acc);
    }

    const uint plane_stride = p.batch * p.m * p.n;
    for (uint i = 0; i < TM; ++i) {
        for (uint j = 0; j < TN; ++j) {
            const uint row = t.sg_row * (TM * 8) + i * 8;
            const uint col = t.sg_col * (TN * 8) + j * 8;
            if (t.m0 + row >= p.m || t.n0 + col >= p.n) {
                continue;
            }
            // `simdgroup_barrier` waits only for this simdgroup, so the predicate above
            // (uniform per simdgroup, divergent across them) is safe.
            threadgroup float* slot = spill + sgid * 64;
            simdgroup_store(acc.frag[i][j], slot, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64; e += 32) {
                const uint r = e >> 3;
                const uint c = e & 7;
                const uint gm = t.m0 + row + r;
                const uint gn = t.n0 + col + c;
                if (gm < p.m && gn < p.n) {
                    float accv = slot[r * 8 + c];
                    if (WritePartial) {
                        uint plane = (batch_id * p.m + gm) * p.n + gn;
                        partial[split * plane_stride + plane] = accv;
                    } else {
                        uint index = c_base + gm * p.c_m_stride + gn * p.c_n_stride;
                        store_result(C, index, p.alpha, accv, total_scale, p.beta);
                    }
                }
            }
            // The next fragment reuses `slot`; every lane must have read it first.
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// acc = A·B for one 8×8 output tile of the general kernels, staging 8 K steps at a time.
// `Scaled` is as in `tiled_accumulate`.
template <typename T, bool Scaled>
inline void general_accumulate(
    device const T* A,
    device const T* B,
    constant GeneralParams& p,
    threadgroup float* a_tile,
    threadgroup float* b_tile,
    uint3 origin,
    uint lane,
    int2 scale,
    thread simdgroup_matrix<float, 8, 8>& acc) {
    const uint a_batch = p.a_offset + decode(origin.z, p.a_batch);
    const uint b_batch = p.b_offset + decode(origin.z, p.b_batch);
    acc = simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = 0; k0 < p.k; k0 += 8) {
        for (uint e = lane; e < 64; e += 32) {
            const uint r = e >> 3;
            const uint c = e & 7;
            // A tile is M × K, B tile is K × N, both row-major in the 8×8 scratch.
            const uint gm = origin.y + r;
            const uint gk_a = k0 + c;
            float va = 0.0f;
            if (gm < p.m && gk_a < p.k) {
                va = float(A[a_batch + decode(gm, p.a_m) + decode(gk_a, p.a_k)]);
            }
            const uint gk_b = k0 + r;
            const uint gn = origin.x + c;
            float vb = 0.0f;
            if (gk_b < p.k && gn < p.n) {
                vb = float(B[b_batch + decode(gn, p.b_n) + decode(gk_b, p.b_k)]);
            }
            if (Scaled) {
                va = scale_up(va, scale.x);
                vb = scale_up(vb, scale.y);
            }
            a_tile[e] = va;
            b_tile[e] = vb;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_matrix<float, 8, 8> a_frag;
        simdgroup_matrix<float, 8, 8> b_frag;
        simdgroup_load(a_frag, a_tile, 8);
        simdgroup_load(b_frag, b_tile, 8);
        simdgroup_multiply_accumulate(acc, a_frag, b_frag, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// One simdgroup computes an 8×8 output tile and stores `α·acc + β·C`. `Rescale` is as in
// `contraction_tiled_body`.
template <typename T, bool Rescale>
inline void contraction_general_body(
    device const T* A,
    device const T* B,
    device T* C,
    device const uint* stats,
    constant GeneralParams& p,
    threadgroup float* a_tile,
    threadgroup float* b_tile,
    threadgroup float* spill,
    uint3 tgid,
    uint lane) {
    const uint3 origin = uint3(tgid.x * 8, tgid.y * 8, tgid.z);
    if (origin.y >= p.m || origin.x >= p.n || origin.z >= p.batch) {
        return;
    }
    int2 scale = int2(0);
    if (Rescale) {
        scale = plan_scale(stats, p.k);
    }
    const int total_scale = scale.x + scale.y;
    simdgroup_matrix<float, 8, 8> acc;
    if (total_scale != 0) {
        general_accumulate<T, true>(A, B, p, a_tile, b_tile, origin, lane, scale, acc);
    } else {
        general_accumulate<T, false>(A, B, p, a_tile, b_tile, origin, lane, scale, acc);
    }

    const uint c_batch = p.c_offset + decode(origin.z, p.c_batch);
    simdgroup_store(acc, spill, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < 64; e += 32) {
        const uint r = e >> 3;
        const uint c = e & 7;
        const uint gm = origin.y + r;
        const uint gn = origin.x + c;
        if (gm < p.m && gn < p.n) {
            uint index = c_batch + decode(gm, p.c_m) + decode(gn, p.c_n);
            store_result(C, index, p.alpha, spill[r * 8 + c], total_scale, p.beta);
        }
    }
}

// Fixed-order sum of Split-K partials. α and β are applied here and nowhere else on this
// path. With `Rescale`, the partials all carry the scale that `plan_scale` gives.
template <typename T, bool Rescale>
inline void contraction_reduce_body(
    device const float* partial,
    device T* C,
    device const uint* stats,
    constant ReduceParams& p,
    uint tid) {
    if (tid >= p.count) {
        return;
    }
    uint b, row, col;
    unpack_mn(tid, p.m, p.n, b, row, col);
    const uint plane = (b * p.m + row) * p.n + col;
    const uint plane_stride = p.batch * p.m * p.n;
    float acc = 0.0f;
    for (uint s = 0; s < p.splits; ++s) {
        acc += partial[s * plane_stride + plane];
    }
    int2 scale = int2(0);
    if (Rescale) {
        scale = plan_scale(stats, p.k);
    }
    uint index = p.c_offset + decode(b, p.c_batch) + decode(row, p.c_m) + decode(col, p.c_n);
    store_result(C, index, p.alpha, acc, scale.x + scale.y, p.beta);
}

inline uint scale_index(constant ScaleParams& p, uint tid) {
    uint b, row, col;
    unpack_mn(tid, p.m, p.n, b, row, col);
    return p.c_offset + decode(b, p.c_batch) + decode(row, p.c_m) + decode(col, p.c_n);
}

// --- Kernels -----------------------------------------------------------------------------

/// First pass of every f32 contraction that reads A and B: folds the magnitudes of one
/// operand into the shared record (largest, and smallest nonzero minus one) that
/// `plan_scale` reads. The grid is (chunks of the inner group) × (outer elements); each
/// thread reads up to `SCAN_CHUNK` elements, four at a time where the inner group is one
/// aligned contiguous run. Max and min commute, so the record does not depend on thread
/// order. The operand itself is never written.
kernel void contraction_scan_f32(
    device const float* X [[buffer(0)]],
    device atomic_uint* stats [[buffer(1)]],
    constant ScanParams& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint begin = gid.x * SCAN_CHUNK;
    if (gid.y >= p.outer_count || begin >= p.inner_count) {
        return;
    }
    const uint end = min(begin + SCAN_CHUNK, p.inner_count);
    const uint base =
        p.offset + decode(gid.y / p.mid_count, p.batch) + decode(gid.y % p.mid_count, p.mid);
    uint highest = 0u;
    uint lowest = 0xffffffffu;
    const bool contiguous = p.inner.ext[0] == 1u && p.inner.ext[1] == 1u &&
        p.inner.ext[2] == 1u && p.inner.stride[3] == 1u;
    if (contiguous && ((base + begin) & 3u) == 0u) {
        uint i = begin;
        for (; i + 4u <= end; i += 4u) {
            float4 v = *reinterpret_cast<const device float4*>(X + base + i);
            uint4 mag = as_type<uint4>(v) & 0x7fffffffu;
            highest = max(highest, max4(mag));
            lowest = min(lowest, min4(mag - 1u));
        }
        for (; i < end; ++i) {
            uint mag = mag_bits(X[base + i]);
            highest = max(highest, mag);
            lowest = min(lowest, mag - 1u);
        }
    } else {
        // Walk the inner group as an odometer, fastest axis last.
        uint digit[4];
        uint rest = begin;
        for (int axis = 3; axis >= 0; --axis) {
            digit[axis] = rest % p.inner.ext[axis];
            rest /= p.inner.ext[axis];
        }
        for (uint i = begin; i < end; ++i) {
            uint index = base + digit[0] * p.inner.stride[0] + digit[1] * p.inner.stride[1] +
                digit[2] * p.inner.stride[2] + digit[3] * p.inner.stride[3];
            uint mag = mag_bits(X[index]);
            highest = max(highest, mag);
            lowest = min(lowest, mag - 1u);
            for (int axis = 3; axis >= 0; --axis) {
                digit[axis] += 1u;
                if (digit[axis] < p.inner.ext[axis]) {
                    break;
                }
                digit[axis] = 0u;
            }
        }
    }
    // Most threads cannot improve the record; skip their atomic writes.
    if (highest > atomic_load_explicit(&stats[p.slot], memory_order_relaxed)) {
        atomic_fetch_max_explicit(&stats[p.slot], highest, memory_order_relaxed);
    }
    if (lowest < atomic_load_explicit(&stats[p.slot + 1u], memory_order_relaxed)) {
        atomic_fetch_min_explicit(&stats[p.slot + 1u], lowest, memory_order_relaxed);
    }
}

/// f32 GEMM (plain, strided, or transposed) whose batch, M, N, and K groups each collapsed
/// to one stride. A threadgroup of 16 simdgroups owns a 64×64 tile of C, stages 64×32 of A
/// and 32×64 of B per K block in threadgroup memory, accumulates with
/// `simdgroup_multiply_accumulate`, and applies α and β once at the store. `stats` is the
/// record written by `contraction_scan_f32`; when it calls for it, the operands are staged
/// scaled by powers of two so that no product is flushed.
kernel void contraction_tiled_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant TiledParams& p [[buffer(3)]],
    device const uint* stats [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sA[SA_ELEMS];
    threadgroup float sB[SB_ELEMS];
    threadgroup float spill[WM * WN * 64];
    contraction_tiled_body<float, false, true>(
        A, B, C, nullptr, stats, p, sA, sB, spill, tgid, tid, sgid, lane);
}

/// f16 counterpart of `contraction_tiled_f32`. Half tiles are widened exactly to float as
/// they are staged, the dot accumulates in f32 through the same `simdgroup_matrix` path,
/// and the result is rounded to half once at the store. β = 0 does not load C.
kernel void contraction_tiled_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant TiledParams& p [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sA[SA_ELEMS];
    threadgroup float sB[SB_ELEMS];
    threadgroup float spill[WM * WN * 64];
    contraction_tiled_body<half, false, false>(
        A, B, C, nullptr, nullptr, p, sA, sB, spill, tgid, tid, sgid, lane);
}

/// One K slice of the f32 tiled kernel (Split-K). Writes a raw f32 partial, never C, so α
/// and β are applied once, by `contraction_reduce_f32`. `partial` holds `splits` planes,
/// all at the scale that `stats` calls for.
kernel void contraction_partial_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* partial [[buffer(2)]],
    constant TiledParams& p [[buffer(3)]],
    device const uint* stats [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sA[SA_ELEMS];
    threadgroup float sB[SB_ELEMS];
    threadgroup float spill[WM * WN * 64];
    contraction_tiled_body<float, true, true>(
        A, B, nullptr, partial, stats, p, sA, sB, spill, tgid, tid, sgid, lane);
}

/// One K slice of the f16 tiled kernel (Split-K). Writes an f32 partial, never C, so α
/// and β are applied once, by `contraction_reduce_f16`. `partial` holds `splits` planes.
kernel void contraction_partial_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* partial [[buffer(2)]],
    constant TiledParams& p [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sA[SA_ELEMS];
    threadgroup float sB[SB_ELEMS];
    threadgroup float spill[WM * WN * 64];
    contraction_tiled_body<half, true, false>(
        A, B, nullptr, partial, nullptr, p, sA, sB, spill, tgid, tid, sgid, lane);
}

/// f32 contraction with up to four strides per axis group, for views the host could not
/// merge down to one stride each. One simdgroup computes an 8×8 output tile through
/// threadgroup staging and `simdgroup_multiply_accumulate`, with the same use of `stats`
/// as the tiled kernel.
kernel void contraction_general_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant GeneralParams& p [[buffer(3)]],
    device const uint* stats [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float a_tile[64];
    threadgroup float b_tile[64];
    threadgroup float spill[64];
    contraction_general_body<float, true>(A, B, C, stats, p, a_tile, b_tile, spill, tgid, lane);
}

/// f16 counterpart of `contraction_general_f32`: half operands widened to float in the
/// staged tiles, f32 accumulation, one rounding to half at the store.
kernel void contraction_general_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant GeneralParams& p [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float a_tile[64];
    threadgroup float b_tile[64];
    threadgroup float spill[64];
    contraction_general_body<half, false>(
        A, B, C, nullptr, p, a_tile, b_tile, spill, tgid, lane);
}

/// Second pass of f32 Split-K: one thread per element of C adds its partials in split
/// order and stores `α·sum + β·C` once, removing the scale that `stats` calls for.
kernel void contraction_reduce_f32(
    device const float* partial [[buffer(0)]],
    device float* C [[buffer(1)]],
    constant ReduceParams& p [[buffer(2)]],
    device const uint* stats [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    contraction_reduce_body<float, true>(partial, C, stats, p, tid);
}

/// Second pass of f16 Split-K: one thread per element of C adds its f32 partials in split
/// order and stores `α·sum + β·C`, rounded to half once.
kernel void contraction_reduce_f16(
    device const float* partial [[buffer(0)]],
    device half* C [[buffer(1)]],
    constant ReduceParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
    contraction_reduce_body<half, false>(partial, C, nullptr, p, tid);
}

/// C = β·C for f32, one thread per element. Used when α = 0 or K = 0 and β is neither 0
/// nor 1, so A and B are not referenced. The product is rounded once and is never
/// flushed: a subnormal C or a subnormal β·C goes through the integer path of `combine_f32`.
kernel void contraction_scale_f32(
    device float* C [[buffer(0)]],
    constant ScaleParams& p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= p.count) {
        return;
    }
    uint index = scale_index(p, tid);
    C[index] = as_type<float>(combine_f32(p.beta, C[index], 0, 0.0f, 0.0f, false));
}

/// C = β·C for f16, one thread per element, for the same plans as `contraction_scale_f32`.
/// Casting `half(β * f32(C))` rounds twice and can land one ulp off (`β = -0.999`,
/// C = 0x0DDC → 0x8DDA instead of 0x8DDB). `two_prod` keeps the exact product and
/// `dd_to_half` rounds it once. Non-finite inputs, and a product that overflows f32, take
/// the IEEE cast.
kernel void contraction_scale_f16(
    device half* C [[buffer(0)]],
    constant ScaleParams& p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= p.count) {
        return;
    }
    uint index = scale_index(p, tid);
    float c = float(C[index]);
    if (!isfinite(p.beta) || !isfinite(c)) {
        C[index] = half(p.beta * c);
        return;
    }
    float hi, lo;
    two_prod(p.beta, c, hi, lo);
    if (!isfinite(hi) || !isfinite(lo)) {
        C[index] = half(p.beta * c);
        return;
    }
    C[index] = dd_to_half(hi, lo);
}

/// C = 0 for f32, one thread per element. Used when α = 0 or K = 0 and β = 0. It never
/// loads C, so a NaN already in C cannot survive β = 0.
kernel void contraction_zero_f32(
    device float* C [[buffer(0)]],
    constant ScaleParams& p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= p.count) {
        return;
    }
    C[scale_index(p, tid)] = 0.0f;
}

/// C = 0 for f16, one thread per element. Like `contraction_zero_f32`, it never loads C.
kernel void contraction_zero_f16(
    device half* C [[buffer(0)]],
    constant ScaleParams& p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= p.count) {
        return;
    }
    C[scale_index(p, tid)] = half(0.0f);
}
