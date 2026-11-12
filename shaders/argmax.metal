// argmax: index of the element of largest magnitude in a strided 1-D view.
//
// Total order used by every tier (thread, SIMD group, threadgroup, second pass):
//   1. a NaN magnitude outranks every number; among NaNs the lowest index wins;
//   2. otherwise the larger magnitude wins;
//   3. equal magnitudes (-0 vs +0, equal infinities, +x vs -x) go to the lowest index.
// Magnitudes are compared as integer keys (the IEEE bit pattern of a non-negative float orders
// like its value, and every NaN maps to one key above +inf). Complex |re| and |im| are reduced
// to that key with integer scaling, so a subnormal part is not flushed before the f32 formula.
// The order is total on (key, index) pairs and indices are unique, so the
// winner does not depend on the order in which candidates are combined: results are bitwise
// deterministic and no atomics are needed.
//
// Kernels: argmax_{f32,c32_l1,c32_hypot}_single   one threadgroup, writes the result;
//          argmax_{f32,c32_l1,c32_hypot}_partial  many threadgroups, one candidate each;
//          argmax_final                           one threadgroup folds the candidates.

#include <metal_stdlib>
using namespace metal;

// Threads per threadgroup; mirrors THREADGROUP_SIZE in src/ops/argmax.rs.
constant constexpr uint kThreadgroupSize = 256;
// Upper bound on SIMD groups per threadgroup (1024 threads / SIMD width 32).
constant constexpr uint kMaxSimdGroups = 32;
// Key shared by every NaN: above the key of +inf (0x7f800000).
constant constexpr uint kNanKey = 0x7fc00000u;
// Launch parameters; mirrors `Params` in src/ops/argmax.rs field for field.
struct Params {
    ulong offset;      // element offset of the first input element
    ulong stride;      // element stride between consecutive input elements
    ulong out_offset;  // element offset of the u32 output slot
    uint n;            // number of input elements (1 ..= u32::MAX)
    uint groups;       // threadgroups of the first pass; candidates read by argmax_final
};

// A candidate packs (key << 32) | (0xFFFFFFFF - index): the numerically larger packed value
// is the better candidate, and a tie on the key falls to the smaller index. Zero is the
// neutral element: it loses to every real candidate because real indices are below 2^32 - 1.
typedef ulong Candidate;

inline Candidate pack(uint key, uint idx) { return (ulong(key) << 32) | ulong(0xFFFFFFFFu - idx); }
inline uint index_of(Candidate c) { return 0xFFFFFFFFu - uint(c); }

// Integer key of a non-negative float magnitude `m`; all NaNs share kNanKey.
inline uint key_of(float m) {
    const uint bits = as_type<uint>(m) & 0x7fffffffu;
    return bits > 0x7f800000u ? kNanKey : bits;
}

// Magnitude of a real element: |x|, read from the bit pattern so subnormals survive.
struct RealMagnitude {
    using Element = float;
    static uint key(float v) { return key_of(v); }
};

// Sign-stripped IEEE bits. For a non-negative finite value these bits are its magnitude key.
inline uint abs_bits(float p) { return as_type<uint>(p) & 0x7fffffffu; }

inline uint clz_ulong(ulong x) {
    const uint hi = uint(x >> 32);
    return hi != 0u ? clz(hi) : 32u + clz(uint(x));
}

// Correctly rounded key of `prod * 2^binary_exp`. `prod` is nonzero.
// Normal results take one round-to-nearest-even of the 24-bit significand. Subnormal
// results round `prod * 2^(binary_exp + 149)` onto the 2^-149 grid. No float arithmetic,
// so flush-to-zero cannot change the bits.
inline uint round_scaled_int(ulong prod, int binary_exp) {
    const uint p = 63u - clz_ulong(prod);
    int biased = int(p) + binary_exp + 127;
    if (biased >= 255) return 0x7f800000u;
    if (p >= 23u && biased >= 1) {
        const uint shift = p - 23u;
        ulong sig = prod >> shift;
        if (shift > 0u) {
            const ulong mid = ulong(1) << (shift - 1u);
            const ulong rem = prod & ((mid << 1u) - 1ul);
            if (rem > mid || (rem == mid && (sig & 1ul) != 0ul)) {
                sig += 1ul;
                if (sig == 0x1000000ul) {
                    sig >>= 1u;
                    biased += 1;
                    if (biased >= 255) return 0x7f800000u;
                }
            }
        }
        if (biased >= 1) return (uint(biased) << 23) | uint(sig & 0x7ffffful);
    }
    // Subnormal or zero: integer number of 2^-149 units.
    const int q = binary_exp + 149;
    if (q >= 63) return 0x7f800000u;
    if (q >= 0) {
        const ulong lifted = prod << uint(q);
        return lifted >= 0x800000ul ? 0x00800000u : uint(lifted);
    }
    const uint right = uint(-q);
    if (right >= 64u) return 0u;
    ulong main = prod >> right;
    const ulong mid = ulong(1) << (right - 1u);
    const ulong rem = prod & ((mid << 1u) - 1ul);
    if (rem > mid || (rem == mid && (main & 1ul) != 0ul)) main += 1ul;
    if (main >= 0x800000ul) return 0x00800000u | uint(main - 0x800000ul);
    return uint(main);
}

// `s * m * 2^pow2`, with `s` a finite positive normal and `m` a 24-bit integer.
inline uint scale_f32_u24(float s, uint m, int pow2) {
    const uint bits = as_type<uint>(s);
    const uint exp = bits >> 23;
    const ulong sig = ulong(0x800000u | (bits & 0x7fffffu));
    return round_scaled_int(sig * ulong(m), int(exp) - 150 + pow2);
}

// Subtract `delta` from the biased exponent of a positive normal. The callers that use
// this only do so when the result stays normal, so the significand does not round again.
inline uint exp_delta(float v, int delta) {
    const uint bits = as_type<uint>(v);
    const int exp = int(bits >> 23) + delta;
    if (exp >= 255) return 0x7f800000u;
    if (exp >= 1) return (uint(exp) << 23) | (bits & 0x7fffffu);
    const uint sig = 0x800000u | (bits & 0x7fffffu);
    return round_scaled_int(ulong(sig), int(bits >> 23) - 150 + delta);
}

// `m * 2^-126` for a subnormal field `m` in 1 .. 2^23-1. That is the subnormal times 2^23,
// and it is a normal, built without a floating multiply.
inline float subnormal_times_2_to_23(uint m) {
    const uint p = 31u - clz(m);
    const uint sig = m << (23u - p);
    return as_type<float>(((p + 1u) << 23) | (sig & 0x7fffffu));
}

inline float scale_normal(uint bits, int delta) {
    const int exp = int(bits >> 23) + delta;
    return as_type<float>((uint(exp) << 23) | (bits & 0x7fffffu));
}

// |re| + |im| in f32. Subnormal parts are summed in integer units of 2^-149, which is
// exact: two subnormal fields add to the IEEE bits of their sum (a subnormal, or the
// normal just above the smallest normal). A subnormal added to a normal uses those same
// units when it can reach half an ulp, and is ignored when it cannot.
inline uint l1_key(uint a, uint b) {
    if (a > 0x7f800000u || b > 0x7f800000u) return kNanKey;
    if (a == 0x7f800000u || b == 0x7f800000u) return 0x7f800000u;
    if (a < b) {
        const uint swap = a;
        a = b;
        b = swap;
    }
    if (a < 0x00800000u) return a + b;
    if (b == 0u || b >= 0x00800000u) return key_of(as_type<float>(a) + as_type<float>(b));
    const uint exp = a >> 23;
    // At biased exp 25 and above, a subnormal is strictly under half an ulp of `a`.
    if (exp >= 25u) return a;
    const ulong sig = ulong(0x800000u | (a & 0x7fffffu));
    return round_scaled_int((sig << (exp - 1u)) + ulong(b), -149);
}

// max·sqrt(1+(min/max)^2) in f32. A subnormal input is scaled into the normal range by
// an exact power of two, the f32 formula runs there, and the power is removed by an
// integer exponent adjustment so the rounding matches IEEE and not flush-to-zero.
inline uint hypot_key(uint a, uint b) {
    if (a > 0x7f800000u || b > 0x7f800000u) return kNanKey;
    if (a == 0x7f800000u || b == 0x7f800000u) return 0x7f800000u;
    if (a < b) {
        const uint swap = a;
        a = b;
        b = swap;
    }
    if (b == 0u) return a;
    if (a >= 0x00800000u && b >= 0x00800000u) {
        const float hi = as_type<float>(a);
        const float lo = as_type<float>(b);
        const float r = precise::divide(lo, hi);
        return key_of(hi * precise::sqrt(fma(r, r, 1.0f)));
    }
    if (a < 0x00800000u) {
        const float r = precise::divide(float(b), float(a));
        const float root = precise::sqrt(fma(r, r, 1.0f));
        return scale_f32_u24(root, a, -149);
    }
    // `a` is normal and `b` is subnormal. Once `a` reaches 2^-114, `b/a < 2^-12`,
    // so `(b/a)^2` is strictly below half an ulp of 1 and the hypot is `a`.
    if ((a >> 23) >= 13u) return a;
    const float hi = scale_normal(a, 23);
    const float lo = subnormal_times_2_to_23(b);
    const float r = precise::divide(lo, hi);
    const float scaled = hi * precise::sqrt(fma(r, r, 1.0f));
    return exp_delta(scaled, -23);
}

// L1 magnitude of a complex element: |re| + |im| in f32 (NaN if either part is NaN,
// +inf on overflow).
struct ComplexL1 {
    using Element = float2;
    static uint key(float2 z) { return l1_key(abs_bits(z.x), abs_bits(z.y)); }
};

// Overflow-safe complex modulus: hi * sqrt(1 + (lo / hi)^2) with hi = max(|re|, |im|).
// NaN if either part is NaN; +inf if either part is infinite; 0 if both parts are zero.
struct ComplexHypot {
    using Element = float2;
    static uint key(float2 z) { return hypot_key(abs_bits(z.x), abs_bits(z.y)); }
};

// Grid-stride serial accumulation over elements start, start + step, ... of the view.
// Indices ascend, so a strict `>` keeps the lowest index among equal keys.
template <typename Magnitude>
inline Candidate scan_view(device const typename Magnitude::Element* x, constant Params& p,
                           uint start, uint step) {
    if (start >= p.n) return 0;
    uint best_idx = start;
    uint best_key = Magnitude::key(x[p.offset + ulong(start) * p.stride]);
    for (ulong i = ulong(start) + step; i < p.n; i += step) {
        const uint key = Magnitude::key(x[p.offset + i * p.stride]);
        if (key > best_key) {
            best_key = key;
            best_idx = uint(i);
        }
    }
    return pack(best_key, best_idx);
}

// Grid-stride serial accumulation over candidates start, start + step, ... of `partials`.
inline Candidate scan_candidates(device const Candidate* partials, uint count, uint start,
                                 uint step) {
    Candidate best = 0;
    for (uint i = start; i < count; i += step) best = max(best, partials[i]);
    return best;
}

// Register reduction across one SIMD group with simd_shuffle_down; lane 0 ends with the best.
inline Candidate reduce_simd_group(Candidate c, uint lane, uint width) {
    for (uint delta = width / 2; delta > 0; delta >>= 1) {
        const uint hi = simd_shuffle_down(uint(c >> 32), delta);
        const uint lo = simd_shuffle_down(uint(c), delta);
        const Candidate other = (ulong(hi) << 32) | ulong(lo);
        if (lane + delta < width) c = max(c, other);
    }
    return c;
}

// Threadgroup reduction: SIMD-group reductions in registers, then the first SIMD group folds
// the per-group winners staged in threadgroup memory. Thread 0 ends with the best.
inline Candidate reduce_threadgroup(Candidate c, threadgroup Candidate* staged, uint lane,
                                    uint simd_group, uint width, uint simd_groups) {
    c = reduce_simd_group(c, lane, width);
    if (lane == 0) staged[simd_group] = c;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        c = lane < simd_groups ? staged[lane] : Candidate(0);
        c = reduce_simd_group(c, lane, width);
    }
    return c;
}

// Single-threadgroup fast path: scans the whole view and writes the winning index.
template <typename Magnitude>
kernel void argmax_single(device const typename Magnitude::Element* x [[buffer(0)]],
                          device uint* out [[buffer(1)]],
                          constant Params& p [[buffer(2)]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint simd_group [[simdgroup_index_in_threadgroup]],
                          uint width [[threads_per_simdgroup]],
                          uint simd_groups [[simdgroups_per_threadgroup]]) {
    threadgroup Candidate staged[kMaxSimdGroups];
    Candidate best = scan_view<Magnitude>(x, p, tid, kThreadgroupSize);
    best = reduce_threadgroup(best, staged, lane, simd_group, width, simd_groups);
    if (tid == 0) out[p.out_offset] = index_of(best);
}

// First pass of the two-pass path: threadgroup g scans a grid-stride slice of the view and
// writes its winner to partials[g].
template <typename Magnitude>
kernel void argmax_partial(device const typename Magnitude::Element* x [[buffer(0)]],
                           device Candidate* partials [[buffer(1)]],
                           constant Params& p [[buffer(2)]],
                           uint group [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint simd_group [[simdgroup_index_in_threadgroup]],
                           uint width [[threads_per_simdgroup]],
                           uint simd_groups [[simdgroups_per_threadgroup]]) {
    threadgroup Candidate staged[kMaxSimdGroups];
    Candidate best = scan_view<Magnitude>(x, p, group * kThreadgroupSize + tid,
                                          p.groups * kThreadgroupSize);
    best = reduce_threadgroup(best, staged, lane, simd_group, width, simd_groups);
    if (tid == 0) partials[group] = best;
}

#define ARGMAX_SINGLE(NAME, MAGNITUDE)                                                     \
    template [[host_name(NAME)]] kernel void argmax_single<MAGNITUDE>(                     \
        device const MAGNITUDE::Element*, device uint*, constant Params&, uint, uint, uint, \
        uint, uint);

#define ARGMAX_PARTIAL(NAME, MAGNITUDE)                                                    \
    template [[host_name(NAME)]] kernel void argmax_partial<MAGNITUDE>(                    \
        device const MAGNITUDE::Element*, device Candidate*, constant Params&, uint, uint, \
        uint, uint, uint, uint);

ARGMAX_SINGLE("argmax_f32_single", RealMagnitude)
ARGMAX_SINGLE("argmax_c32_l1_single", ComplexL1)
ARGMAX_SINGLE("argmax_c32_hypot_single", ComplexHypot)
ARGMAX_PARTIAL("argmax_f32_partial", RealMagnitude)
ARGMAX_PARTIAL("argmax_c32_l1_partial", ComplexL1)
ARGMAX_PARTIAL("argmax_c32_hypot_partial", ComplexHypot)

// Second pass: one threadgroup folds the `groups` candidates and writes the winning index.
kernel void argmax_final(device const Candidate* partials [[buffer(0)]],
                         device uint* out [[buffer(1)]],
                         constant Params& p [[buffer(2)]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint simd_group [[simdgroup_index_in_threadgroup]],
                         uint width [[threads_per_simdgroup]],
                         uint simd_groups [[simdgroups_per_threadgroup]]) {
    threadgroup Candidate staged[kMaxSimdGroups];
    Candidate best = scan_candidates(partials, p.groups, tid, kThreadgroupSize);
    best = reduce_threadgroup(best, staged, lane, simd_group, width, simd_groups);
    if (tid == 0) out[p.out_offset] = index_of(best);
}
