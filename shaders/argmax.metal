// argmax: index of the element of largest magnitude in a strided 1-D view.
//
// Total order used by every tier (thread, SIMD group, threadgroup, second pass):
//   1. a NaN magnitude outranks every number; among NaNs the lowest index wins;
//   2. otherwise the larger magnitude wins;
//   3. equal magnitudes (-0 vs +0, equal infinities, +x vs -x) go to the lowest index.
// Magnitudes are compared as integer keys (the IEEE bit pattern of a non-negative float orders
// like its value, and every NaN maps to one key above +inf). Complex |re| and |im| are reduced
// to that key in integer arithmetic, so the GPU's flush-to-zero handling of subnormals cannot
// change a bit: the L1 key is the correctly rounded f32 sum and the Hypot key is the correctly
// rounded f32 modulus (error at most 0.5 ULP, see hypot_key).
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
// Launch parameters; mirrors `ArgmaxParams` in src/ops/argmax.rs field for field.
struct ArgmaxParams {
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

// Largest exponent gap at which the smaller part can still move the rounded modulus.
// With a gap of 13 or more, lo/hi < 2^-12, so hypot - hi < hi * 2^-25, which is strictly
// below half an ulp of `hi`: the correctly rounded modulus is `hi` itself.
constant constexpr uint kHypotMaxExpGap = 12;
// Correction steps allowed after a float estimate. Each estimate is within 3 ulp of the
// exact value (three rounded operations, a relative 2^-23), so at most 3 steps are taken.
constant constexpr uint kHypotMaxSteps = 6;

// Correctly rounded sqrt(a^2 + b^2) for two subnormal fields `a`, `b` (not both zero), in
// units of 2^-149. The modulus is below 2^23.5 units, where f32 values are the integers, so
// the key is sqrt(a^2 + b^2) rounded to the nearest integer; an integer's square root is
// never half way between two integers, so there are no ties. A result of 2^23 or more is the
// bit pattern of that normal value. The float sqrt only seeds the search (its operand is at
// least 1, so nothing is flushed); the integer tests decide.
inline uint hypot_key_subnormal(uint a, uint b) {
    const ulong sum = ulong(a) * ulong(a) + ulong(b) * ulong(b);
    ulong q = ulong(precise::sqrt(float(sum)));
    for (uint step = 0; step < kHypotMaxSteps; ++step) {
        // q is the nearest integer exactly when q^2 - q < sum <= q^2 + q.
        if (sum > q * q + q) {
            q += 1ul;
        } else if (sum <= q * q - q) {
            q -= 1ul;
        } else {
            break;
        }
    }
    return uint(q);
}

// Exact rounding of sqrt(A^2 + (B * 2^-gap)^2) starting from the candidate `est` (float bits
// in [1, 4), in units where A * 2^-23 is the larger part). Returns the correctly rounded
// candidate bits (round to nearest, ties to even).
//
// In units where X = A * 2^(gap+1) and Y = 2B are integers, a candidate with significand q
// and ulp u = 2^s stands for Q = q * u. The residual R = X^2 + Y^2 - Q^2 is far below 2^63
// in magnitude, so 64-bit wrapping arithmetic yields it exactly. The exact modulus exceeds
// the midpoint above Q exactly when 4R - 4Qu - u^2 > 0, and is below the midpoint under Q
// when 4R + 4Qv - v^2 < 0, where v is the ulp under Q (u/2 at a power of two). A zero means
// an exact tie, which goes to the even significand. The candidate steps one ulp at a time
// until neither test fires.
inline uint hypot_round_exact(uint est, uint sig_a, uint sig_b, uint gap) {
    const ulong x = ulong(sig_a) << (gap + 1u);
    const ulong y = ulong(sig_b) << 1u;
    const ulong sum = x * x + y * y;
    for (uint step = 0; step < kHypotMaxSteps; ++step) {
        const uint q = 0x800000u | (est & 0x7fffffu);
        const uint s = (est >> 23) - 126u + gap;
        const ulong big = ulong(q) << s;
        const long r4 = long((sum - big * big) << 2u);
        const bool odd = (q & 1u) != 0u;
        const long above = r4 - long(big << (s + 2u)) - long(1ul << (2u * s));
        const uint s_below = q == 0x800000u ? s - 1u : s;
        const long below = r4 + long(big << (s_below + 2u)) - long(1ul << (2u * s_below));
        if (above > 0 || (above == 0 && odd)) {
            est += 1u;
        } else if (below < 0 || (below == 0 && odd)) {
            est -= 1u;
        } else {
            break;
        }
    }
    return est;
}

// Key of the complex modulus sqrt(re^2 + im^2), correctly rounded to f32 (round to nearest,
// ties to even): the error is at most 0.5 ULP for every finite input, the result is +inf
// only when the rounded modulus exceeds FLT_MAX, and no intermediate overflows or underflows.
// `a` and `b` are the sign-stripped bits of the two parts.
//
// Method, for hi >= lo > 0 with significands A, B (24-bit integers) and exponent gap d:
//   1. Estimate. fa = A * 2^-23 in [1, 2) and fb = B * 2^(-23-d) in (0, 2) are exact normal
//      floats, and est = sqrt(fa^2 + fb^2) lies in [1, 2.83], within 3 ulp of the exact
//      value whether or not the multiply-add is contracted. All operands and results are
//      normal, so the GPU's flush-to-zero mode is never reached. The estimate only seeds
//      the integer tests below; the key does not depend on how it was rounded.
//   2. 32-bit certificate. With q the significand of est, e its exponent (0 or 1) and
//      u = 2^(e-23) its ulp, the residual fa^2 + fb^2 - est^2 in units of 2^-46 is
//      r + f with r = A^2 + floor(B^2 / 4^d) - q^2 * 4^e and 0 <= f < 1. It is below 2^29 in
//      magnitude, so wrapping 32-bit arithmetic yields r exactly. In the same units
//      w = q * 4^e is est * u, and the exact value lies above the midpoint over est exactly
//      when r + f > w + 4^e/4, and below the midpoint under est when r + f < -w + 4^e/4
//      (for one common ulp). Hence est is correctly rounded if 2 - w <= r < w, est + u if
//      w + 2 <= r < 3w, and est - u if 10 - 3w <= r < -w; the bounds are tightened where
//      the ulp halves under a power of two. These ranges exclude every exact tie.
//   3. Anything else (a value within about 2^-23 ulp of a midpoint, or an estimate more than
//      one step off) goes to hypot_round_exact.
//   4. The exponent of `hi` is added back as an integer; the significand is not rounded
//      again, and a carry past the largest finite exponent gives +inf.
inline uint hypot_key(uint a, uint b) {
    const uint hi = max(a, b);
    const uint lo = min(a, b);
    a = hi;
    b = lo;
    if (a >= 0x7f800000u) return a > 0x7f800000u ? kNanKey : a;
    const uint exp_a = a >> 23;
    const uint gap = exp_a - max(b >> 23, 1u);
    if (b == 0u || gap > kHypotMaxExpGap) {
        return (a < 0x00800000u && b != 0u) ? hypot_key_subnormal(a, b) : a;
    }
    const uint sig_a = 0x800000u | (a & 0x7fffffu);
    const uint sig_b = b >= 0x00800000u ? (0x800000u | (b & 0x7fffffu)) : b;

    const float fa = as_type<float>(0x3f800000u | (a & 0x7fffffu));
    const float fb = float(sig_b) * as_type<float>((104u - gap) << 23);
    // The clamp keeps the integer steps well defined whatever the float unit returns.
    uint est = clamp(as_type<uint>(precise::sqrt(fa * fa + fb * fb)), 0x3f800000u, 0x407fffffu);

    const uint q = 0x800000u | (est & 0x7fffffu);
    const uint e = (est >> 23) - 127u;
    const uint big = q << e;
    // floor(B^2 / 4^gap) to within 2 units, from B = hb * 2^gap + rem without a wide product.
    const uint hb = sig_b >> gap;
    const uint rem = sig_b - (hb << gap);
    const int r = int((sig_a - big) * (sig_a + big) + hb * hb + ((2u * hb * rem) >> gap));
    const int w = int(big << e);
    const bool pow2 = q == 0x800000u;
    const bool keep = r >= (pow2 ? 0 : 2 - w) && r < w - 1;
    const bool up = r >= w + 2 && r < 3 * w - 1;
    const bool down = !pow2 && r < -w - 1 && r >= (q == 0x800001u ? -2 * w : 10 - 3 * w);
    if (keep || up || down) {
        est = est + uint(up) - uint(down);
    } else {
        est = hypot_round_exact(est, sig_a, sig_b, gap);
    }
    // est is in [1, 4], so its bits are at least those of 1.0 and the sum cannot wrap.
    return min((est - 0x3f800000u) + (exp_a << 23), 0x7f800000u);
}

// L1 magnitude of a complex element: |re| + |im| in f32 (NaN if either part is NaN,
// +inf on overflow).
struct ComplexL1 {
    using Element = float2;
    static uint key(float2 z) { return l1_key(abs_bits(z.x), abs_bits(z.y)); }
};

// Overflow-safe complex modulus sqrt(re^2 + im^2), correctly rounded to f32 (see hypot_key).
// NaN if either part is NaN; +inf if either part is infinite; 0 if both parts are zero.
struct ComplexHypot {
    using Element = float2;
    static uint key(float2 z) { return hypot_key(abs_bits(z.x), abs_bits(z.y)); }
};

// Grid-stride serial accumulation over elements start, start + step, ... of the view.
// Indices ascend, so a strict `>` keeps the lowest index among equal keys.
template <typename Magnitude>
inline Candidate scan_view(device const typename Magnitude::Element* x,
                           constant ArgmaxParams& p, uint start, uint step) {
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
                          constant ArgmaxParams& p [[buffer(2)]],
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
                           constant ArgmaxParams& p [[buffer(2)]],
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
        device const MAGNITUDE::Element*, device uint*, constant ArgmaxParams&, uint, uint, \
        uint, uint, uint);

#define ARGMAX_PARTIAL(NAME, MAGNITUDE)                                                    \
    template [[host_name(NAME)]] kernel void argmax_partial<MAGNITUDE>(                    \
        device const MAGNITUDE::Element*, device Candidate*, constant ArgmaxParams&, uint, \
        uint, uint, uint, uint, uint);

ARGMAX_SINGLE("argmax_f32_single", RealMagnitude)
ARGMAX_SINGLE("argmax_c32_l1_single", ComplexL1)
ARGMAX_SINGLE("argmax_c32_hypot_single", ComplexHypot)
ARGMAX_PARTIAL("argmax_f32_partial", RealMagnitude)
ARGMAX_PARTIAL("argmax_c32_l1_partial", ComplexL1)
ARGMAX_PARTIAL("argmax_c32_hypot_partial", ComplexHypot)

// Second pass: one threadgroup folds the `groups` candidates and writes the winning index.
kernel void argmax_final(device const Candidate* partials [[buffer(0)]],
                         device uint* out [[buffer(1)]],
                         constant ArgmaxParams& p [[buffer(2)]],
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
