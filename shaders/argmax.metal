// argmax: index of the element of largest magnitude in a strided 1-D view.
//
// Total order used by every tier (thread, SIMD group, threadgroup, second pass):
//   1. a NaN magnitude outranks every number; among NaNs the lowest index wins;
//   2. otherwise the larger magnitude wins;
//   3. equal magnitudes (-0 vs +0, equal infinities, +x vs -x) go to the lowest index.
// Magnitudes are compared as integer keys (the IEEE bit pattern of a non-negative float orders
// like its value, and every NaN maps to one key above +inf), so subnormals are never flushed
// by float compares. The order is total on (key, index) pairs and indices are unique, so the
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
// Smallest positive normal f32; complex parts below it count as zero.
constant constexpr float kMinNormal = 1.17549435e-38f;

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

// A complex part for magnitude arithmetic: |p|, with parts below the smallest normal
// (subnormals, which the GPU may flush) counted as zero so results do not depend on flushing.
inline float part(float p) {
    const float a = fabs(p);
    return (a >= kMinNormal || isnan(a)) ? a : 0.0f;
}

// L1 magnitude of a complex element: |re| + |im| in f32 (NaN if either part is NaN,
// +inf on overflow).
struct ComplexL1 {
    using Element = float2;
    static uint key(float2 z) { return key_of(part(z.x) + part(z.y)); }
};

// Overflow-safe complex modulus: hi * sqrt(1 + (lo / hi)^2) with hi = max(|re|, |im|).
// NaN if either part is NaN; +inf if either part is infinite; 0 if both parts are zero.
struct ComplexHypot {
    using Element = float2;
    static uint key(float2 z) {
        const float re = part(z.x);
        const float im = part(z.y);
        if (isnan(re) || isnan(im)) return kNanKey;
        const float hi = max(re, im);
        const float lo = min(re, im);
        if (hi == 0.0f) return 0u;
        if (isinf(hi)) return as_type<uint>(INFINITY);
        const float r = precise::divide(lo, hi);
        return key_of(hi * precise::sqrt(fma(r, r, 1.0f)));
    }
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
