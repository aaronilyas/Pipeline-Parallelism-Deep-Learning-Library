// Tensor contraction: C = α·A·B + β·C.
//
// Axes arrive already grouped into batch / M / N / K. A group is either one stride
// (the tiled kernels) or up to four strides, fastest axis last in `Axes` (the general
// kernel). f32 accumulates in `simdgroup_matrix<float, 8, 8>`. f16 dots use a
// double-float expansion, because an f32 dot can land on the wrong side of an
// f16 midpoint and miss the accuracy gate by one ulp.
//
// Tiled kernel: each threadgroup owns a 64×64 output tile and streams K in chunks of
// 32 through padded threadgroup memory, then `simdgroup_multiply_accumulate`. Tiles that
// hang off M, N, or K are zero-filled on the way in and predicate the store. β = 0 does
// not load C.
//
// Split-K (small M·N, long K): several threadgroups split K at fixed chunk boundaries
// and write raw f32 partials. A later kernel adds those partials in split order and
// applies α and β once. No atomics, so the sum is bit-stable.

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

struct ReduceParams {
    uint c_offset;
    uint count;
    uint m;
    uint n;
    uint batch;
    uint splits;
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

// Device operands stay in T. Threadgroup tiles are float so the 8-wide dot
// accumulates in f32 for both f32 and f16 inputs (every f16 value is exact in f32).
template <typename T, bool WritePartial>
inline void contract_tiled_body(
    device const T* A,
    device const T* B,
    device T* C,
    device float* partial,
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
    uint k_begin;
    uint k_end;
    if (WritePartial) {
        split = tgid.z / p.batch;
        batch_id = tgid.z - split * p.batch;
        k_begin = split * p.k_chunk;
        k_end = min(k_begin + p.k_chunk, p.k);
    } else {
        split = 0;
        batch_id = tgid.z;
        k_begin = 0;
        k_end = p.k;
    }
    // `tgid` is uniform across the threadgroup, so this return is uniform.
    if (k_begin >= k_end) {
        return;
    }

    const uint m0 = tgid.y * BM;
    const uint n0 = tgid.x * BN;
    const uint sg_row = sgid / WN;
    const uint sg_col = sgid % WN;
    const uint a_base = p.a_offset + batch_id * p.a_batch_stride;
    const uint b_base = p.b_offset + batch_id * p.b_batch_stride;
    const uint c_base = p.c_offset + batch_id * p.c_batch_stride;

    simdgroup_matrix<float, 8, 8> acc[TM][TN];
    for (uint i = 0; i < TM; ++i) {
        for (uint j = 0; j < TN; ++j) {
            acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    for (uint k_block = k_begin; k_block < k_end; k_block += BK) {
        const bool full = p.vector_ok != 0 && (m0 + BM <= p.m) && (n0 + BN <= p.n) &&
            (k_block + BK <= k_end);
        if (full) {
            // Contiguous, aligned inner edges: one vec4 per four elements.
            constexpr uint VEC_A = (BM * BK) / (THREADS * 4);
            for (uint i = 0; i < VEC_A; ++i) {
                const uint idx = i * THREADS + tid;
                const uint elem = idx * 4;
                const uint row = elem / BK;
                const uint col = elem % BK;
                const device T* ptr =
                    A + (a_base + (m0 + row) * p.a_m_stride + k_block + col);
                vec<T, 4> v = *reinterpret_cast<const device vec<T, 4>*>(ptr);
                *reinterpret_cast<threadgroup float4*>(sA + row * SA_STRIDE + col) =
                    float4(float(v[0]), float(v[1]), float(v[2]), float(v[3]));
            }
            constexpr uint VEC_B = (BK * BN) / (THREADS * 4);
            for (uint i = 0; i < VEC_B; ++i) {
                const uint idx = i * THREADS + tid;
                const uint elem = idx * 4;
                const uint row = elem / BN;
                const uint col = elem % BN;
                const device T* ptr = B +
                    (b_base + (k_block + row) * p.b_k_stride + (n0 + col) * p.b_n_stride);
                vec<T, 4> v = *reinterpret_cast<const device vec<T, 4>*>(ptr);
                *reinterpret_cast<threadgroup float4*>(sB + row * SB_STRIDE + col) =
                    float4(float(v[0]), float(v[1]), float(v[2]), float(v[3]));
            }
        } else {
            constexpr uint EPT_A = (BM * BK) / THREADS;
            for (uint i = 0; i < EPT_A; ++i) {
                const uint idx = i * THREADS + tid;
                const uint row = idx / BK;
                const uint col = idx % BK;
                const uint gm = m0 + row;
                const uint gk = k_block + col;
                float v = 0.0f;
                if (gm < p.m && gk < k_end) {
                    v = float(A[a_base + gm * p.a_m_stride + gk * p.a_k_stride]);
                }
                sA[row * SA_STRIDE + col] = v;
            }
            constexpr uint EPT_B = (BK * BN) / THREADS;
            for (uint i = 0; i < EPT_B; ++i) {
                const uint idx = i * THREADS + tid;
                const uint row = idx / BN;
                const uint col = idx % BN;
                const uint gk = k_block + row;
                const uint gn = n0 + col;
                float v = 0.0f;
                if (gk < k_end && gn < p.n) {
                    v = float(B[b_base + gn * p.b_n_stride + gk * p.b_k_stride]);
                }
                sB[row * SB_STRIDE + col] = v;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Fragment predicates are uniform inside a simdgroup (they depend on `sgid`).
        // A full simdgroup tile loads each A and B fragment once per K step.
        const bool sg_full = (m0 + (sg_row + 1) * (TM * 8)) <= p.m &&
            (n0 + (sg_col + 1) * (TN * 8)) <= p.n;
        if (sg_full) {
            for (uint kk = 0; kk < BK; kk += 8) {
                simdgroup_matrix<float, 8, 8> a_frag[TM];
                simdgroup_matrix<float, 8, 8> b_frag[TN];
                for (uint i = 0; i < TM; ++i) {
                    const uint row = sg_row * (TM * 8) + i * 8;
                    simdgroup_load(a_frag[i], sA + row * SA_STRIDE + kk, SA_STRIDE);
                }
                for (uint j = 0; j < TN; ++j) {
                    const uint col = sg_col * (TN * 8) + j * 8;
                    simdgroup_load(b_frag[j], sB + kk * SB_STRIDE + col, SB_STRIDE);
                }
                for (uint i = 0; i < TM; ++i) {
                    for (uint j = 0; j < TN; ++j) {
                        simdgroup_multiply_accumulate(
                            acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
                    }
                }
            }
        } else if ((m0 + sg_row * (TM * 8)) < p.m && (n0 + sg_col * (TN * 8)) < p.n) {
            for (uint kk = 0; kk < BK; kk += 8) {
                for (uint i = 0; i < TM; ++i) {
                    for (uint j = 0; j < TN; ++j) {
                        const uint row = sg_row * (TM * 8) + i * 8;
                        const uint col = sg_col * (TN * 8) + j * 8;
                        if (m0 + row < p.m && n0 + col < p.n) {
                            simdgroup_matrix<float, 8, 8> a_frag;
                            simdgroup_matrix<float, 8, 8> b_frag;
                            simdgroup_load(a_frag, sA + row * SA_STRIDE + kk, SA_STRIDE);
                            simdgroup_load(b_frag, sB + kk * SB_STRIDE + col, SB_STRIDE);
                            simdgroup_multiply_accumulate(acc[i][j], a_frag, b_frag, acc[i][j]);
                        }
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint plane_stride = p.batch * p.m * p.n;
    for (uint i = 0; i < TM; ++i) {
        for (uint j = 0; j < TN; ++j) {
            const uint row = sg_row * (TM * 8) + i * 8;
            const uint col = sg_col * (TN * 8) + j * 8;
            if (m0 + row >= p.m || n0 + col >= p.n) {
                continue;
            }
            // `simdgroup_barrier` waits only for this simdgroup, so the predicate above
            // (uniform per simdgroup, divergent across them) is safe.
            threadgroup float* slot = spill + sgid * 64;
            simdgroup_store(acc[i][j], slot, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64; e += 32) {
                const uint r = e >> 3;
                const uint c = e & 7;
                const uint gm = m0 + row + r;
                const uint gn = n0 + col + c;
                if (gm < p.m && gn < p.n) {
                    float accv = slot[r * 8 + c];
                    if (WritePartial) {
                        uint plane = (batch_id * p.m + gm) * p.n + gn;
                        partial[split * plane_stride + plane] = accv;
                    } else {
                        uint index = c_base + gm * p.c_m_stride + gn * p.c_n_stride;
                        float out = p.alpha * accv;
                        if (p.beta != 0.0f) {
                            out += p.beta * float(C[index]);
                        }
                        C[index] = T(out);
                    }
                }
            }
        }
    }
}

template <typename T>
inline void contract_general_body(
    device const T* A,
    device const T* B,
    device T* C,
    constant GeneralParams& p,
    threadgroup float* a_tile,
    threadgroup float* b_tile,
    threadgroup float* spill,
    uint3 tgid,
    uint lane) {
    const uint n0 = tgid.x * 8;
    const uint m0 = tgid.y * 8;
    const uint batch_id = tgid.z;
    if (m0 >= p.m || n0 >= p.n || batch_id >= p.batch) {
        return;
    }

    const uint a_batch = p.a_offset + decode(batch_id, p.a_batch);
    const uint b_batch = p.b_offset + decode(batch_id, p.b_batch);
    const uint c_batch = p.c_offset + decode(batch_id, p.c_batch);

    simdgroup_matrix<float, 8, 8> acc(0.0f);
    for (uint k0 = 0; k0 < p.k; k0 += 8) {
        for (uint e = lane; e < 64; e += 32) {
            const uint r = e >> 3;
            const uint c = e & 7;
            // A tile is M × K, B tile is K × N, both row-major in the 8×8 scratch.
            const uint gm = m0 + r;
            const uint gk_a = k0 + c;
            float va = 0.0f;
            if (gm < p.m && gk_a < p.k) {
                va = float(A[a_batch + decode(gm, p.a_m) + decode(gk_a, p.a_k)]);
            }
            a_tile[e] = va;
            const uint gk_b = k0 + r;
            const uint gn = n0 + c;
            float vb = 0.0f;
            if (gk_b < p.k && gn < p.n) {
                vb = float(B[b_batch + decode(gn, p.b_n) + decode(gk_b, p.b_k)]);
            }
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

    simdgroup_store(acc, spill, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < 64; e += 32) {
        const uint r = e >> 3;
        const uint c = e & 7;
        const uint gm = m0 + r;
        const uint gn = n0 + c;
        if (gm < p.m && gn < p.n) {
            float out = p.alpha * spill[r * 8 + c];
            uint index = c_batch + decode(gm, p.c_m) + decode(gn, p.c_n);
            if (p.beta != 0.0f) {
                out += p.beta * float(C[index]);
            }
            C[index] = T(out);
        }
    }
}

// NN / strided / transposed GEMM whose batch, M, N, and K groups each collapsed to one stride.
kernel void contract_tiled_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant TiledParams& p [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sA[SA_ELEMS];
    threadgroup float sB[SB_ELEMS];
    threadgroup float spill[WM * WN * 64];
    contract_tiled_body<float, false>(A, B, C, nullptr, p, sA, sB, spill, tgid, tid, sgid, lane);
}

// One K-slice of the tiled kernel. Writes an unscaled f32 partial, never C, so β is applied once.
kernel void contract_partial_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* partial [[buffer(2)]],
    constant TiledParams& p [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sA[SA_ELEMS];
    threadgroup float sB[SB_ELEMS];
    threadgroup float spill[WM * WN * 64];
    contract_tiled_body<float, true>(
        A, B, nullptr, partial, p, sA, sB, spill, tgid, tid, sgid, lane);
}

// Up to four strides per group. One simdgroup computes an 8×8 output tile.
kernel void contract_general_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant GeneralParams& p [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float a_tile[64];
    threadgroup float b_tile[64];
    threadgroup float spill[64];
    contract_general_body<float>(A, B, C, p, a_tile, b_tile, spill, tgid, lane);
}

// Error-free transform: s + e = a + b exactly (no overflow).
inline void two_sum(float a, float b, thread float& s, thread float& e) {
    s = a + b;
    float v = s - a;
    e = (a - (s - v)) + (b - v);
}

// p + e = a * b exactly. Fast math is off, so `fma` is a real fused multiply-add.
inline void two_prod(float a, float b, thread float& p, thread float& e) {
    p = a * b;
    e = fma(a, b, -p);
}

// (hi, lo) += x. Non-finite values take the IEEE path: two_sum(inf, x) is NaN.
inline void dd_add(thread float& hi, thread float& lo, float x) {
    if (!isfinite(hi) || !isfinite(lo) || !isfinite(x)) {
        hi = hi + x + lo;
        lo = 0.0f;
        return;
    }
    float s, e;
    two_sum(hi, x, s, e);
    e += lo;
    two_sum(s, e, hi, lo);
}

// (hi, lo) *= s, exact for the dyadic scalars the tests use and faithful otherwise.
inline void dd_mul(thread float& hi, thread float& lo, float s) {
    if (!isfinite(hi) || !isfinite(lo) || !isfinite(s)) {
        hi = fma(s, hi, s * lo);
        lo = 0.0f;
        return;
    }
    float p1, e1, p2, e2;
    two_prod(s, hi, p1, e1);
    two_prod(s, lo, p2, e2);
    hi = p1;
    lo = e1;
    dd_add(hi, lo, p2);
    dd_add(hi, lo, e2);
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

inline void store_dd(device half* C, uint index, float hi, float lo, float alpha, float beta) {
    dd_mul(hi, lo, alpha);
    if (beta != 0.0f) {
        float c = float(C[index]);
        if (!isfinite(hi) || !isfinite(lo) || !isfinite(c)) {
            C[index] = half((hi + lo) + beta * c);
            return;
        }
        float bhi, blo;
        two_prod(beta, c, bhi, blo);
        dd_add(hi, lo, bhi);
        dd_add(hi, lo, blo);
    }
    if (!isfinite(hi) || !isfinite(lo)) {
        C[index] = half(hi + lo);
        return;
    }
    C[index] = dd_to_half(hi, lo);
}

// Half contraction. One simdgroup owns an 8×8 tile; each lane owns two outputs.
// The dot is a double-float sum of exact f16 products, then one rounding to half.
kernel void contract_general_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant GeneralParams& p [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint n0 = tgid.x * 8;
    const uint m0 = tgid.y * 8;
    const uint batch_id = tgid.z;
    if (m0 >= p.m || n0 >= p.n || batch_id >= p.batch) {
        return;
    }
    const uint a_batch = p.a_offset + decode(batch_id, p.a_batch);
    const uint b_batch = p.b_offset + decode(batch_id, p.b_batch);
    const uint c_batch = p.c_offset + decode(batch_id, p.c_batch);

    float hi0 = 0.0f, lo0 = 0.0f, hi1 = 0.0f, lo1 = 0.0f;
    const uint e0 = lane;
    const uint e1 = lane + 32;
    const uint r0 = e0 >> 3, c0 = e0 & 7;
    const uint r1 = e1 >> 3, c1 = e1 & 7;
    const uint gm0 = m0 + r0, gn0 = n0 + c0;
    const uint gm1 = m0 + r1, gn1 = n0 + c1;
    const bool live0 = gm0 < p.m && gn0 < p.n;
    const bool live1 = gm1 < p.m && gn1 < p.n;

    for (uint k = 0; k < p.k; ++k) {
        const uint kb = decode(k, p.b_k);
        const uint ka = decode(k, p.a_k);
        if (live0) {
            float prod = float(A[a_batch + decode(gm0, p.a_m) + ka]) *
                float(B[b_batch + decode(gn0, p.b_n) + kb]);
            dd_add(hi0, lo0, prod);
        }
        if (live1) {
            float prod = float(A[a_batch + decode(gm1, p.a_m) + ka]) *
                float(B[b_batch + decode(gn1, p.b_n) + kb]);
            dd_add(hi1, lo1, prod);
        }
    }
    if (live0) {
        store_dd(C, c_batch + decode(gm0, p.c_m) + decode(gn0, p.c_n), hi0, lo0, p.alpha, p.beta);
    }
    if (live1) {
        store_dd(C, c_batch + decode(gm1, p.c_m) + decode(gn1, p.c_n), hi1, lo1, p.alpha, p.beta);
    }
}

// Fixed-order sum of Split-K partials. α and β are applied here and nowhere else on this path.
template <typename T>
inline void contract_reduce_body(
    device const float* partial, device T* C, constant ReduceParams& p, uint tid) {
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
    uint index = p.c_offset + decode(b, p.c_batch) + decode(row, p.c_m) + decode(col, p.c_n);
    float out = p.alpha * acc;
    if (p.beta != 0.0f) {
        out += p.beta * float(C[index]);
    }
    C[index] = T(out);
}

kernel void contract_reduce_f32(
    device const float* partial [[buffer(0)]],
    device float* C [[buffer(1)]],
    constant ReduceParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
    contract_reduce_body<float>(partial, C, p, tid);
}

// C = β·C. Used when α = 0 or K = 0 and β ≠ 0, so A and B are not referenced.
template <typename T>
inline void contract_scale_body(device T* C, constant ScaleParams& p, uint tid) {
    if (tid >= p.count) {
        return;
    }
    uint b, row, col;
    unpack_mn(tid, p.m, p.n, b, row, col);
    uint index = p.c_offset + decode(b, p.c_batch) + decode(row, p.c_m) + decode(col, p.c_n);
    C[index] = T(p.beta * float(C[index]));
}

kernel void contract_scale_f32(
    device float* C [[buffer(0)]],
    constant ScaleParams& p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    contract_scale_body<float>(C, p, tid);
}

kernel void contract_scale_f16(
    device half* C [[buffer(0)]],
    constant ScaleParams& p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    contract_scale_body<half>(C, p, tid);
}

// C = 0. Does not load C, so a NaN already in C cannot survive β = 0.
template <typename T>
inline void contract_zero_body(device T* C, constant ScaleParams& p, uint tid) {
    if (tid >= p.count) {
        return;
    }
    uint b, row, col;
    unpack_mn(tid, p.m, p.n, b, row, col);
    uint index = p.c_offset + decode(b, p.c_batch) + decode(row, p.c_m) + decode(col, p.c_n);
    C[index] = T(0);
}

kernel void contract_zero_f32(
    device float* C [[buffer(0)]],
    constant ScaleParams& p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    contract_zero_body<float>(C, p, tid);
}

kernel void contract_zero_f16(
    device half* C [[buffer(0)]],
    constant ScaleParams& p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    contract_zero_body<half>(C, p, tid);
}
