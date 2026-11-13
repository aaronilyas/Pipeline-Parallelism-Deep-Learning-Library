#include <metal_stdlib>
using namespace metal;

// Training ops of the core contract. Each formula is evaluated as written, one rounding per
// operation: contraction into fused multiply-add is off for this file.
#pragma METAL fp contract(off)

kernel void add_bias_f32(device const float* x [[buffer(0)]],
                         device const float* b [[buffer(1)]],
                         device float* y [[buffer(2)]],
                         constant uint2& dims [[buffer(3)]],  // (count, cols)
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= dims.x) return;
    y[gid] = x[gid] + b[gid % dims.y];
}

// One thread per column: s = dpre[0][j] + dpre[1][j] + ... in row order, then db[j] + s.
kernel void bias_grad_f32(device const float* dpre [[buffer(0)]],
                          device float* db [[buffer(1)]],
                          constant uint2& dims [[buffer(2)]],  // (rows, cols)
                          uint col [[thread_position_in_grid]]) {
    if (col >= dims.y) return;
    float s = 0.0f;
    for (uint row = 0; row < dims.x; ++row) {
        s = s + dpre[row * dims.y + col];
    }
    db[col] = db[col] + s;
}

// One threadgroup of MSE_THREADS threads. Thread t handles elements t, t + T, ... and keeps a
// running sum of d * d; the partial sums then meet in a fixed pairwise tree. The order depends
// only on n and T, so the same inputs give the same bits.
#define MSE_THREADS 1024
kernel void mse_grad_f32(device const float* y [[buffer(0)]],
                         device const float* target [[buffer(1)]],
                         device float* dy [[buffer(2)]],
                         device float* loss [[buffer(3)]],
                         constant uint& n [[buffer(4)]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint threads [[threads_per_threadgroup]]) {
    threadgroup float partial[MSE_THREADS];
    float nf = float(n);
    float sum = 0.0f;
    for (uint i = tid; i < n; i += threads) {
        float d = y[i] - target[i];
        float sq = d * d;
        sum = sum + sq;
        float twice = 2.0f * d;
        dy[i] = twice / nf;
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial[tid] = partial[tid] + partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        loss[0] = partial[0] / nf;
    }
}

struct SgdArgs {
    float lr;
    uint count;
};

kernel void sgd_step_f32(device float* w [[buffer(0)]],
                         device float* dw [[buffer(1)]],
                         constant SgdArgs& args [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= args.count) return;
    float step = args.lr * dw[gid];
    w[gid] = w[gid] - step;
    dw[gid] = 0.0f;
}

kernel void convert_f32_bf16(device const uint* x [[buffer(0)]],
                             device ushort* y [[buffer(1)]],
                             constant uint& n [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    uint bits = x[gid];
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
        y[gid] = ushort((bits >> 16) | 0x0040u);
        return;
    }
    uint lsb = (bits >> 16) & 1u;
    y[gid] = ushort((bits + 0x7fffu + lsb) >> 16);
}

kernel void convert_bf16_f32(device const ushort* x [[buffer(0)]],
                             device uint* y [[buffer(1)]],
                             constant uint& n [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    y[gid] = uint(x[gid]) << 16;
}
