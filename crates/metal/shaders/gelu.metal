#include <metal_stdlib>
using namespace metal;

// MSL has no erf builtin: Abramowitz and Stegun 7.1.26, |error| < 1.5e-7.
static inline float erf_as(float x) {
    float s = x < 0.0f ? -1.0f : 1.0f;
    float a = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * a);
    float poly = (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
                   - 0.284496736f) * t + 0.254829592f) * t;
    return s * (1.0f - poly * exp(-a * a));
}

kernel void gelu_f32(device const float* x [[buffer(0)]],
                     device float* y [[buffer(1)]],
                     constant uint& n [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    float v = x[gid];
    y[gid] = 0.5f * v * (1.0f + erf_as(v * 0.70710678118654752f));
}

kernel void gelu_backward_f32(device const float* pre [[buffer(0)]],
                              device const float* dy [[buffer(1)]],
                              device float* dx [[buffer(2)]],
                              constant uint& n [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    float v = pre[gid];
    float cdf = 0.5f * (1.0f + erf_as(v * 0.70710678118654752f));
    float pdf = exp(-0.5f * v * v) * 0.39894228040143268f;
    dx[gid] = dy[gid] * (cdf + v * pdf);
}
