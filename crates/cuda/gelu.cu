// GELU and GELU' kernels for compute_61. Compiled once via NVRTC in CudaBackend::new.

extern "C" __global__ void gelu_kernel(float *y, const float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = 0.5f * v * (1.0f + erff(v * 0.7071067811865476f));
    }
}

extern "C" __global__ void gelu_backward_kernel(float *dx, const float *pre, const float *dy,
                                                int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = pre[i];
        float cdf = normcdff(x);
        float pdf = expf(-0.5f * x * x) * 0.3989422804014327f;
        dx[i] = dy[i] * (cdf + x * pdf);
    }
}
