// Training ops of the core contract, compiled once with gelu.cu in CudaBackend::new.
// Every rounding is explicit (__fadd_rn and friends), so nothing contracts into an FMA.

extern "C" __global__ void add_bias_kernel(float *y, const float *x, const float *b, int n,
                                           int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = __fadd_rn(x[i], b[i % cols]);
    }
}

// One thread per column: s sums rows 0, 1, ... in order, then db[j] + s.
extern "C" __global__ void bias_grad_kernel(float *db, const float *dpre, int rows, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < cols) {
        float s = 0.0f;
        for (int r = 0; r < rows; ++r) {
            s = __fadd_rn(s, dpre[(size_t)r * cols + j]);
        }
        db[j] = __fadd_rn(db[j], s);
    }
}

// One block of MSE_THREADS threads. Thread t handles elements t, t + T, ... and keeps a
// running sum of d * d; the partial sums then meet in a fixed pairwise tree, so the order
// depends only on n.
#define MSE_THREADS 1024
extern "C" __global__ void mse_grad_kernel(float *dy, float *loss, const float *y,
                                           const float *target, int n) {
    __shared__ float partial[MSE_THREADS];
    int tid = threadIdx.x;
    float nf = __int2float_rn(n);
    float sum = 0.0f;
    for (int i = tid; i < n; i += MSE_THREADS) {
        float d = __fsub_rn(y[i], target[i]);
        sum = __fadd_rn(sum, __fmul_rn(d, d));
        dy[i] = __fdiv_rn(__fmul_rn(2.0f, d), nf);
    }
    partial[tid] = sum;
    __syncthreads();
    for (int stride = MSE_THREADS / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial[tid] = __fadd_rn(partial[tid], partial[tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        loss[0] = __fdiv_rn(partial[0], nf);
    }
}

extern "C" __global__ void sgd_step_kernel(float *w, float *dw, float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float step = __fmul_rn(lr, dw[i]);
        w[i] = __fsub_rn(w[i], step);
        dw[i] = 0.0f;
    }
}

extern "C" __global__ void convert_f32_bf16_kernel(unsigned short *y, const unsigned int *x,
                                                   int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned int bits = x[i];
        if ((bits & 0x7fffffffu) > 0x7f800000u) {
            y[i] = (unsigned short)((bits >> 16) | 0x0040u);
        } else {
            unsigned int lsb = (bits >> 16) & 1u;
            y[i] = (unsigned short)((bits + 0x7fffu + lsb) >> 16);
        }
    }
}

extern "C" __global__ void convert_bf16_f32_kernel(unsigned int *y, const unsigned short *x,
                                                   int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = ((unsigned int)x[i]) << 16;
    }
}
