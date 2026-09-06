#include "common.cuh"


__device__ __forceinline__ float clip(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

__device__ __forceinline__ float4 clip4(float4 v, float lo, float hi) {
    return make_float4(clip(v.x, lo, hi), clip(v.y, lo, hi), clip(v.z, lo, hi), clip(v.w, lo, hi));
}

__global__ void clip_kernel(const float* __restrict__ in, float* __restrict__ out, int N, float lo, float hi) {
        const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        const size_t stride = gridDim.x * blockDim.x;
        const size_t n4 = N / 4;

        const float4* in4 = reinterpret_cast<const float4*>(in);
        float4* out4 = reinterpret_cast<float4*>(out);

        for (size_t i = tid; i < n4; i += stride) {
            out4[i] = clip4(in4[i], lo, hi);
        }

        for (size_t i = n4 * 4 + tid; i < N; i += stride) {
            out[i] = clip(in[i], lo, hi);
        }
}

void solve(const float* input, float* output, int N, float lo, float hi) {
    const int threads = 256;
    int blocks = CEIL(N / 4, threads);
    if (blocks < 1)
        blocks = 1;
    if (blocks > 1024)
        blocks = 1024;
    clip_kernel<<<blocks, threads>>>(input, output, N, lo, hi);
    cudaDeviceSynchronize();
}
