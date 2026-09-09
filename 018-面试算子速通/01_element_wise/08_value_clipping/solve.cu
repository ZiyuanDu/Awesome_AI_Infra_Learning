#include "common.cuh"

// 数值裁剪：out[i] = min(max(x, lo), hi)。

constexpr int THREADS = 256;
constexpr int MAX_BLOCKS = 1024;

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

        // 向量化主循环。
        for (size_t i = tid; i < n4; i += stride) {
            out4[i] = clip4(in4[i], lo, hi);
        }

        // 标量尾循环。
        for (size_t i = n4 * 4 + tid; i < N; i += stride) {
            out[i] = clip(in[i], lo, hi);
        }
}

void solve(const float* input, float* output, int N, float lo, float hi) {
    int blocks = CEIL(N / 4, THREADS);
    if (blocks < 1)
        blocks = 1;
    if (blocks > MAX_BLOCKS)
        blocks = MAX_BLOCKS;
    clip_kernel<<<blocks, THREADS>>>(input, output, N, lo, hi);
    cudaDeviceSynchronize();
}
