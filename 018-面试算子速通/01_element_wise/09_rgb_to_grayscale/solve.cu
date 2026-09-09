#include "common.cuh"

// RGB 转灰度：gray = 0.299R + 0.587G + 0.114B。
// 输入按 float3 布局，一个像素一个 float3。

constexpr int THREADS = 256;
constexpr int MAX_BLOCKS = 1024;

__device__ __forceinline__ float rgb_to_gray(float3 p) {
    return 0.299f * p.x + 0.587f * p.y + 0.114f * p.z;
}

__global__ void rgb_to_gray_kernel(const float* __restrict__ in, float* __restrict__ out, int nPix) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    const float3* in3 = reinterpret_cast<const float3*>(in);

    for (size_t i = tid; i < nPix; i += stride) {
        out[i] = rgb_to_gray(in3[i]);
    }
}

void solve(const float* input, float* output, int width, int height) {
    const int nPix = width * height;
    int blocks = CEIL(nPix, THREADS);
    if (blocks < 1)
        blocks = 1;
    if (blocks > MAX_BLOCKS)
        blocks = MAX_BLOCKS;
    rgb_to_gray_kernel<<<blocks, THREADS>>>(input, output, nPix);
    cudaDeviceSynchronize();
}
