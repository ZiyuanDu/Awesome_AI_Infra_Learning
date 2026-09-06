#include "common.cuh"


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
    const int threads = 256;
    int blocks = CEIL(nPix, threads);
    if (blocks < 1)
        blocks = 1;
    if (blocks > 1024)
        blocks = 1024;
    rgb_to_gray_kernel<<<blocks, threads>>>(input, output, nPix);
    cudaDeviceSynchronize();
}
