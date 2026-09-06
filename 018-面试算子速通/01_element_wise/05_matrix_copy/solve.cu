#include "common.cuh"


__global__ void copy_kernel(const float* __restrict__ A, float* __restrict__ B, int n) {

    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    const size_t n4 = n / 4;

    const float4* A4 = reinterpret_cast<const float4*>(A);
    float4* B4 = reinterpret_cast<float4*>(B);
    for (size_t i = tid; i < n4; i += stride)
        B4[i] = __ldg(A4 + i);

    for (size_t i = n4 * 4 + tid; i < n; i += stride)
        B[i] = __ldg(A + i);
}

void solve(const float* A, float* B, int N) {
    const size_t n = N * N;
    const int threads = 256;
    int blocks = CEIL(n / 4, threads);
    if (blocks < 1)
        blocks = 1;
    if (blocks > 1024)
        blocks = 1024;
    copy_kernel<<<blocks, threads>>>(A, B, n);
    cudaDeviceSynchronize();
}
