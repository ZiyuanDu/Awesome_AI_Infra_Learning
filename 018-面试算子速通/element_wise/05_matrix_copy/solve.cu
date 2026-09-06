#include "common.cuh"


__global__ void copy_kernel(const float* __restrict__ A, float* __restrict__ B, int n) {

    // 线程 ID
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 线程步长
    const int stride = gridDim.x * blockDim.x;
    // 4 字节对齐的元素个数
    const int n4 = n / 4;

    // 4 字节对齐的指针
    const float4* A4 = reinterpret_cast<const float4*>(A);
    float4* B4 = reinterpret_cast<float4*>(B);
    // 4 字节对齐的元素拷贝
    for (int i = tid; i < n4; i += stride)
        B4[i] = __ldg(A4 + i);

    // 剩余元素拷贝
    for (int i = n4 * 4 + tid; i < n; i += stride)
        B[i] = __ldg(A + i);
}

void solve(const float* A, float* B, int N) {
    const int n = N * N;
    const int threads = 256;
    int blocks = CEIL(n / 4, threads);
    if (blocks < 1)
        blocks = 1;
    if (blocks > 1024)
        blocks = 1024;
    copy_kernel<<<blocks, threads>>>(A, B, n);
    cudaDeviceSynchronize();
}
