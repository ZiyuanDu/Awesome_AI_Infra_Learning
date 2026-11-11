#include "common.cuh"

// 矩阵拷贝：B[i] = A[i]。
// 输入是 N×N 矩阵，但这里按一维连续内存拷贝，float4 + __ldg。

constexpr int THREADS = 256;
constexpr int MAX_BLOCKS = 1024;

__global__ void copy_kernel(const float* __restrict__ A, float* __restrict__ B, int n) {

    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    const size_t n4 = n / 4;

    const float4* A4 = reinterpret_cast<const float4*>(A);
    float4* B4 = reinterpret_cast<float4*>(B);
    // 向量化主循环：只读路径显式用 __ldg。
    for (size_t i = tid; i < n4; i += stride)
        B4[i] = __ldg(A4 + i);

    // 标量尾循环。
    for (size_t i = n4 * 4 + tid; i < n; i += stride)
        B[i] = __ldg(A + i);
}

void solve(const float* A, float* B, int N) {
    const size_t n = N * N;
    int blocks = CEIL(n / 4, THREADS);
    if (blocks < 1)
        blocks = 1;
    if (blocks > MAX_BLOCKS)
        blocks = MAX_BLOCKS;
    copy_kernel<<<blocks, THREADS>>>(A, B, n);
    cudaDeviceSynchronize();
}
