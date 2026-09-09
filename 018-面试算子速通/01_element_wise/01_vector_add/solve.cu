#include "common.cuh"

// C[i] = A[i] + B[i]，纯访存型算子。
// 写法：grid-stride + float4 向量化，尾部标量补齐。

constexpr int THREADS = 256;

__device__ __forceinline__ float4 addFloat4(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__global__ void vectorAdd(
    const float* __restrict__ A, 
    const float* __restrict__ B,
    float* __restrict__ C, 
    int N
) {

    // 每个线程负责一个全局索引，按 stride 跨网格循环，避免越界。
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    // float4 主循环：一次处理 4 个连续元素。
    const size_t n4 = N / 4;

    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);

    for (size_t i = tid; i < n4; i += stride) {
        C4[i] = addFloat4(A4[i], B4[i]);
    }

    // 标量尾循环：处理 N % 4 剩下的元素。
    for (size_t i = n4 * 4 + tid; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

void solve(const float* A, const float* B, float* C, int N) {
    const int vecCount = N / 4;
    int blocks = CEIL(vecCount, THREADS);
    if (blocks < 1) {
        blocks = 1;
    }

    vectorAdd<<<blocks, THREADS>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
