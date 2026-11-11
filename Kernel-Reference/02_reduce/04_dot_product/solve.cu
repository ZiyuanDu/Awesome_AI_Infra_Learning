#include "common.cuh"

// 向量点积：sum(A[i] * B[i])。
// 与 reduction 同骨架，只是多读一个输入向量。

constexpr int BLOCK = 256;

__device__ __forceinline__ float dot4(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__global__ void dot_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ out, int N) {
    float partial = 0.f;
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    const size_t n4 = (size_t)N / 4;

    const float4* a4 = reinterpret_cast<const float4*>(A);
    const float4* b4 = reinterpret_cast<const float4*>(B);

    // 向量化主循环。
    for (size_t i = tid; i < n4; i += stride) {
        float4 a = a4[i];
        float4 b = b4[i];
        partial += dot4(a, b);
    }
    // 标量尾循环。
    for (size_t i = n4 * 4 + tid; i < N; i += stride)
        partial += A[i] * B[i];

    partial = blockReduceSum<BLOCK>(partial);
    if (threadIdx.x == 0) {
        atomicAdd(out, partial);
    }
}

void solve(const float* A, const float* B, float* output, int N) {
    cudaMemset(output, 0, sizeof(float));

    // 根据 SM 数量选择 block 数，兼顾并行度和调度开销。
    int sm = 0;
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
    int blocks = sm > 0 ? sm * 2 : 128;
    int need = CEIL(N, BLOCK);
    if (need < blocks)
        blocks = need;
    if (blocks < 1)
        blocks = 1;

    dot_kernel<<<blocks, BLOCK>>>(A, B, output, N);
    cudaDeviceSynchronize();
}
