#include "common.cuh"

// 交替合并：out = [A0, B0, A1, B1, ...]。
// 每两个输出元素组合成一个 float2；四个输出组合成一个 float4。

constexpr int THREADS = 256;
constexpr int MAX_BLOCKS = 1024;

__global__ void interleave_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ out, int N) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int n2 = N / 2;

    const float2* A2 = reinterpret_cast<const float2*>(A);
    const float2* B2 = reinterpret_cast<const float2*>(B);
    float4* out4 = reinterpret_cast<float4*>(out);

    // 向量化主循环：每次写 4 个输出元素。
    for (int i = tid; i < n2; i += stride) {
        float2 a = A2[i];
        float2 b = B2[i];
        out4[i] = make_float4(a.x, b.x, a.y, b.y);
    }

    // 尾循环：剩余不足 4 个输出的部分按 float2 处理。
    float2* out2 = reinterpret_cast<float2*>(out);
    for (int i = n2 * 2 + tid; i < N; i += stride)
        out2[i] = make_float2(A[i], B[i]);
}

void solve(const float* A, const float* B, float* output, int N) {
    int blocks = CEIL(N / 2, THREADS);
    if (blocks < 1)
        blocks = 1;
    if (blocks > MAX_BLOCKS)
        blocks = MAX_BLOCKS;
    interleave_kernel<<<blocks, THREADS>>>(A, B, output, N);
    cudaDeviceSynchronize();
}
