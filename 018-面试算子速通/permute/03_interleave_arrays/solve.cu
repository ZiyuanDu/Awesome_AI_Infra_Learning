#include "common.cuh"

/*
 * SoA → AoS：out[2i]=A[i], out[2i+1]=B[i]。数值不动，只换下标 → permute。
 * 读 A/B 的 float2，拼成 float4 一次写 16B；相邻线程写相邻 float4，读写都合并。
 * 不要 reduce：没有多→少，也没有线程间通信。
 */
__global__ void interleave_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ out, int N) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int n2 = N / 2;

    const float2* A2 = reinterpret_cast<const float2*>(A);
    const float2* B2 = reinterpret_cast<const float2*>(B);
    float4* out4 = reinterpret_cast<float4*>(out);

    for (int i = tid; i < n2; i += stride) {
        float2 a = A2[i];
        float2 b = B2[i];
        out4[i] = make_float4(a.x, b.x, a.y, b.y);
    }

    float2* out2 = reinterpret_cast<float2*>(out);
    for (int i = n2 * 2 + tid; i < N; i += stride)
        out2[i] = make_float2(A[i], B[i]);
}

void solve(const float* A, const float* B, float* output, int N) {
    const int threads = 256;
    int blocks = CEIL(N / 2, threads);
    if (blocks < 1)
        blocks = 1;
    if (blocks > 1024)
        blocks = 1024;
    interleave_kernel<<<blocks, threads>>>(A, B, output, N);
    cudaDeviceSynchronize();
}
