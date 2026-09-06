#include "common.cuh"

constexpr int BLOCK = 256;

__global__ void dot_kernel(const float* __restrict__ A, const float* __restrict__ B,
                           float* __restrict__ out, int N) {
    float v = 0.f;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int n4 = N / 4;
    const float4* a4 = reinterpret_cast<const float4*>(A);
    const float4* b4 = reinterpret_cast<const float4*>(B);

    for (int i = tid; i < n4; i += stride) {
        float4 a = a4[i];
        float4 b = b4[i];
        v += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int i = n4 * 4 + tid; i < N; i += stride)
        v += A[i] * B[i];

    v = blockReduceSum<BLOCK>(v);
    if (threadIdx.x == 0)
        atomicAdd(out, v);
}

void solve(const float* A, const float* B, float* output, int N) {
    cudaMemset(output, 0, sizeof(float));

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
