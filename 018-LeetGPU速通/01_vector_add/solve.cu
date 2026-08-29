#include "solve.h"

__device__ __forceinline__ float4 addFloat4(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__global__ void vectorAdd(const float* __restrict__ A, const float* __restrict__ B,
                          float* __restrict__ C, int N) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int n4 = N / 4;

    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);

    for (int i = tid; i < n4; i += stride) {
        C4[i] = addFloat4(A4[i], B4[i]);
    }

    for (int i = n4 * 4 + tid; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

void solve(const float* A, const float* B, float* C, int N) {
    const int threads = 256;
    const int n4 = N / 4;
    int blocks = CEIL(n4, threads);
    if (blocks < 1) {
        blocks = 1;
    }

    vectorAdd<<<blocks, threads>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
