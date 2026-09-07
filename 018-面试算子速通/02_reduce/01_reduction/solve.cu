#include "common.cuh"

constexpr int BLOCK = 256;

__device__ __forceinline__ float sum4(float4 a) { return a.x + a.y + a.z + a.w; }


__global__ void reduce_sum(const float* __restrict__ x, float* __restrict__ out, int N) {

    float v = 0.f;

    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    const size_t n4 = (size_t)N / 4;

    const float4* x4 = reinterpret_cast<const float4*>(x);

    for (size_t i = tid; i < n4; i += stride) {
        v += sum4(x4[i]);
    }

    for (size_t i = n4 * 4 + tid; i < (size_t)N; i += stride) {
        v += x[i];
    }

    v = blockReduceSum<BLOCK>(v);
    if (threadIdx.x == 0) {
        atomicAdd(out, v);
    }
}

void solve(const float* input, float* output, int N) {
    cudaMemset(output, 0, sizeof(float));

    int sm = 0;
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
    int blocks = sm > 0 ? sm * 2 : 128;
    const int need = CEIL(N, BLOCK);
    if (need < blocks)
        blocks = need;
    if (blocks < 1)
        blocks = 1;

    reduce_sum<<<blocks, BLOCK>>>(input, output, N);
    cudaDeviceSynchronize();
}
