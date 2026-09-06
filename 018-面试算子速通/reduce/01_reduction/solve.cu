#include "common.cuh"


constexpr int BLOCK = 256;

__global__ void reduce_kernel(const float* __restrict__ x, float* __restrict__ out, int N) {
    float v = 0.f;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int n4 = N / 4;
    const float4* x4 = reinterpret_cast<const float4*>(x);

    for (int i = tid; i < n4; i += stride) {
        float4 a = x4[i];
        v += a.x + a.y + a.z + a.w;
    }
    for (int i = n4 * 4 + tid; i < N; i += stride)
        v += x[i];

    v = blockReduceSum<BLOCK>(v);
    if (threadIdx.x == 0)
        atomicAdd(out, v);
}

void solve(const float* input, float* output, int N) {
    cudaMemset(output, 0, sizeof(float));

    int sm = 0;
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
    int blocks = sm > 0 ? sm * 2 : 128;
    int need = CEIL(N, BLOCK);
    if (need < blocks)
        blocks = need;
    if (blocks < 1)
        blocks = 1;

    reduce_kernel<<<blocks, BLOCK>>>(input, output, N);
    cudaDeviceSynchronize();
}
