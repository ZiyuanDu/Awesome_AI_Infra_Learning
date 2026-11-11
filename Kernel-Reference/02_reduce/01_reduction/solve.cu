#include "common.cuh"

// 一维数组求和：每个 block 做局部归约，再 atomicAdd 到全局标量。
// 主循环用 float4 减少访存指令，尾循环处理 N % 4。

constexpr int BLOCK = 256;

__device__ __forceinline__ float sum4(float4 a) { return a.x + a.y + a.z + a.w; }


__global__ void reduce_sum(const float* __restrict__ x, float* __restrict__ out, int N) {

    float partial = 0.f;

    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    const size_t n4 = (size_t)N / 4;

    const float4* x4 = reinterpret_cast<const float4*>(x);

    // 向量化主循环：一个线程一次累加 4 个 float。
    for (size_t i = tid; i < n4; i += stride) {
        partial += sum4(x4[i]);
    }

    // 标量尾循环。
    for (size_t i = n4 * 4 + tid; i < (size_t)N; i += stride) {
        partial += x[i];
    }

    partial = blockReduceSum<BLOCK>(partial);
    if (threadIdx.x == 0) {
        atomicAdd(out, partial);
    }
}

void solve(const float* input, float* output, int N) {
    cudaMemset(output, 0, sizeof(float));

    // 根据 SM 数量选择 block 数，避免过多空 block 或过少并行度。
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
