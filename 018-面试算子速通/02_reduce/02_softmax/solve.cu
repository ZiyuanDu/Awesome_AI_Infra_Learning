#include "common.cuh"

// 一维 softmax：multi-block online。
// K1 统计每个 block 的 (max, sum)，K2 合并全局 (max, sum)，K3 归一化写回。
// 相比 naive 3-pass 少读一遍输入；见本目录 README.md。

constexpr int BLOCK = 256;

__global__ void softmax_stats(const float* __restrict__ x, float2* __restrict__ blk, int N) {
    float m = -INFINITY, l = 0.f;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int n4 = N / 4;
    const float4* x4 = reinterpret_cast<const float4*>(x);

    // 向量化主循环：onlineUpdate 逐个元素维护 (m, l)。
    for (int i = tid; i < n4; i += stride) {
        float4 a = x4[i];
        onlineUpdate(m, l, a.x);
        onlineUpdate(m, l, a.y);
        onlineUpdate(m, l, a.z);
        onlineUpdate(m, l, a.w);
    }
    // 标量尾循环。
    for (int i = n4 * 4 + tid; i < N; i += stride)
        onlineUpdate(m, l, x[i]);

    // 把 block 内线程状态归约成一个 (m, l)。
    blockOnlineReduce<BLOCK>(m, l);
    if (threadIdx.x == 0)
        blk[blockIdx.x] = make_float2(m, l);
}

__global__ void softmax_merge(const float2* __restrict__ blk, float2* __restrict__ ml, int nBlocks) {
    float m = -INFINITY, l = 0.f;
    // 一个 block 合并所有 block 的部分状态。
    for (int i = threadIdx.x; i < nBlocks; i += blockDim.x)
        onlineMerge(m, l, blk[i].x, blk[i].y);
    blockOnlineReduce<BLOCK>(m, l);
    if (threadIdx.x == 0)
        ml[0] = make_float2(m, l);
}

__global__ void softmax_norm(const float* __restrict__ x, float* __restrict__ y,
                             const float2* __restrict__ ml, int N) {
    const float m = ml[0].x;
    const float inv = 1.f / ml[0].y;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int n4 = N / 4;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* y4 = reinterpret_cast<float4*>(y);

    // 逆序遍历：优先命中 stats 阶段留在 L2 里的数据尾部。
    for (int i = n4 - 1 - tid; i >= 0; i -= stride) {
        float4 a = x4[i];
        y4[i] = make_float4(__expf(a.x - m) * inv, __expf(a.y - m) * inv, __expf(a.z - m) * inv,
                            __expf(a.w - m) * inv);
    }
    // 标量尾循环。
    for (int i = n4 * 4 + tid; i < N; i += stride)
        y[i] = __expf(x[i] - m) * inv;
}

void solve(const float* input, float* output, int N) {
    // 选择与 SM 数量和 N 都匹配的 block 数。
    int sm = 0;
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
    int blocks = sm > 0 ? sm * 4 : 128;
    int need = CEIL(N, BLOCK);
    if (need < blocks)
        blocks = need;
    if (blocks < 1)
        blocks = 1;

    // 函数内 static workspace：跨调用复用，避免每次都 malloc。
    static float2* d_blk = nullptr;
    static float2* d_ml = nullptr;
    static int cap = 0;
    if (blocks > cap) {
        cudaFree(d_blk);
        if (!d_ml)
            cudaMalloc(&d_ml, sizeof(float2));
        cudaMalloc(&d_blk, (size_t)blocks * sizeof(float2));
        cap = blocks;
    }

    softmax_stats<<<blocks, BLOCK>>>(input, d_blk, N);
    softmax_merge<<<1, BLOCK>>>(d_blk, d_ml, blocks);
    softmax_norm<<<blocks, BLOCK>>>(input, output, d_ml, N);
    cudaDeviceSynchronize();
}
