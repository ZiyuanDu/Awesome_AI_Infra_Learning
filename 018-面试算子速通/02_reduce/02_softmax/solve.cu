#include "common.cuh"


constexpr int BLOCK = 256;

__global__ void softmax_partial(const float* __restrict__ x, float* __restrict__ block_m,
                                float* __restrict__ block_l, int N) {
    float m = -INFINITY, l = 0.f;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int n4 = N / 4;
    const float4* x4 = reinterpret_cast<const float4*>(x);

    for (int i = tid; i < n4; i += stride) {
        float4 a = x4[i];
        onlineUpdate(m, l, a.x);
        onlineUpdate(m, l, a.y);
        onlineUpdate(m, l, a.z);
        onlineUpdate(m, l, a.w);
    }
    for (int i = n4 * 4 + tid; i < N; i += stride)
        onlineUpdate(m, l, x[i]);

    blockOnlineReduce<BLOCK>(m, l);
    if (threadIdx.x == 0) {
        block_m[blockIdx.x] = m;
        block_l[blockIdx.x] = l;
    }
}

__global__ void softmax_merge(const float* __restrict__ block_m, const float* __restrict__ block_l,
                              float* __restrict__ ml, int nBlocks) {
    float m = -INFINITY, l = 0.f;
    for (int i = threadIdx.x; i < nBlocks; i += blockDim.x)
        onlineMerge(m, l, block_m[i], block_l[i]);
    blockOnlineReduce<BLOCK>(m, l);
    if (threadIdx.x == 0) {
        ml[0] = m;
        ml[1] = l;
    }
}

__global__ void softmax_write(const float* __restrict__ x, float* __restrict__ y,
                              const float* __restrict__ ml, int N) {
    const float m = ml[0];
    const float inv = 1.f / ml[1];
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    // 逆序：第二遍读 x 时更容易命中 K1 留下的 L2（CACHE_OPT）
    for (int i = N - 1 - tid; i >= 0; i -= stride)
        y[i] = __expf(x[i] - m) * inv;
}

void solve(const float* input, float* output, int N) {
    int sm = 0;
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
    int blocks = sm > 0 ? sm * 4 : 128;
    int need = CEIL(N, BLOCK);
    if (need < blocks)
        blocks = need;
    if (blocks < 1)
        blocks = 1;

    // 临时 (m,ℓ)；容量按需扩，避免每次计时都 malloc
    static float* d_bm = nullptr;
    static float* d_bl = nullptr;
    static float* d_ml = nullptr;
    static int cap = 0;
    if (blocks > cap) {
        cudaFree(d_bm);
        cudaFree(d_bl);
        if (!d_ml)
            cudaMalloc(&d_ml, 2 * sizeof(float));
        cudaMalloc(&d_bm, blocks * sizeof(float));
        cudaMalloc(&d_bl, blocks * sizeof(float));
        cap = blocks;
    }

    softmax_partial<<<blocks, BLOCK>>>(input, d_bm, d_bl, N);
    softmax_merge<<<1, BLOCK>>>(d_bm, d_bl, d_ml, blocks);
    softmax_write<<<blocks, BLOCK>>>(input, output, d_ml, N);
    cudaDeviceSynchronize();
}
