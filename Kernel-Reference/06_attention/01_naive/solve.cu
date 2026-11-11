#include "common.cuh"

// Naive 3-pass softmax attention。
// 每个线程负责一个 query：先求 max，再求 sum，最后按权重累加 V。
// 内存访问重复明显，适合作为 online / FA 的对照。

constexpr int MAX_D = 128;

__global__ void attn_naive_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                                  const float* __restrict__ V, float* __restrict__ O, int M,
                                  int N, int d) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M; i += gridDim.x * blockDim.x) {
        const float* qi = Q + (size_t)i * d;
        float scale = rsqrtf((float)d);

        // pass 1：求行内 max。
        float m = -INFINITY;
        for (int j = 0; j < N; ++j) {
            const float* kj = K + (size_t)j * d;
            float s = 0.f;
            for (int t = 0; t < d; ++t)
                s += qi[t] * kj[t];
            m = fmaxf(m, s * scale);
        }

        // pass 2：求 sum(exp)。
        float l = 0.f;
        for (int j = 0; j < N; ++j) {
            const float* kj = K + (size_t)j * d;
            float s = 0.f;
            for (int t = 0; t < d; ++t)
                s += qi[t] * kj[t];
            l += __expf(s * scale - m);
        }

        // pass 3：用概率加权 V，最后再归一化。
        float o[MAX_D];
        for (int t = 0; t < d; ++t)
            o[t] = 0.f;
        for (int j = 0; j < N; ++j) {
            const float* kj = K + (size_t)j * d;
            const float* vj = V + (size_t)j * d;
            float s = 0.f;
            for (int t = 0; t < d; ++t)
                s += qi[t] * kj[t];
            float p = __expf(s * scale - m);
            for (int t = 0; t < d; ++t)
                o[t] += p * vj[t];
        }

        float inv = 1.f / l;
        float* oi = O + (size_t)i * d;
        for (int t = 0; t < d; ++t)
            oi[t] = o[t] * inv;
    }
}

void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    const int threads = 256;
    int blocks = CEIL(M, threads);
    if (blocks < 1)
        blocks = 1;
    if (blocks > 1024)
        blocks = 1024;
    attn_naive_kernel<<<blocks, threads>>>(Q, K, V, output, M, N, d);
    cudaDeviceSynchronize();
}
