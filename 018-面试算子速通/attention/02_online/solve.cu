#include "common.cuh"

/*
 * Softmax Attention — Online（Milakov 2018 代数，单 query 流式扫 K）
 *
 * 一遍维护：
 *   m' = max(m, s)
 *   α  = exp(m - m')          // 相等取 1
 *   p  = exp(s - m')
 *   ℓ' = α ℓ + p
 *   O' = α O + p v            // 未归一化；最后 O/ℓ
 *
 * 与 3-pass 精确等价；K/V 只读一遍。FA 把这段嵌进 KV tile。
 * 并行：一线程一 query（d≤128，Oacc 寄存器）。
 */
__global__ void attn_online_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                                   const float* __restrict__ V, float* __restrict__ O, int M,
                                   int N, int d) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M; i += gridDim.x * blockDim.x) {
        const float* qi = Q + (size_t)i * d;
        float scale = rsqrtf((float)d);

        float m = -INFINITY, l = 0.f;
        float o[128];
        for (int t = 0; t < d; ++t)
            o[t] = 0.f;

        for (int j = 0; j < N; ++j) {
            const float* kj = K + (size_t)j * d;
            const float* vj = V + (size_t)j * d;
            float s = 0.f;
            for (int t = 0; t < d; ++t)
                s += qi[t] * kj[t];
            s *= scale;

            float m_new = fmaxf(m, s);
            float alpha = (m == m_new) ? 1.f : __expf(m - m_new);
            float p = __expf(s - m_new);
            l = l * alpha + p;
            for (int t = 0; t < d; ++t)
                o[t] = o[t] * alpha + p * vj[t];
            m = m_new;
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
    attn_online_kernel<<<blocks, threads>>>(Q, K, V, output, M, N, d);
    cudaDeviceSynchronize();
}
