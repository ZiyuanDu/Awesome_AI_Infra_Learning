#include "common.cuh"


constexpr int TILE = 32;

__global__ void attn_fa2_kernel(const float* __restrict__ Q, const float* __restrict__ K,
                                const float* __restrict__ V, float* __restrict__ O, int M, int N,
                                int d) {
    const int tx = threadIdx.x;
    const int q = blockIdx.x * TILE + tx;
    const int ld = d + 1;
    const int sld = TILE + 1;

    extern __shared__ float smem[];
    float* Qi = smem;
    float* Kj = Qi + TILE * ld;
    float* Vj = Kj + TILE * ld;
    float* S = Vj + TILE * ld;
    float* Oacc = S + TILE * sld;

    for (int t = 0; t < d; ++t)
        Qi[tx * ld + t] = (q < M) ? Q[(size_t)q * d + t] : 0.f;
    for (int t = 0; t < d; ++t)
        Oacc[tx * ld + t] = 0.f;

    float m_i = -INFINITY, l_i = 0.f;
    const float scale = rsqrtf((float)d);
    const int Tc = CEIL(N, TILE);

    for (int jc = 0; jc < Tc; ++jc) {
        const int kv = jc * TILE + tx;
        for (int t = 0; t < d; ++t) {
            Kj[tx * ld + t] = (kv < N) ? K[(size_t)kv * d + t] : 0.f;
            Vj[tx * ld + t] = (kv < N) ? V[(size_t)kv * d + t] : 0.f;
        }
        __syncthreads();

        float row_m = -INFINITY;
#pragma unroll
        for (int y = 0; y < TILE; ++y) {
            const int kj = jc * TILE + y;
            float s = 0.f;
            if (q < M && kj < N) {
                for (int t = 0; t < d; ++t)
                    s += Qi[tx * ld + t] * Kj[y * ld + t];
                s *= scale;
            } else {
                s = -INFINITY;
            }
            S[tx * sld + y] = s;
            row_m = fmaxf(row_m, s);
        }

        if (row_m > -INFINITY) {
            float row_l = 0.f;
#pragma unroll
            for (int y = 0; y < TILE; ++y) {
                const int kj = jc * TILE + y;
                float p = (q < M && kj < N) ? __expf(S[tx * sld + y] - row_m) : 0.f;
                S[tx * sld + y] = p;
                row_l += p;
            }

            const float m_new = fmaxf(m_i, row_m);
            const float alpha = (m_i == m_new) ? 1.f : __expf(m_i - m_new);
            const float beta = (row_m == m_new) ? 1.f : __expf(row_m - m_new);
            l_i = l_i * alpha + row_l * beta;

            for (int t = 0; t < d; ++t) {
                float pv = 0.f;
#pragma unroll
                for (int y = 0; y < TILE; ++y)
                    pv += S[tx * sld + y] * Vj[y * ld + t];
                Oacc[tx * ld + t] = Oacc[tx * ld + t] * alpha + pv * beta;
            }
            m_i = m_new;
        }
        __syncthreads();
    }

    if (q < M && l_i > 0.f) {
        const float inv = 1.f / l_i;
        for (int t = 0; t < d; ++t)
            O[(size_t)q * d + t] = Oacc[tx * ld + t] * inv;
    }
}

void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    const int ld = d + 1;
    const int sld = TILE + 1;
    const size_t smem =
        sizeof(float) * (size_t)(3 * TILE * ld + TILE * sld + TILE * ld);
    const int blocks = CEIL(M, TILE);
    attn_fa2_kernel<<<blocks, TILE, smem>>>(Q, K, V, output, M, N, d);
    cudaDeviceSynchronize();
}
