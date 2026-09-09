#include "bench.cuh"

void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d);

#include <cmath>
#include <cstdio>
#include <vector>

static void attnCpu(const float* Q, const float* K, const float* V, float* O, int M, int N, int d) {
    const float scale = 1.f / sqrtf((float)d);
    std::vector<float> s(N);
    for (int i = 0; i < M; ++i) {
        const float* qi = Q + (size_t)i * d;
        float mx = -INFINITY;
        for (int j = 0; j < N; ++j) {
            const float* kj = K + (size_t)j * d;
            float dot = 0.f;
            for (int t = 0; t < d; ++t)
                dot += qi[t] * kj[t];
            s[j] = dot * scale;
            mx = fmaxf(mx, s[j]);
        }
        double sum = 0.0;
        for (int j = 0; j < N; ++j)
            sum += exp((double)s[j] - (double)mx);
        float* oi = O + (size_t)i * d;
        for (int t = 0; t < d; ++t)
            oi[t] = 0.f;
        for (int j = 0; j < N; ++j) {
            float p = (float)(exp((double)s[j] - (double)mx) / sum);
            const float* vj = V + (size_t)j * d;
            for (int t = 0; t < d; ++t)
                oi[t] += p * vj[t];
        }
    }
}

static bool close(float got, float ref) {
    float den = fmaxf(1e-3f, fabsf(ref));
    return fabsf(got - ref) / den <= 5e-3f || fabsf(got - ref) <= 5e-3f;
}

static void check(const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V,
                  int M, int N, int d, const char* name) {
    std::vector<float> ref((size_t)M * d), got((size_t)M * d);
    attnCpu(Q.data(), K.data(), V.data(), ref.data(), M, N, d);

    float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
    cudaMalloc(&dQ, Q.size() * sizeof(float));
    cudaMalloc(&dK, K.size() * sizeof(float));
    cudaMalloc(&dV, V.size() * sizeof(float));
    cudaMalloc(&dO, got.size() * sizeof(float));
    cudaMemcpy(dQ, Q.data(), Q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K.data(), K.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V.data(), V.size() * sizeof(float), cudaMemcpyHostToDevice);
    solve(dQ, dK, dV, dO, M, N, d);
    cudaMemcpy(got.data(), dO, got.size() * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < got.size(); ++i) {
        if (!close(got[i], ref[i])) {
            printf("FAIL %s  [%zu]  %g vs %g\n", name, i, got[i], ref[i]);
            std::exit(1);
        }
    }
    printf("OK  %s\n", name);
    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
}

static void fill(std::vector<float>& a, int seed) {
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = (float)(((int)i * 17 + seed) % 13) - 6.f;
}

int main() {
    // Example 1: M=2 N=3 d=4
    std::vector<float> Q1 = {1, 0, 0, 0, 0, 1, 0, 0};
    std::vector<float> K1 = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    std::vector<float> V1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    check(Q1, K1, V1, 2, 3, 4, "ex1");

    // Example 2: M=1 N=2 d=2
    std::vector<float> Q2 = {1, 2};
    std::vector<float> K2 = {1, 0, 0, 1};
    std::vector<float> V2 = {3, 4, 5, 6};
    check(Q2, K2, V2, 1, 2, 2, "ex2");

    {
        const int M = 7, N = 11, d = 8;
        std::vector<float> Q((size_t)M * d), K((size_t)N * d), V((size_t)N * d);
        fill(Q, 1);
        fill(K, 2);
        fill(V, 3);
        check(Q, K, V, M, N, d, "odd");
    }

    const int M = 512, N = 256, d = 64;
    std::vector<float> Q((size_t)M * d), K((size_t)N * d), V((size_t)N * d);
    fill(Q, 4);
    fill(K, 5);
    fill(V, 6);
    check(Q, K, V, M, N, d, "leetgpu");

    float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
    cudaMalloc(&dQ, Q.size() * sizeof(float));
    cudaMalloc(&dK, K.size() * sizeof(float));
    cudaMalloc(&dV, V.size() * sizeof(float));
    cudaMalloc(&dO, (size_t)M * d * sizeof(float));
    cudaMemcpy(dQ, Q.data(), Q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K.data(), K.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V.data(), V.size() * sizeof(float), cudaMemcpyHostToDevice);
    // 粗算：每 query 读 K、V 各一遍 ≈ 2·M·N·d + 写 O
    double bytes = (2.0 * M * N * d + (double)M * d) * sizeof(float);
    float ms = timeMs([&] { solve(dQ, dK, dV, dO, M, N, d); }, 10, 30);
    printf("    %.3f ms  %.0f GB/s (traffic model)\n", ms, bytes / (ms * 1e6));
    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
}
