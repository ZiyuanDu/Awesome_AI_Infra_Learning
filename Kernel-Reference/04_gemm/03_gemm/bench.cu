#include "bench.cuh"
#include <cuda_fp16.h>

void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta);

#include <cmath>
#include <cstdio>
#include <vector>

static void gemmCpu(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
                    int M, int N, int K, float alpha, float beta) {
    std::vector<float> out(M * N);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double s = 0.0;
            for (int k = 0; k < K; ++k)
                s += (double)A[m * K + k] * (double)B[k * N + n];
            out[m * N + n] = alpha * (float)s + beta * C[m * N + n];
        }
    C.swap(out);
}

static bool close(float got, float ref) {
    float den = fmaxf(1.f, fabsf(ref));
    return fabsf(got - ref) / den <= 5e-2f || fabsf(got - ref) <= 0.15f;  // half 容差
}

static void toHalf(const std::vector<float>& f, half* d, size_t n) {
    std::vector<half> h(n);
    for (size_t i = 0; i < n; ++i)
        h[i] = __float2half(f[i]);
    cudaMemcpy(d, h.data(), n * sizeof(half), cudaMemcpyHostToDevice);
}

static void fromHalf(half* d, std::vector<float>& f, size_t n) {
    std::vector<half> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(half), cudaMemcpyDeviceToHost);
    f.resize(n);
    for (size_t i = 0; i < n; ++i)
        f[i] = __half2float(h[i]);
}

static void check(int M, int N, int K, float alpha, float beta, const char* name) {
    std::vector<float> A(M * K), B(K * N), C(M * N), ref;
    for (int i = 0; i < M * K; ++i)
        A[i] = (float)((i % 7) - 3);
    for (int i = 0; i < K * N; ++i)
        B[i] = (float)((i % 5) - 2);
    for (int i = 0; i < M * N; ++i)
        C[i] = (float)((i % 3) + 1);
    ref = C;
    gemmCpu(A, B, ref, M, N, K, alpha, beta);

    half *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaMalloc(&dA, M * K * sizeof(half));
    cudaMalloc(&dB, K * N * sizeof(half));
    cudaMalloc(&dC, M * N * sizeof(half));
    toHalf(A, dA, A.size());
    toHalf(B, dB, B.size());
    toHalf(C, dC, C.size());
    solve(dA, dB, dC, M, N, K, alpha, beta);

    std::vector<float> got;
    fromHalf(dC, got, M * N);
    for (int i = 0; i < M * N; ++i) {
        if (!close(got[i], ref[i])) {
            printf("FAIL %s  [%d] %g vs %g\n", name, i, got[i], ref[i]);
            std::exit(1);
        }
    }
    printf("OK  %s\n", name);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

static void benchCase(int M, int N, int K, const char* name) {
    std::vector<float> A(M * K), B(K * N), C(M * N, 0.f);
    for (int i = 0; i < M * K; ++i)
        A[i] = (float)((i % 7) - 3);
    for (int i = 0; i < K * N; ++i)
        B[i] = (float)((i % 5) - 2);

    half *dA, *dB, *dC;
    cudaMalloc(&dA, M * K * sizeof(half));
    cudaMalloc(&dB, K * N * sizeof(half));
    cudaMalloc(&dC, M * N * sizeof(half));
    toHalf(A, dA, A.size());
    toHalf(B, dB, B.size());
    toHalf(C, dC, C.size());

    float ms = timeMs([&] { solve(dA, dB, dC, M, N, K, 1.f, 0.f); }, 20, 50);
    const double flops = 2.0 * M * N * K;
    printf("%s  %dx%dx%d  %.3f ms  %.1f TFLOPS\n", name, M, N, K, ms, flops / (ms * 1e9));
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

int main() {
    // 官方例：alpha=1 beta=0 → [[22,28],[49,64]]
    {
        std::vector<float> A = {1, 2, 3, 4, 5, 6};
        std::vector<float> B = {1, 2, 3, 4, 5, 6};
        std::vector<float> C = {1, 1, 1, 1};
        gemmCpu(A, B, C, 2, 2, 3, 1.f, 0.f);
        if (!close(C[0], 22.f) || !close(C[1], 28.f) || !close(C[2], 49.f) || !close(C[3], 64.f)) {
            printf("FAIL ex cpu\n");
            return 1;
        }
        half *dA, *dB, *dC;
        cudaMalloc(&dA, 6 * sizeof(half));
        cudaMalloc(&dB, 6 * sizeof(half));
        cudaMalloc(&dC, 4 * sizeof(half));
        toHalf(A, dA, 6);
        toHalf(B, dB, 6);
        std::vector<float> C0 = {1, 1, 1, 1};
        toHalf(C0, dC, 4);
        solve(dA, dB, dC, 2, 2, 3, 1.f, 0.f);
        std::vector<float> got;
        fromHalf(dC, got, 4);
        const float expv[4] = {22.f, 28.f, 49.f, 64.f};
        for (int i = 0; i < 4; ++i)
            if (!close(got[i], expv[i])) {
                printf("FAIL ex gpu %g\n", got[i]);
                return 1;
            }
        printf("OK  ex1\n");
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
    }

    check(16, 16, 16, 1.f, 0.f, "16");
    check(17, 19, 18, 1.f, 0.f, "tail");
    check(64, 64, 64, 1.2f, 0.5f, "ab");
    check(128, 128, 128, 1.f, 0.f, "128");

    printf("\n--- perf ---\n");
    benchCase(1024, 1024, 1024, "leetgpu");
    benchCase(2048, 2048, 2048, "2k");
}
