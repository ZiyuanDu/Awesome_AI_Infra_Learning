#include "solve.h"
#include "bench.cuh"

#include <vector>

static void gemmCpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k) {
            float s = 0.f;
            for (int n = 0; n < N; ++n)
                s += A[m * N + n] * B[n * K + k];
            C[m * K + k] = s;
        }
}

static void fill(std::vector<float>& A, std::vector<float>& B, int M, int N, int K) {
    A.resize(M * N);
    B.resize(N * K);
    for (int i = 0; i < M * N; ++i)
        A[i] = (float)(i % 7);
    for (int i = 0; i < N * K; ++i)
        B[i] = (float)(i % 5);
}

static void test(int M, int N, int K, const char* name) {
    std::vector<float> A, B;
    fill(A, B, M, N, K);
    bench(name, {&A, &B}, (size_t)M * K,
          [=](const float* const* in, float* out) { gemmCpu(in[0], in[1], out, M, N, K); },
          [=](const float* const* in, float* out) { solve(in[0], in[1], out, M, N, K); });
}

int main() {
    test(2, 2, 2, "2x2");
    test(2, 3, 2, "ex2");
    test(17, 19, 13, "tail");
    test(128, 128, 128, "128");
    test(130, 131, 129, "tail2");

    const int M = 8192, N = 6144, K = 4096;
    std::vector<float> A, B;
    fill(A, B, M, N, K);
    bench("leetgpu", {&A, &B}, (size_t)M * K,
          [](const float* const*, float*) {},
          [=](const float* const* in, float* out) { solve(in[0], in[1], out, M, N, K); }, 0.0,
          2.0 * M * N * K, false);
}
