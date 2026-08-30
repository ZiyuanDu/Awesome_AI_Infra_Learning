#include "solve.h"
#include "bench.cuh"

#include <cstdio>
#include <vector>

__global__ void reverse_kernel_naive(float* __restrict__ in, int N);

static void solve_naive(float* in, int N) {
    const int threads = 256;
    int blocks = CEIL(N / 2, threads);
    if (blocks < 1)
        blocks = 1;
    reverse_kernel_naive<<<blocks, threads>>>(in, N);
    cudaDeviceSynchronize();
}

static void test(int N) {
    std::vector<float> in(N);
    for (int i = 0; i < N; ++i)
        in[i] = (float)i;

    std::vector<float> ref(in.rbegin(), in.rend());
    float* dIn = nullptr;
    cudaMalloc(&dIn, N * sizeof(float));

    cudaMemcpy(dIn, in.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    solve(dIn, N);
    std::vector<float> got(N);
    cudaMemcpy(got.data(), dIn, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) {
        if (got[i] != ref[i]) {
            printf("FAIL N=%d  [%d]  %g vs %g\n", N, i, got[i], ref[i]);
            std::exit(1);
        }
    }
    printf("OK  reverse N=%d\n", N);
    cudaFree(dIn);
}

int main() {
    for (int N : {1, 2, 3, 4, 5, 7, 8, 9, 12, 16, 17, 10007})
        test(N);

    const int N = 1 << 22; // 4M 个 float
    float* dIn = nullptr;
    cudaMalloc(&dIn, N * sizeof(float));
    cudaMemset(dIn, 1, N * sizeof(float));

    float ms = timeMs([&] { solve(dIn, N); }, 30, 80);
    printf("vec    %d   %.3f ms  %.0f GB/s\n", N, ms, 2.0 * N * sizeof(float) / (ms * 1e6));
    ms = timeMs([&] { solve_naive(dIn, N); }, 30, 80);
    printf("naive  %d   %.3f ms  %.0f GB/s\n", N, ms, 2.0 * N * sizeof(float) / (ms * 1e6));
    cudaFree(dIn);
}
