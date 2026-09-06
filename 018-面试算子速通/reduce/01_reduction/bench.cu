#include "bench.cuh"

void solve(const float* input, float* output, int N);

#include <cmath>
#include <cstdio>
#include <vector>

static float reduceCpu(const float* in, int N) {
    double s = 0.0;
    for (int i = 0; i < N; ++i)
        s += in[i];
    return (float)s;
}

static bool closeRel(float got, float ref) {
    float den = fmaxf(1.f, fabsf(ref));
    return fabsf(got - ref) / den <= 2e-4f;
}

static void check(int N, const char* name) {
    std::vector<float> in(N);
    for (int i = 0; i < N; ++i)
        in[i] = (float)((i % 13) - 6);

    float ref = reduceCpu(in.data(), N);
    float* dIn = nullptr;
    float* dOut = nullptr;
    cudaMalloc(&dIn, N * sizeof(float));
    cudaMalloc(&dOut, sizeof(float));
    cudaMemcpy(dIn, in.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    solve(dIn, dOut, N);
    float got = 0.f;
    cudaMemcpy(&got, dOut, sizeof(float), cudaMemcpyDeviceToHost);
    if (!closeRel(got, ref)) {
        printf("FAIL %s  %g vs %g\n", name, got, ref);
        std::exit(1);
    }
    printf("OK  %s\n", name);
    cudaFree(dIn);
    cudaFree(dOut);
}

int main() {
    std::vector<float> ex1 = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    float* dIn = nullptr;
    float* dOut = nullptr;
    cudaMalloc(&dIn, ex1.size() * sizeof(float));
    cudaMalloc(&dOut, sizeof(float));
    cudaMemcpy(dIn, ex1.data(), ex1.size() * sizeof(float), cudaMemcpyHostToDevice);
    solve(dIn, dOut, (int)ex1.size());
    float got = 0.f;
    cudaMemcpy(&got, dOut, sizeof(float), cudaMemcpyDeviceToHost);
    if (!closeRel(got, 36.f)) {
        printf("FAIL ex1  %g vs 36\n", got);
        return 1;
    }
    printf("OK  ex1\n");

    std::vector<float> ex2 = {-2.5f, 1.5f, -1.f, 2.f};
    cudaMemcpy(dIn, ex2.data(), ex2.size() * sizeof(float), cudaMemcpyHostToDevice);
    solve(dIn, dOut, (int)ex2.size());
    cudaMemcpy(&got, dOut, sizeof(float), cudaMemcpyDeviceToHost);
    if (!closeRel(got, 0.f)) {
        printf("FAIL ex2  %g vs 0\n", got);
        return 1;
    }
    printf("OK  ex2\n");
    cudaFree(dIn);
    cudaFree(dOut);

    check(1, "n1");
    check(3, "tail");
    check(10007, "odd");

    const int N = 4194304;
    std::vector<float> in(N);
    for (int i = 0; i < N; ++i)
        in[i] = (float)((i % 13) - 6);
    cudaMalloc(&dIn, N * sizeof(float));
    cudaMalloc(&dOut, sizeof(float));
    cudaMemcpy(dIn, in.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    float ref = reduceCpu(in.data(), N);
    solve(dIn, dOut, N);
    cudaMemcpy(&got, dOut, sizeof(float), cudaMemcpyDeviceToHost);
    if (!closeRel(got, ref)) {
        printf("FAIL leetgpu  %g vs %g\n", got, ref);
        return 1;
    }
    printf("OK  leetgpu\n");

    float ms = timeMs([&] { solve(dIn, dOut, N); }, 10, 30);
    printf("    %.3f ms  %.0f GB/s\n", ms, (double)N * sizeof(float) / (ms * 1e6));
    cudaFree(dIn);
    cudaFree(dOut);
}
