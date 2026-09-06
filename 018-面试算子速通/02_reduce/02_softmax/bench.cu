#include "bench.cuh"

void solve(const float* input, float* output, int N);

#include <cmath>
#include <cstdio>
#include <vector>

static void softmaxCpu(const float* in, float* out, int N) {
    float m = in[0];
    for (int i = 1; i < N; ++i)
        m = fmaxf(m, in[i]);
    double sum = 0.0;
    for (int i = 0; i < N; ++i)
        sum += exp((double)in[i] - (double)m);
    for (int i = 0; i < N; ++i)
        out[i] = (float)(exp((double)in[i] - (double)m) / sum);
}

static bool close(float got, float ref) {
    float den = fmaxf(1e-6f, fabsf(ref));
    return fabsf(got - ref) / den <= 2e-3f || fabsf(got - ref) <= 2e-4f;
}

static void check(const std::vector<float>& in, const char* name) {
    const int N = (int)in.size();
    std::vector<float> ref(N), got(N);
    softmaxCpu(in.data(), ref.data(), N);

    float *dIn = nullptr, *dOut = nullptr;
    cudaMalloc(&dIn, N * sizeof(float));
    cudaMalloc(&dOut, N * sizeof(float));
    cudaMemcpy(dIn, in.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    solve(dIn, dOut, N);
    cudaMemcpy(got.data(), dOut, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        if (!close(got[i], ref[i])) {
            printf("FAIL %s  [%d]  %g vs %g\n", name, i, got[i], ref[i]);
            std::exit(1);
        }
    }
    printf("OK  %s\n", name);
    cudaFree(dIn);
    cudaFree(dOut);
}

static void test(int N, const char* name, double bytes = 0) {
    std::vector<float> in(N);
    for (int i = 0; i < N; ++i)
        in[i] = (float)((i % 17) - 8) * 0.5f;
    check(in, name);

    if (bytes > 0) {
        float *dIn = nullptr, *dOut = nullptr;
        cudaMalloc(&dIn, N * sizeof(float));
        cudaMalloc(&dOut, N * sizeof(float));
        cudaMemcpy(dIn, in.data(), N * sizeof(float), cudaMemcpyHostToDevice);
        float ms = timeMs([&] { solve(dIn, dOut, N); }, 10, 30);
        printf("    %.3f ms  %.0f GB/s\n", ms, bytes / (ms * 1e6));
        cudaFree(dIn);
        cudaFree(dOut);
    }
}

int main() {
    check({1.f, 2.f, 3.f}, "ex1");
    check({-10.f, -5.f, 0.f, 5.f, 10.f}, "ex2");
    test(1, "n1");
    test(3, "tail");
    test(10007, "odd");

    // online：读 x 两遍 + 写 y 一遍 ≈ 3N floats
    const int N = 500000;
    test(N, "leetgpu", 3.0 * N * sizeof(float));
}
