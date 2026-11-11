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

static void fillPattern(std::vector<float>& in) {
    for (size_t i = 0; i < in.size(); ++i)
        in[i] = (float)((i % 17) - 8) * 0.5f;
}

// online: read x twice + write y ≈ 3N floats
static constexpr double kBytesPerElem = 3.0 * sizeof(float);

static void check(const std::vector<float>& in, const char* name) {
    const int N = (int)in.size();
    std::vector<float> ref(N), got(N);
    softmaxCpu(in.data(), ref.data(), N);

    float *dIn = nullptr, *dOut = nullptr;
    cudaMalloc(&dIn, (size_t)N * sizeof(float));
    cudaMalloc(&dOut, (size_t)N * sizeof(float));
    cudaMemcpy(dIn, in.data(), (size_t)N * sizeof(float), cudaMemcpyHostToDevice);
    solve(dIn, dOut, N);
    cudaMemcpy(got.data(), dOut, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost);

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

static void benchSoftmax(int N, const char* name, bool doCheck) {
    float *dIn = nullptr, *dOut = nullptr;
    cudaMalloc(&dIn, (size_t)N * sizeof(float));
    cudaMalloc(&dOut, (size_t)N * sizeof(float));

    if (doCheck) {
        std::vector<float> in(N);
        fillPattern(in);
        check(in, name);
        cudaMemcpy(dIn, in.data(), (size_t)N * sizeof(float), cudaMemcpyHostToDevice);
    } else {
        cudaMemset(dIn, 1, (size_t)N * sizeof(float));
        printf("RUN %s\n", name);
    }

    const double bytes = (double)N * kBytesPerElem;
    float ms = timeMsMedian([&] { solve(dIn, dOut, N); });
    printf("    N=%d  %.4f ms (median)  %.0f GB/s\n", N, ms, bytes / (ms * 1e6));

    cudaFree(dIn);
    cudaFree(dOut);
}

int main() {
    check({1.f, 2.f, 3.f}, "ex1");
    check({-10.f, -5.f, 0.f, 5.f, 10.f}, "ex2");
    benchSoftmax(1, "n1", true);
    benchSoftmax(3, "tail", true);
    benchSoftmax(10007, "odd", true);

    // LeetGPU timing size — fits in L2; launch/reduce show up in the time
    benchSoftmax(500000, "leetgpu", true);

    // Local BW: larger than L2 so the 2-read + 1-write bound is visible
    benchSoftmax(1 << 24, "bw16M", false);   // 64MB x, 192MB traffic
    benchSoftmax(50000000, "bw50M", false);  // 200MB x, 600MB traffic
}
