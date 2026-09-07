#include "bench.cuh"

void solve(const float* input, float* output, int N);

#include <cmath>
#include <cstdio>
#include <vector>

static void scanCpu(const float* in, float* out, int N) {
    float s = 0.f;
    for (int i = 0; i < N; ++i) {
        s += in[i];
        out[i] = s;
    }
}

static bool close(float got, float ref) {
    float den = fmaxf(1.f, fabsf(ref));
    return fabsf(got - ref) / den <= 2e-4f || fabsf(got - ref) <= 1e-3f;
}

static void fillPattern(std::vector<float>& in) {
    for (size_t i = 0; i < in.size(); ++i)
        in[i] = (float)((i % 13) - 6);
}

static void check(const std::vector<float>& in, const char* name) {
    const int N = (int)in.size();
    std::vector<float> ref(N), got(N);
    scanCpu(in.data(), ref.data(), N);

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

static void checkN(int N, const char* name) {
    std::vector<float> in(N);
    fillPattern(in);
    check(in, name);
}

// Traffic ≈ tile_sum read N + tile_scan read N + write N
static void benchBw(int N, const char* name) {
    float *dIn = nullptr, *dOut = nullptr;
    cudaMalloc(&dIn, (size_t)N * sizeof(float));
    cudaMalloc(&dOut, (size_t)N * sizeof(float));
    cudaMemset(dIn, 1, (size_t)N * sizeof(float));

    const double bytes = 3.0 * (double)N * sizeof(float);
    float ms = timeMsMedian([&] { solve(dIn, dOut, N); });
    printf("RUN %s\n", name);
    printf("    %.3f ms (median)  %.0f GB/s  (≈3N: sum+scan R/W)\n", ms, bytes / (ms * 1e6));

    cudaFree(dIn);
    cudaFree(dOut);
}

int main() {
    check({1.f, 2.f, 3.f, 4.f}, "ex1");
    check({5.f, -2.f, 3.f, 1.f, -4.f}, "ex2");

    checkN(1, "n1");
    checkN(31, "warp");
    checkN(512, "block");
    checkN(2048, "one-tile");
    checkN(2049, "two-tile");
    checkN(10007, "odd");
    checkN(200000, "multi-tile");

    // LeetGPU timing size — correctness only (too small for HBM GB/s)
    checkN(250000, "leetgpu");

    // Boundary: two-level totals path
    checkN(512 * 512 + 3, "l2-totals");

    // Local BW (>> L2)
    benchBw(1 << 24, "bw_16M");
    benchBw(100000000, "bw_1e8");
}
