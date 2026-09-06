#include "bench.cuh"

void solve(const float* A, float* B, int N);

#include <vector>

static void copyCpu(const float* A, float* B, int N) {
    const int n = N * N;
    for (int i = 0; i < n; ++i)
        B[i] = A[i];
}

static void test(int N, const char* name, double bytes = 0, bool check = true) {
    std::vector<float> A((size_t)N * N);
    for (size_t i = 0; i < A.size(); ++i)
        A[i] = (float)(i % 17);
    bench(name, {&A}, (size_t)N * N,
          [=](const float* const* a, float* out) { copyCpu(a[0], out, N); },
          [=](const float* const* a, float* out) { solve(a[0], out, N); }, bytes, 0, check);
}

int main() {
    test(1, "1x1");
    test(2, "2x2");
    test(3, "3x3");
    test(17, "tail");

    // LeetGPU N=4096 (~128MB RW) — borderline L2; no GB/s here
    test(4096, "leetgpu");

    // Multiple disjoint matrices → footprint >> L2
    {
        const int N = 4096;
        const size_t elems = (size_t)N * N;
        const double bytes1 = 2.0 * elems * sizeof(float);
        const int tiles = 8;  // ~1GB total traffic per sample
        float *dA = nullptr, *dB = nullptr;
        cudaMalloc(&dA, tiles * elems * sizeof(float));
        cudaMalloc(&dB, tiles * elems * sizeof(float));
        cudaMemset(dA, 1, tiles * elems * sizeof(float));

        float ms = timeMsMedian([&] {
            for (int t = 0; t < tiles; ++t)
                solve(dA + (size_t)t * elems, dB + (size_t)t * elems, N);
        });
        printf("RUN bw (%dx tiles)\n", tiles);
        printf("    %.3f ms (median)  %.0f GB/s\n", ms, (tiles * bytes1) / (ms * 1e6));
        cudaFree(dA);
        cudaFree(dB);
    }
}
