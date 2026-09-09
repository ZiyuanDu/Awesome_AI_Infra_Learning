#include "bench.cuh"

void solve(const float* input, float* output, int N, int k);

#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <random>

// CPU reference: k largest in descending order
static void topKCpu(std::vector<float> v, int k, std::vector<float>& out) {
    std::partial_sort(v.begin(), v.begin() + k, v.end(), std::greater<float>());
    out.assign(v.begin(), v.begin() + k);
}

static bool close(float got, float ref) {
    return std::fabs(got - ref) <= 1e-5f * std::max(1.f, std::fabs(ref));
}

static void check(const std::vector<float>& in, int k, const char* name) {
    const int N = (int)in.size();
    std::vector<float> ref, got((size_t)k);
    topKCpu(in, k, ref);

    float *dIn = nullptr, *dOut = nullptr;
    cudaMalloc(&dIn, (size_t)N * sizeof(float));
    cudaMalloc(&dOut, (size_t)k * sizeof(float));
    cudaMemcpy(dIn, in.data(), (size_t)N * sizeof(float), cudaMemcpyHostToDevice);
    solve(dIn, dOut, N, k);
    cudaMemcpy(got.data(), dOut, (size_t)k * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < k; ++i) {
        if (!close(got[i], ref[i])) {
            printf("FAIL %s  [%d]  %g vs %g\n", name, i, got[i], ref[i]);
            for (int t = 0; t < k && t < 10; ++t)
                printf("   got[%d]=%g  ref[%d]=%g\n", t, got[t], t, ref[t]);
            cudaFree(dIn);
            cudaFree(dOut);
            return;
        }
    }
    printf("OK  %s\n", name);
    cudaFree(dIn);
    cudaFree(dOut);
}

static void checkRand(int N, int k, const char* name) {
    std::vector<float> in((size_t)N);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1e3f, 1e3f);
    for (auto& x : in) x = dist(rng);
    check(in, k, name);
}

int main() {
    // Problem examples
    check({1.f, 5.f, 3.f, 2.f, 4.f}, 3, "ex1");            // {5,4,3}
    check({7.2f, -1.f, 3.3f, 8.8f, 2.2f}, 2, "ex2");       // {8.8,7.2}

    // Edges
    check({1.f}, 1, "n1");
    check({-5.f, -3.f, -1.f, -4.f}, 2, "neg");             // {-1,-3}
    check({5.f, 5.f, 5.f, 1.f, 2.f}, 3, "dup");            // {5,5,5}
    check({0.f, 0.f, -0.f, 3.f}, 2, "signed-zero");
    check({1.f, 2.f, 3.f}, 3, "k=N");
    check({9.f, 8.f, 7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f}, 4, "sorted");

    // Random moderate
    checkRand(1000, 7, "rand-1k-k7");
    checkRand(10007, 50, "rand-10k-k50");
    checkRand(100000, 1, "rand-100k-k1");
    checkRand(1 << 20, 100, "rand-1M-k100");
    checkRand(1 << 20, 5000, "rand-1M-k5k");

    // Deterministic stress
    {
        std::vector<float> asc(10007), desc(10007);
        for (int i = 0; i < 10007; ++i) {
            asc[i] = (float)i;
            desc[i] = (float)(10007 - i);
        }
        check(asc, 50, "asc-10k-k50");
        check(desc, 50, "desc-10k-k50");
    }
    checkRand(1 << 20, 9000, "rand-1M-k9k");   // k > SORT_CAP -> host fallback
    check(std::vector<float>(100000, 7.f), 50, "all-same-100k");  // dup fallback

    // Perf: LeetGPU measured config N=50M, k=100
    const int N = 50000000, k = 100;
    std::vector<float> in((size_t)N);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1e3f, 1e3f);
    for (auto& x : in) x = dist(rng);

    float *dIn = nullptr, *dOut = nullptr;
    cudaMalloc(&dIn, (size_t)N * sizeof(float));
    cudaMalloc(&dOut, (size_t)k * sizeof(float));
    cudaMemcpy(dIn, in.data(), (size_t)N * sizeof(float), cudaMemcpyHostToDevice);

    solve(dIn, dOut, N, k);  // warm + correctness of the big case
    cudaDeviceSynchronize();
    std::vector<float> got((size_t)k);
    cudaMemcpy(got.data(), dOut, (size_t)k * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<float> ref;
    topKCpu(in, k, ref);
    int bad = 0;
    for (int i = 0; i < k; ++i)
        if (!close(got[i], ref[i])) ++bad;
    printf("%s  big-50M-k100\n", bad ? "FAIL" : "OK");

    float ms = timeMsMedian([&] { solve(dIn, dOut, N, k); });
    printf("RUN  N=%d k=%d\n", N, k);
    printf("     %.3f ms (median)  %.0f GB/s (input read once, ~2 passes)\n",
           ms, (double)N * sizeof(float) / (ms * 1e6));

    cudaFree(dIn);
    cudaFree(dOut);
    return 0;
}
