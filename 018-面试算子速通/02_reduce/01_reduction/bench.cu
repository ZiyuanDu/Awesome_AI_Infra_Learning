#include "bench.cuh"

void solve(const float* input, float* output, int N);

#include <cmath>
#include <cstdio>
#include <vector>

// Reduction: read N floats once → 1 scalar. bytes = N * sizeof(float).

static float reduceCpu(const float* in, int N) {
    double s = 0.0;
    for (int i = 0; i < N; ++i)
        s += (double)in[i];
    return (float)s;
}

static bool closeRel(float got, float ref) {
    float den = fmaxf(1.f, fabsf(ref));
    return fabsf(got - ref) / den <= 2e-4f;
}

static void fillPattern(std::vector<float>& in) {
    for (size_t i = 0; i < in.size(); ++i)
        in[i] = (float)((i % 13) - 6);
}

static void check(int N, const char* name) {
    std::vector<float> in(N);
    fillPattern(in);
    const float ref = reduceCpu(in.data(), N);

    float *dIn = nullptr, *dOut = nullptr;
    cudaMalloc(&dIn, (size_t)N * sizeof(float));
    cudaMalloc(&dOut, sizeof(float));
    cudaMemcpy(dIn, in.data(), (size_t)N * sizeof(float), cudaMemcpyHostToDevice);

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

static void benchReduce(int N, const char* name, double bytes, bool doCheck) {
    float *dIn = nullptr, *dOut = nullptr;
    cudaMalloc(&dIn, (size_t)N * sizeof(float));
    cudaMalloc(&dOut, sizeof(float));

    if (doCheck) {
        std::vector<float> in(N);
        fillPattern(in);
        cudaMemcpy(dIn, in.data(), (size_t)N * sizeof(float), cudaMemcpyHostToDevice);
        const float ref = reduceCpu(in.data(), N);
        solve(dIn, dOut, N);
        float got = 0.f;
        cudaMemcpy(&got, dOut, sizeof(float), cudaMemcpyDeviceToHost);
        if (!closeRel(got, ref)) {
            printf("FAIL %s  %g vs %g\n", name, got, ref);
            std::exit(1);
        }
        printf("OK  %s\n", name);
    } else {
        cudaMemset(dIn, 1, (size_t)N * sizeof(float));  // pattern irrelevant for BW
        printf("RUN %s\n", name);
    }

    if (bytes > 0) {
        float ms = timeMsMedian([&] { solve(dIn, dOut, N); });
        printf("    %.3f ms (median)  %.0f GB/s\n", ms, bytes / (ms * 1e6));
    }

    cudaFree(dIn);
    cudaFree(dOut);
}

// D2D moves read+write; report 2N so GB/s is comparable bus throughput vs reduce's N-read.
static void benchMemcpyCeiling(int N) {
    float *dA = nullptr, *dB = nullptr;
    cudaMalloc(&dA, (size_t)N * sizeof(float));
    cudaMalloc(&dB, (size_t)N * sizeof(float));
    cudaMemset(dA, 1, (size_t)N * sizeof(float));
    const double bytes = 2.0 * (double)N * sizeof(float);
    float ms = timeMsMedian([&] {
        cudaMemcpy(dB, dA, (size_t)N * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
    });
    printf("RUN memcpy_ceiling\n");
    printf("    %.3f ms (median)  %.0f GB/s  (D2D bus: 2×N floats)\n", ms, bytes / (ms * 1e6));
    cudaFree(dA);
    cudaFree(dB);
}

int main() {
    // Official examples
    {
        std::vector<float> ex1 = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
        float *dIn = nullptr, *dOut = nullptr;
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
    }

    check(1, "n1");
    check(3, "tail");
    check(10007, "odd");

    // LeetGPU timing size (16MB) — correctness only; too small for HBM GB/s
    constexpr int N_leet = 1 << 22;  // 4194304
    benchReduce(N_leet, "leetgpu", /*bytes=*/0, /*doCheck=*/true);

    // Local BW: >> L2; median; traffic = one read of N floats
    constexpr int N_bw = 100000000;  // 400MB
    const double bytes = (double)N_bw * sizeof(float);
    benchReduce(N_bw, "bw", bytes, /*doCheck=*/false);
    benchMemcpyCeiling(N_bw);
}
