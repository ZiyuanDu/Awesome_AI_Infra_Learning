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

static void check(const std::vector<float>& in, const char* name) {
    const int N = (int)in.size();
    std::vector<float> ref(N), got(N);
    scanCpu(in.data(), ref.data(), N);

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
        in[i] = (float)((i % 13) - 6);
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
    check({1.f, 2.f, 3.f, 4.f}, "ex1");
    check({5.f, -2.f, 3.f, 1.f, -4.f}, "ex2");
    test(1, "n1");
    test(31, "warp");
    test(512, "tile");
    test(513, "tile+1");
    test(10007, "odd");
    test(200000, "two-level");  // > 512*512? 200k/512 < 512, still one level on totals

    const int N = 250000;
    test(N, "leetgpu", 2.0 * N * sizeof(float));

    // 约束上限量级（只对答案，不强制计时）
    test(512 * 512 + 3, "l2");

    // 大 N：看真实带宽（LeetGPU 计时点太小，被 launch 淹没）
    test(1 << 24, "16M", 2.0 * (1 << 24) * sizeof(float));
    test(100000000, "1e8", 2.0 * 100000000 * sizeof(float));
}
