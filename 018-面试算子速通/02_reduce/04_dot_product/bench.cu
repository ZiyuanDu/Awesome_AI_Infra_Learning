#include "bench.cuh"

void solve(const float* A, const float* B, float* output, int N);

#include <cmath>
#include <cstdio>
#include <vector>

static float dotCpu(const float* A, const float* B, int N) {
    double s = 0.0;
    for (int i = 0; i < N; ++i)
        s += (double)A[i] * (double)B[i];
    return (float)s;
}

static bool closeRel(float got, float ref) {
    float den = fmaxf(1.f, fabsf(ref));
    return fabsf(got - ref) / den <= 2e-4f || fabsf(got - ref) <= 1e-3f;
}

static void check(int N, const char* name) {
    std::vector<float> A(N), B(N);
    for (int i = 0; i < N; ++i) {
        A[i] = (float)((i % 13) - 6);
        B[i] = (float)((i % 7) - 3);
    }
    float ref = dotCpu(A.data(), B.data(), N);

    float *dA = nullptr, *dB = nullptr, *dOut = nullptr;
    cudaMalloc(&dA, N * sizeof(float));
    cudaMalloc(&dB, N * sizeof(float));
    cudaMalloc(&dOut, sizeof(float));
    cudaMemcpy(dA, A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    solve(dA, dB, dOut, N);
    float got = 0.f;
    cudaMemcpy(&got, dOut, sizeof(float), cudaMemcpyDeviceToHost);
    if (!closeRel(got, ref)) {
        printf("FAIL %s  %g vs %g\n", name, got, ref);
        std::exit(1);
    }
    printf("OK  %s\n", name);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dOut);
}

int main() {
    // [1,2,3,4]·[5,6,7,8] = 70
    {
        float A[] = {1.f, 2.f, 3.f, 4.f};
        float B[] = {5.f, 6.f, 7.f, 8.f};
        float *dA, *dB, *dOut;
        cudaMalloc(&dA, 16);
        cudaMalloc(&dB, 16);
        cudaMalloc(&dOut, 4);
        cudaMemcpy(dA, A, 16, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B, 16, cudaMemcpyHostToDevice);
        solve(dA, dB, dOut, 4);
        float got = 0.f;
        cudaMemcpy(&got, dOut, 4, cudaMemcpyDeviceToHost);
        if (!closeRel(got, 70.f)) {
            printf("FAIL ex1  %g vs 70\n", got);
            return 1;
        }
        printf("OK  ex1\n");
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dOut);
    }

    check(1, "n1");
    check(3, "tail");
    check(10007, "odd");

    const int N = 1 << 22;  // 同 Reduction 计时量级
    std::vector<float> A(N), B(N);
    for (int i = 0; i < N; ++i) {
        A[i] = (float)((i % 13) - 6);
        B[i] = (float)((i % 7) - 3);
    }
    float *dA = nullptr, *dB = nullptr, *dOut = nullptr;
    cudaMalloc(&dA, N * sizeof(float));
    cudaMalloc(&dB, N * sizeof(float));
    cudaMalloc(&dOut, sizeof(float));
    cudaMemcpy(dA, A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    float ref = dotCpu(A.data(), B.data(), N);
    solve(dA, dB, dOut, N);
    float got = 0.f;
    cudaMemcpy(&got, dOut, sizeof(float), cudaMemcpyDeviceToHost);
    if (!closeRel(got, ref)) {
        printf("FAIL leetgpu  %g vs %g\n", got, ref);
        return 1;
    }
    printf("OK  leetgpu\n");

    float ms = timeMs([&] { solve(dA, dB, dOut, N); }, 10, 30);
    // 读 A+B
    printf("    %.3f ms  %.0f GB/s\n", ms, 2.0 * N * sizeof(float) / (ms * 1e6));
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dOut);
}
