#include "bench.cuh"

void solve(const float* A, const float* x, float* y, int M, int N, int nnz);

#include <cmath>
#include <cstdio>
#include <vector>

static void gemvCpu(const float* A, const float* x, float* y, int M, int N) {
    for (int i = 0; i < M; ++i) {
        double s = 0.0;
        const float* row = A + (size_t)i * N;
        for (int j = 0; j < N; ++j)
            s += (double)row[j] * (double)x[j];
        y[i] = (float)s;
    }
}

static bool close(float got, float ref) {
    float den = fmaxf(1.f, fabsf(ref));
    return fabsf(got - ref) / den <= 2e-3f || fabsf(got - ref) <= 1e-2f;
}

static int countNnz(const std::vector<float>& A) {
    int n = 0;
    for (float v : A)
        if (v != 0.f)
            ++n;
    return n;
}

static void fillSparse(std::vector<float>& A, int M, int N, float density, unsigned seed) {
    A.assign((size_t)M * N, 0.f);
    unsigned s = seed;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            s = s * 1664525u + 1013904223u;
            if ((s & 0xffff) / 65535.f < density) {
                s = s * 1664525u + 1013904223u;
                A[(size_t)i * N + j] = (float)((int)(s % 13) - 6);
            }
        }
    }
}

static void check(int M, int N, float density, const char* name) {
    std::vector<float> A, x(N), y(M), ref(M);
    fillSparse(A, M, N, density, 7u);
    for (int j = 0; j < N; ++j)
        x[j] = (float)((j % 11) - 5);
    gemvCpu(A.data(), x.data(), ref.data(), M, N);

    float *dA = nullptr, *dX = nullptr, *dY = nullptr;
    cudaMalloc(&dA, A.size() * sizeof(float));
    cudaMalloc(&dX, N * sizeof(float));
    cudaMalloc(&dY, M * sizeof(float));
    cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, x.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    const int nnz = countNnz(A);
    solve(dA, dX, dY, M, N, nnz);
    cudaMemcpy(y.data(), dY, M * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; ++i) {
        if (!close(y[i], ref[i])) {
            printf("FAIL %s  y[%d]=%g vs %g\n", name, i, y[i], ref[i]);
            std::exit(1);
        }
    }
    printf("OK  %s  (nnz=%d dens=%.2f)\n", name, nnz, (float)nnz / (float)(M * N));
    cudaFree(dA);
    cudaFree(dX);
    cudaFree(dY);
}

static void benchCase(int M, int N, float density, const char* name) {
    std::vector<float> A, x(N);
    fillSparse(A, M, N, density, 42u);
    for (int j = 0; j < N; ++j)
        x[j] = (float)((j % 11) - 5);

    float *dA = nullptr, *dX = nullptr, *dY = nullptr;
    cudaMalloc(&dA, A.size() * sizeof(float));
    cudaMalloc(&dX, N * sizeof(float));
    cudaMalloc(&dY, M * sizeof(float));
    cudaMemcpy(dA, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, x.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    const int nnz = countNnz(A);

    // correctness
    std::vector<float> ref(M), got(M);
    gemvCpu(A.data(), x.data(), ref.data(), M, N);
    solve(dA, dX, dY, M, N, nnz);
    cudaMemcpy(got.data(), dY, M * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < M; ++i) {
        if (!close(got[i], ref[i])) {
            printf("FAIL %s check y[%d]\n", name, i);
            std::exit(1);
        }
    }

    float ms = timeMs([&] { solve(dA, dX, dY, M, N, nnz); }, 50, 100);
    // 名义：整表 A + x + y；理想下 x 多在 L2
    const double bytesAll = ((double)M * N + N + M) * sizeof(float);
    const double bytesA = (double)M * N * sizeof(float);
    const double flops = 2.0 * M * N;  // 含乘零
    printf("%s  M=%d N=%d  %.3f ms  %.0f GB/s(A+x+y)  %.0f GB/s(A)  %.1f TFLOPS\n", name, M, N, ms,
           bytesAll / (ms * 1e6), bytesA / (ms * 1e6), flops / (ms * 1e9));

    cudaFree(dA);
    cudaFree(dX);
    cudaFree(dY);
}

int main() {
    // example 3x4
    {
        float A[] = {5, 0, 0, 1, 0, 2, 3, 0, 0, 0, 0, 4};
        float x[] = {1, 2, 3, 4};
        float y[3], ref[3];
        gemvCpu(A, x, ref, 3, 4);
        float *dA, *dX, *dY;
        cudaMalloc(&dA, sizeof(A));
        cudaMalloc(&dX, sizeof(x));
        cudaMalloc(&dY, sizeof(y));
        cudaMemcpy(dA, A, sizeof(A), cudaMemcpyHostToDevice);
        cudaMemcpy(dX, x, sizeof(x), cudaMemcpyHostToDevice);
        solve(dA, dX, dY, 3, 4, 5);
        cudaMemcpy(y, dY, sizeof(y), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 3; ++i) {
            if (!close(y[i], ref[i])) {
                printf("FAIL ex  %g vs %g\n", y[i], ref[i]);
                return 1;
            }
        }
        printf("OK  ex1\n");
        cudaFree(dA);
        cudaFree(dX);
        cudaFree(dY);
    }

    check(17, 31, 0.35f, "odd");
    check(128, 256, 0.35f, "mid");
    check(1000, 10000, 0.35f, "leetgpu-density");

    printf("\n--- perf ---\n");
    benchCase(1000, 10000, 0.35f, "leetgpu");
    benchCase(1000, 10000, 1.00f, "dense");   // 同布局，证明跳零无必要
    benchCase(4096, 8192, 0.35f, "large");

    // 对照：只搬 A 的 memcpy 带宽上限
    {
        const int M = 1000, N = 10000;
        const size_t bytes = (size_t)M * N * sizeof(float);
        float *dA, *dB;
        cudaMalloc(&dA, bytes);
        cudaMalloc(&dB, bytes);
        float ms = timeMs([&] { cudaMemcpy(dB, dA, bytes, cudaMemcpyDeviceToDevice); }, 10, 30);
        printf("memcpy A  %.3f ms  %.0f GB/s  (ceiling for reading A once)\n", ms,
               (double)bytes / (ms * 1e6));
        cudaFree(dA);
        cudaFree(dB);
    }
}
