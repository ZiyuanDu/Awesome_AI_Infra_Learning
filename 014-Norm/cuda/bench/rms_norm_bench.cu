/*
 * rms_norm_bench.cu — RMSNorm 性能与正确性基准
 *
 * 冷缓存场景: 每次计时前驱逐 L2 cache，测量真实 cold-cache 带宽。
 * 同时与 CPU 参考实现对比验证正确性。
 */

#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../rms_norm.cuh"

using namespace cuda_norm;

#define CHK(c) do { if (auto e = (c)) { printf("CUDA Err: %d\n", e); exit(1); } } while (0)

// L2 cache 驱逐内核 — 写满 ~64M int 使 cache 行失效
__global__ void evict_kernel(int* buf) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < 16777216; i += blockDim.x * gridDim.x)
        buf[i] = 0;
}

// CPU 参考实现
void cpu_rms_norm(const float* x, float* y, const float* gamma,
                  int N, int K, float eps) {
    for (int n = 0; n < N; ++n) {
        float sq = 0;
        for (int k = 0; k < K; ++k) {
            float v = x[n * K + k];
            sq += v * v;
        }
        float inv_rms = 1.0f / sqrtf(sq / K + eps);
        for (int k = 0; k < K; ++k)
            y[n * K + k] = x[n * K + k] * inv_rms * gamma[k];
    }
}

template <typename T>
void bench(int N, int K, const char* label) {
    int tot = N * K;
    std::vector<float> fx(tot), fref(tot), fg(K);
    std::vector<T>    hx(tot), hy(tot), hg(K);

    for (int i = 0; i < tot; ++i) { fx[i] = sinf(float(i)); hx[i] = T(fx[i]); }
    for (int i = 0; i < K;   ++i) { fg[i] = cosf(float(i)); hg[i] = T(fg[i]); }

    cpu_rms_norm(fx.data(), fref.data(), fg.data(), N, K, 1e-5f);

    T *dx, *dy, *dg;
    int *dev;
    CHK(cudaMalloc(&dx,  tot * sizeof(T)));
    CHK(cudaMalloc(&dy,  tot * sizeof(T)));
    CHK(cudaMalloc(&dg,  K   * sizeof(T)));
    CHK(cudaMalloc(&dev, 64 << 20));  // 64 MB L2 驱逐区

    CHK(cudaMemcpy(dx, hx.data(), tot * sizeof(T), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dg, hg.data(), K   * sizeof(T), cudaMemcpyHostToDevice));

    const float eps = 1e-5f;

    // 预热 (5 次驱逐 + 执行)
    for (int i = 0; i < 5; ++i) {
        evict_kernel<<<1024, 1024>>>(dev);
        rms_norm_forward(dx, dy, dg, N, K, eps, sizeof(T) == 2);
    }
    CHK(cudaDeviceSynchronize());

    // 正式计时 (50 次，每次驱逐后单独计时)
    cudaEvent_t start, stop;
    CHK(cudaEventCreate(&start));
    CHK(cudaEventCreate(&stop));

    float total_ms = 0;
    for (int i = 0; i < 50; ++i) {
        evict_kernel<<<1024, 1024>>>(dev);
        CHK(cudaDeviceSynchronize());

        CHK(cudaEventRecord(start));
        rms_norm_forward(dx, dy, dg, N, K, eps, sizeof(T) == 2);
        CHK(cudaEventRecord(stop));
        CHK(cudaDeviceSynchronize());

        float ms;
        CHK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }

    // 正确性校验
    CHK(cudaMemcpy(hy.data(), dy, tot * sizeof(T), cudaMemcpyDeviceToHost));
    float max_err = 0;
    for (int i = 0; i < tot; ++i)
        max_err = std::max(max_err, std::abs(float(hy[i]) - fref[i]));

    // 带宽 = (读 x + 读 gamma + 写 y) × 50 次 / 总时间
    double bw = (2.0 * tot + K) * sizeof(T) * 50 / (total_ms * 1e6);
    bool pass = max_err < (sizeof(T) == 2 ? 5e-2f : 5e-4f);

    printf("%-20s | %4dx%-5d | %7.3f ms | %8.1f GB/s | err: %.1e [%s]\n",
           label, N, K, total_ms / 50, bw, max_err, pass ? "OK" : "FAIL");

    CHK(cudaEventDestroy(start));
    CHK(cudaEventDestroy(stop));
    CHK(cudaFree(dx)); CHK(cudaFree(dy)); CHK(cudaFree(dg)); CHK(cudaFree(dev));
}

int main() {
    printf("--- RMSNorm 性能测试 ---\n");
    printf("%-20s | %-9s | %-9s | %-10s | correctness\n", "kernel", "shape", "avg_ms", "bandwidth");
    printf("--------------------------------------------------------------------\n");

    int shapes[][2] = {
        {1024,   256},   // K=256  → WarpImpl
        {1024,  1024},   // K=1024 → WarpImpl
        {2048,  4096},   // K=4096 → BlockSMemImpl
        {4096,  8192},   // K=8192 → BlockSMemImpl
    };

    for (auto& s : shapes) {
        bench<float>(s[0], s[1], "RMSNorm FP32");
        bench<__half>(s[0], s[1], "RMSNorm FP16");
        printf("\n");
    }

    return 0;
}
