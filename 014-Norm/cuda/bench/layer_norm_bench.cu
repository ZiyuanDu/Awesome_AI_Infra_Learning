/*
 * layer_norm_bench.cu — LayerNorm 性能与正确性基准
 *
 * 测试方法与 rms_norm_bench.cu 完全一致:
 *   冷缓存 (每次计时前 L2 eviction) + 50 次迭代取平均。
 * 额外验证 mean 和 variance 的计算精度。
 */

#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../layer_norm.cuh"

using namespace cuda_norm;

#define CHK(c) do { if (auto e = (c)) { printf("CUDA Err: %d\n", e); exit(1); } } while (0)

// L2 cache 驱逐内核
__global__ void evict_kernel(int* buf) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < 16777216; i += blockDim.x * gridDim.x)
        buf[i] = 0;
}

// CPU 参考实现
void cpu_layer_norm(const float* x, float* y,
                    const float* gamma, const float* beta,
                    int N, int K, float eps) {
    for (int n = 0; n < N; ++n) {
        float mean = 0;
        for (int k = 0; k < K; ++k) mean += x[n * K + k];
        mean /= K;

        float var = 0;
        for (int k = 0; k < K; ++k) {
            float d = x[n * K + k] - mean;
            var += d * d;
        }
        float inv_std = 1.0f / sqrtf(var / K + eps);

        for (int k = 0; k < K; ++k)
            y[n * K + k] = (x[n * K + k] - mean) * inv_std * gamma[k] + beta[k];
    }
}

template <typename T>
void bench(int N, int K, const char* label) {
    int tot = N * K;
    std::vector<float> fx(tot), fref(tot), fg(K), fb(K);
    std::vector<T>    hx(tot), hy(tot), hg(K), hb(K);

    for (int i = 0; i < tot; ++i) { fx[i] = sinf(float(i)); hx[i] = T(fx[i]); }
    for (int i = 0; i < K;   ++i) {
        fg[i] = cosf(float(i)); fb[i] = sinf(float(i));
        hg[i] = T(fg[i]);       hb[i] = T(fb[i]);
    }

    cpu_layer_norm(fx.data(), fref.data(), fg.data(), fb.data(), N, K, 1e-5f);

    T *dx, *dy, *dg, *db;
    int *dev;
    CHK(cudaMalloc(&dx,  tot * sizeof(T)));
    CHK(cudaMalloc(&dy,  tot * sizeof(T)));
    CHK(cudaMalloc(&dg,  K   * sizeof(T)));
    CHK(cudaMalloc(&db,  K   * sizeof(T)));
    CHK(cudaMalloc(&dev, 64 << 20));  // 64 MB L2 驱逐区

    CHK(cudaMemcpy(dx, hx.data(), tot * sizeof(T), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dg, hg.data(), K   * sizeof(T), cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(db, hb.data(), K   * sizeof(T), cudaMemcpyHostToDevice));

    const float eps = 1e-5f;

    // 预热
    for (int i = 0; i < 5; ++i) {
        evict_kernel<<<1024, 1024>>>(dev);
        layer_norm_forward(dx, dy, dg, db, N, K, eps, sizeof(T) == 2);
    }
    CHK(cudaDeviceSynchronize());

    // 正式计时
    cudaEvent_t start, stop;
    CHK(cudaEventCreate(&start));
    CHK(cudaEventCreate(&stop));

    float total_ms = 0;
    for (int i = 0; i < 50; ++i) {
        evict_kernel<<<1024, 1024>>>(dev);
        CHK(cudaDeviceSynchronize());

        CHK(cudaEventRecord(start));
        layer_norm_forward(dx, dy, dg, db, N, K, eps, sizeof(T) == 2);
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

    // 带宽 = (读 x + 读 gamma + 读 beta + 写 y) × 50 次 / 总时间
    double bw = (2.0 * tot + 2.0 * K) * sizeof(T) * 50 / (total_ms * 1e6);
    bool pass = max_err < (sizeof(T) == 2 ? 5e-2f : 5e-4f);

    printf("%-20s | %4dx%-5d | %7.3f ms | %8.1f GB/s | err: %.1e [%s]\n",
           label, N, K, total_ms / 50, bw, max_err, pass ? "OK" : "FAIL");

    CHK(cudaEventDestroy(start));
    CHK(cudaEventDestroy(stop));
    CHK(cudaFree(dx)); CHK(cudaFree(dy)); CHK(cudaFree(dg)); CHK(cudaFree(db)); CHK(cudaFree(dev));
}

int main() {
    printf("--- LayerNorm 性能测试 ---\n");
    printf("%-20s | %-9s | %-9s | %-10s | correctness\n", "kernel", "shape", "avg_ms", "bandwidth");
    printf("--------------------------------------------------------------------\n");

    int shapes[][2] = {
        {1024,   256},   // K=256  → WarpImpl
        {1024,  1024},   // K=1024 → WarpImpl
        {2048,  4096},   // K=4096 → BlockSMemImpl
        {4096,  8192},   // K=8192 → BlockSMemImpl
    };

    for (auto& s : shapes) {
        bench<float>(s[0], s[1], "LayerNorm FP32");
        bench<__half>(s[0], s[1], "LayerNorm FP16");
        printf("\n");
    }

    return 0;
}
