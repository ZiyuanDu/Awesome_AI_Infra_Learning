/*
 * scan_bench.cu — Scan 性能与正确性基准
 *
 * 测试所有 scan 变体 (v0 naive, v1 blelloch, v2 multi-block, opt warp/block/device)
 * 包含冷缓存场景: 每次计时前驱逐 L2 cache，测量真实 cold-cache 带宽。
 * 同时与 CPU 参考实现对比验证正确性。
 */

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#include "../reduce.cuh"
#include "../scan_v0_naive.cuh"
#include "../scan_v1_blelloch.cuh"
#include "../scan_v2_multiblock.cuh"
#include "../scan_opt.cuh"

using namespace cuda_scan;

#define CHK(c) do { if (auto e = (c)) { printf("CUDA Err: %d\n", e); exit(1); } } while (0)

// ---- L2 cache 驱逐内核 ------------------------------------------------------
// 写满 ~64M int 使 L2 cache 行失效
__global__ void evict_kernel(int* buf) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < 16777216; i += blockDim.x * gridDim.x)
        buf[i] = 0;
}

// ---- CPU 参考实现 ----------------------------------------------------------
void cpu_scan_exclusive(const float* in, float* out, int N) {
    double acc = 0;
    for (int i = 0; i < N; i++) {
        out[i] = (float)acc;
        acc += (double)in[i];
    }
}

// ---- 正确性验证 ------------------------------------------------------------
bool validate_scan(const float* ref, const float* gpu, int N, double rtol = 1e-4) {
    for (int i = 0; i < N; i++) {
        double err = std::abs((double)ref[i] - (double)gpu[i]);
        if (err > std::abs((double)ref[i]) * rtol + 1e-15) return false;
    }
    return true;
}

// ---- 通用 Benchmark harness ------------------------------------------------
// 对每个 kernel 变体: warmup → 冷缓存计时 (多次取平均) → 验证正确性
template<typename Launch>
void bench_scan(
    const char* name, int N, int iters,
    const std::vector<float>& h_in,
    Launch&& launch,
    int* dev_evict)
{
    std::vector<float> ref(N);
    cpu_scan_exclusive(h_in.data(), ref.data(), N);

    float *d_in, *d_out;
    CHK(cudaMalloc(&d_in,  N * sizeof(float)));
    CHK(cudaMalloc(&d_out, N * sizeof(float)));
    CHK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Warmup (含 L2 evict)
    for (int i = 0; i < 3; ++i) {
        evict_kernel<<<1024, 1024>>>(dev_evict);
        launch(d_in, d_out, N);
    }
    CHK(cudaDeviceSynchronize());

    // 正式计时 (每次先 evict L2)
    cudaEvent_t start, stop;
    CHK(cudaEventCreate(&start));
    CHK(cudaEventCreate(&stop));

    float total_ms = 0;
    for (int i = 0; i < iters; ++i) {
        evict_kernel<<<1024, 1024>>>(dev_evict);
        CHK(cudaDeviceSynchronize());

        CHK(cudaEventRecord(start));
        launch(d_in, d_out, N);
        CHK(cudaEventRecord(stop));
        CHK(cudaDeviceSynchronize());

        float ms;
        CHK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }

    // 正确性校验
    std::vector<float> out(N);
    CHK(cudaMemcpy(out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    bool pass = validate_scan(ref.data(), out.data(), N);

    double avg_ms = total_ms / iters;
    // 带宽: 读 in + 写 out = 2 * N * sizeof(float) bytes
    double bw = (2.0 * N * sizeof(float)) / (avg_ms * 1e6);

    printf("  %-12s  %8.4f ms  |  %7.2f GB/s  |  %s\n",
           name, avg_ms, bw, pass ? "PASS" : "FAIL");

    CHK(cudaEventDestroy(start));
    CHK(cudaEventDestroy(stop));
    CHK(cudaFree(d_in));
    CHK(cudaFree(d_out));
}

// ---- Main ------------------------------------------------------------------
int main() {
    cudaDeviceProp prop;
    CHK(cudaGetDeviceProperties(&prop, 0));

    printf("================================================================\n");
    printf("  CUDA Scan — From Naive to Optimized\n");
    printf("  GPU: %s  |  SMs: %d\n", prop.name, prop.multiProcessorCount);
    printf("================================================================\n\n");

    // 64 MB L2 驱逐区
    int* dev_evict;
    CHK(cudaMalloc(&dev_evict, 64 << 20));

    // 测试配置: {N, iterations}
    struct { int N, iters; } tests[] = {
        {32,       100},   // 小规模: 多迭代测延迟
        {64,       100},
        {128,      100},
        {256,      100},
        {512,      100},
        {1024,     100},
        {1 << 20,  50},    // 1M  → 4 MB
        {16 << 20, 20},    // 16M → 64 MB
        {32 << 20, 10},    // 32M → 128 MB (大数组)
    };

    for (auto t : tests) {
        int N = t.N, iters = t.iters;
        int bytes = N * (int)sizeof(float);

        char sizestr[16];
        if (bytes < 1024)       snprintf(sizestr, 16, "%4d B",  bytes);
        else if (bytes < 1 << 20) snprintf(sizestr, 16, "%4d KB", bytes >> 10);
        else                    snprintf(sizestr, 16, "%4d MB", bytes >> 20);

        printf("--- N = %-8d  (%s)  |  %d iters ---\n", N, sizestr, iters);

        // 生成随机输入
        std::vector<float> h_in(N);
        {
            std::mt19937 rng(42);
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            for (int i = 0; i < N; i++) h_in[i] = dist(rng);
        }

        // Level 0–2: 单 block kernel（仅 N ≤ 1024 时可用）
        if (N <= 1024) {
            bench_scan("v0-naive",    N, iters, h_in, launch_v0,         dev_evict);
            bench_scan("v1-blelloch", N, iters, h_in, launch_v1,         dev_evict);
            if (N <= 32)
                bench_scan("opt-warp",   N, iters, h_in, launch_opt_warp,  dev_evict);
            bench_scan("opt-block",   N, iters, h_in, launch_opt_block,  dev_evict);
        }

        // Level 3: 多 block kernel（任意 N）
        bench_scan("v2-multi",    N, iters, h_in, launch_v2,          dev_evict);
        bench_scan("opt-device",  N, iters, h_in, launch_opt_device,  dev_evict);

        CHK(cudaDeviceSynchronize());
        printf("\n");
    }

    CHK(cudaFree(dev_evict));
    printf("Done.\n");
    return 0;
}
