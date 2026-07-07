/*
 * softmax_bench.cu — Softmax 性能与正确性基准
 *
 * 对比 Naive 3-pass vs Online softmax，包含:
 *   - 冷缓存场景 (L2 eviction)
 *   - 正确性验证 (vs CPU reference)
 *   - 多组 (rows, cols) 配置
 */

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#include "../reduce.cuh"
#include "../softmax_naive.cuh"
#include "../softmax_online.cuh"

using namespace cuda_softmax;

#define CHK(c) do { if (auto e = (c)) { printf("CUDA Err: %d\n", e); exit(1); } } while (0)

// ---- L2 cache 驱逐内核 ------------------------------------------------------
__global__ void evict_kernel(int* buf) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < 16777216; i += blockDim.x * gridDim.x)
        buf[i] = 0;
}

// ---- CPU 参考实现 ----------------------------------------------------------
void softmax_cpu(const float* x, float* y, int64_t rows, int64_t cols) {
    for (int64_t r = 0; r < rows; ++r) {
        float mx = -INFINITY;
        for (int64_t c = 0; c < cols; ++c)
            mx = fmaxf(mx, x[r * cols + c]);
        float sm = 0;
        for (int64_t c = 0; c < cols; ++c) {
            y[r * cols + c] = expf(x[r * cols + c] - mx);
            sm += y[r * cols + c];
        }
        for (int64_t c = 0; c < cols; ++c)
            y[r * cols + c] /= sm;
    }
}

float max_relative_error(const float* a, const float* b, int64_t N) {
    float max_err = 0;
    for (int64_t i = 0; i < N; ++i) {
        float denom = fmaxf(fabsf(b[i]), 1e-9f);
        float err = fabsf(a[i] - b[i]) / denom;
        if (err > max_err) max_err = err;
    }
    return max_err;
}

// ---- Benchmark helper -------------------------------------------------------
template<typename Fn>
float benchmark(const char* name, Fn&& fn, int warmup, int iters,
                cudaStream_t stream, int* dev_evict) {
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        evict_kernel<<<1024, 1024>>>(dev_evict);
        fn();
    }
    CHK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CHK(cudaEventCreate(&start));
    CHK(cudaEventCreate(&stop));

    float total_ms = 0;
    for (int i = 0; i < iters; ++i) {
        evict_kernel<<<1024, 1024>>>(dev_evict);
        CHK(cudaStreamSynchronize(stream));

        CHK(cudaEventRecord(start, stream));
        fn();
        CHK(cudaEventRecord(stop, stream));
        CHK(cudaEventSynchronize(stop));

        float ms;
        CHK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }

    float avg_ms = total_ms / iters;
    printf("  %-36s  %9.4f ms  (%8.4f ms/iter)\n", name, total_ms, avg_ms);

    CHK(cudaEventDestroy(start));
    CHK(cudaEventDestroy(stop));
    return avg_ms;
}

// ---- Main ------------------------------------------------------------------
int main() {
    cudaDeviceProp prop;
    CHK(cudaGetDeviceProperties(&prop, 0));

    printf("================================================================\n");
    printf("  CUDA Softmax Benchmark — Naive vs Online\n");
    printf("  GPU: %s  |  SMs: %d\n", prop.name, prop.multiProcessorCount);
    printf("================================================================\n\n");

    // 64 MB L2 驱逐区
    int* dev_evict;
    CHK(cudaMalloc(&dev_evict, 64 << 20));

    // 测试配置: {rows, cols}
    struct { int rows, cols, warmup, iters; } tests[] = {
        // 小 cols → warp path
        {256,    256,   5, 200},
        {64,     512,   5, 200},
        {128,    1024,  5, 200},
        // 中等 cols → block SMem / uncached path
        {256,    4096,  5, 100},
        {512,    8192,  5, 100},
        // 大 cols (经典 LLM 场景)
        {1024,   8192,  3, 50},
        {1024,  32768,  3, 30},
        // 大 batch
        {4096,   1024,  3, 50},
    };

    cudaStream_t stream;
    CHK(cudaStreamCreate(&stream));

    for (auto t : tests) {
        int rows = t.rows, cols = t.cols;
        int64_t N = (int64_t)rows * cols;
        size_t bytes = N * sizeof(float);

        printf("--- rows=%-6d  cols=%-6d  (%.1f MB)  |  %d iters ---\n",
               rows, cols, bytes / (1024.0 * 1024.0), t.iters);

        // 生成数据
        std::vector<float> h_input(N), h_ref(N), h_output(N);
        {
            std::mt19937 rng(42);
            std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
            for (int64_t i = 0; i < N; ++i) h_input[i] = dist(rng);
        }
        softmax_cpu(h_input.data(), h_ref.data(), rows, cols);

        // 分配 GPU 内存
        float *d_input, *d_output;
        CHK(cudaMalloc(&d_input, bytes));
        CHK(cudaMalloc(&d_output, bytes));
        CHK(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

        // --- 正确性验证 ---
        printf("  [Correctness]\n");

        // Naive
        {
            CHK(cudaMemset(d_output, 0, bytes));
            launch_naive_softmax(d_input, d_output, rows, cols, stream);
            CHK(cudaStreamSynchronize(stream));
            CHK(cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
            printf("    %-34s  max rel err = %.6e\n",
                   "Naive 3-pass", max_relative_error(h_output.data(), h_ref.data(), N));
        }

        // Online CACHE_OPT=true
        {
            CHK(cudaMemset(d_output, 0, bytes));
            LaunchSoftmax<float, true>(stream, d_input, d_output, rows, cols);
            CHK(cudaStreamSynchronize(stream));
            CHK(cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
            printf("    %-34s  max rel err = %.6e\n",
                   "Online (CACHE_OPT=true)", max_relative_error(h_output.data(), h_ref.data(), N));
        }

        // Online CACHE_OPT=false
        {
            CHK(cudaMemset(d_output, 0, bytes));
            LaunchSoftmax<float, false>(stream, d_input, d_output, rows, cols);
            CHK(cudaStreamSynchronize(stream));
            CHK(cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
            printf("    %-34s  max rel err = %.6e\n",
                   "Online (CACHE_OPT=false)", max_relative_error(h_output.data(), h_ref.data(), N));
        }

        // --- 性能测试 ---
        printf("  [Performance]\n");

        benchmark("Naive 3-pass (baseline)",
            [&]() { launch_naive_softmax(d_input, d_output, rows, cols, stream); },
            t.warmup, t.iters, stream, dev_evict);

        benchmark("Online (CACHE_OPT=true)",
            [&]() { LaunchSoftmax<float, true>(stream, d_input, d_output, rows, cols); },
            t.warmup, t.iters, stream, dev_evict);

        benchmark("Online (CACHE_OPT=false)",
            [&]() { LaunchSoftmax<float, false>(stream, d_input, d_output, rows, cols); },
            t.warmup, t.iters, stream, dev_evict);

        CHK(cudaFree(d_input));
        CHK(cudaFree(d_output));
        printf("\n");
    }

    CHK(cudaStreamDestroy(stream));
    CHK(cudaFree(dev_evict));
    printf("Done.\n");
    return 0;
}
