/*
 * reduce_bench.cu — warp/block 规约原语微基准
 *
 * 在循环内反复调用 block_reduce_sum，分摊 launch 开销，
 * 测量纯规约操作的延迟。
 */

#include <cstdio>
#include <cuda_runtime.h>
#include "../reduce.cuh"

using namespace cuda_norm;

#define CHK(c) do { if (auto e = (c)) { printf("CUDA Err: %d\n", e); exit(1); } } while (0)

template <int BLOCK_SIZE, typename T, int ITERS>
__global__ void reduce_bench_kernel(T* out) {
    T val = static_cast<T>(threadIdx.x);

#pragma unroll 1
    for (int i = 0; i < ITERS; ++i)
        val = block_reduce_sum<BLOCK_SIZE>(val);

    if (threadIdx.x == 0) out[blockIdx.x] = val;
}

template <int BLOCK_SIZE, typename T>
void run_bench(const char* label, int num_blocks) {
    constexpr int ITERS = 10000;

    T* d_out;
    CHK(cudaMalloc(&d_out, num_blocks * sizeof(T)));

    // 预热
    reduce_bench_kernel<BLOCK_SIZE, T, ITERS><<<num_blocks, BLOCK_SIZE>>>(d_out);
    CHK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHK(cudaEventCreate(&start));
    CHK(cudaEventCreate(&stop));

    float total_ms = 0;
    for (int i = 0; i < 20; ++i) {
        CHK(cudaEventRecord(start));
        reduce_bench_kernel<BLOCK_SIZE, T, ITERS><<<num_blocks, BLOCK_SIZE>>>(d_out);
        CHK(cudaEventRecord(stop));
        CHK(cudaEventSynchronize(stop));

        float ms;
        CHK(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }

    float avg_ms = total_ms / 20.0f;
    printf("%-30s | block=%3d | %5d iters | %7.3f ms\n",
           label, BLOCK_SIZE, ITERS, avg_ms);

    CHK(cudaEventDestroy(start));
    CHK(cudaEventDestroy(stop));
    CHK(cudaFree(d_out));
}

int main() {
    printf("--- Reduce 微基准测试 ---\n");
    printf("%-30s | %-8s | %-10s | %s\n", "kernel", "block", "iters", "avg_time");
    printf("---------------------------------------------------------------\n");

    run_bench<128, float>("block_reduce_sum<float>",  2048);
    run_bench<256, float>("block_reduce_sum<float>",  1024);
    run_bench<512, float>("block_reduce_sum<float>",   512);
    run_bench<1024, float>("block_reduce_sum<float>",  256);

    return 0;
}
