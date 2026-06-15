/*
 * io_bench.cu — 内存带宽基准测试
 *
 * 对比标量 (VEC=1) 和向量化 (VEC=2/4/8) copy kernel 的有效带宽，
 * 验证 Pack<T,N> 128-bit 访存的实际收益。
 * 纯 memcpy 操作，带宽即上限。
 */

#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../io.cuh"

using namespace cuda_norm;

#define CHK(c) do { if (auto e = (c)) { printf("CUDA Err: %d\n", e); exit(1); } } while (0)

template <typename T, int VEC_SIZE>
__global__ void vector_copy_kernel(const T* __restrict__ src, T* __restrict__ dst, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x * VEC_SIZE;

    for (int base = idx * VEC_SIZE; base < N; base += stride) {
        T tmp[VEC_SIZE];
        Pack<T, VEC_SIZE> pack;

        // 向量化加载 (对齐路径触发 LDG.128)
        if (base + VEC_SIZE <= N) {
            pack = *reinterpret_cast<const Pack<T, VEC_SIZE>*>(src + base);
#pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) tmp[i] = pack.elem[i];
        } else {
#pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i)
                tmp[i] = (base + i < N) ? src[base + i] : T(0);
        }

        // 向量化存储 (对齐路径触发 STG.128)
        if (base + VEC_SIZE <= N) {
#pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) pack.elem[i] = tmp[i];
            *reinterpret_cast<Pack<T, VEC_SIZE>*>(dst + base) = pack;
        } else {
#pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i)
                if (base + i < N) dst[base + i] = tmp[i];
        }
    }
}

template <typename T>
__global__ void scalar_copy_kernel(const T* __restrict__ src, T* __restrict__ dst, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) dst[i] = src[i];
}

template <typename T, int VEC_SIZE>
void run(const char* label, int N) {
    size_t bytes = N * sizeof(T);
    T *d_src, *d_dst;
    CHK(cudaMalloc(&d_src, bytes));
    CHK(cudaMalloc(&d_dst, bytes));

    int threads = 256;
    int blocks  = 4096;
    int iters   = 100;

    // 预热
    vector_copy_kernel<T, VEC_SIZE><<<blocks, threads>>>(d_src, d_dst, N);
    CHK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHK(cudaEventCreate(&start));
    CHK(cudaEventCreate(&stop));

    CHK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
        vector_copy_kernel<T, VEC_SIZE><<<blocks, threads>>>(d_src, d_dst, N);
    CHK(cudaEventRecord(stop));
    CHK(cudaEventSynchronize(stop));

    float ms;
    CHK(cudaEventElapsedTime(&ms, start, stop));
    float bw = (2.0 * bytes * iters) / (ms * 1e6);  // GB/s

    printf("%-32s | %7.3f ms | %8.1f GB/s\n", label, ms / iters, bw);

    CHK(cudaEventDestroy(start));
    CHK(cudaEventDestroy(stop));
    CHK(cudaFree(d_src));
    CHK(cudaFree(d_dst));
}

int main() {
    int N = 100 * 1024 * 1024 / sizeof(float);  // 100M floats ≈ 400 MB
    printf("--- IO 带宽基准测试  (%d M floats, %.0f MB) ---\n",
           N / (1024 * 1024), N * sizeof(float) / (1024.0 * 1024.0));
    printf("%-32s | %-8s | %s\n", "kernel", "avg_time", "bandwidth");
    printf("--------------------------------------------------------------\n");

    // 标量 copy 手动跑一次
    float *d_src, *d_dst;
    CHK(cudaMalloc(&d_src, N * sizeof(float)));
    CHK(cudaMalloc(&d_dst, N * sizeof(float)));

    cudaEvent_t start, stop;
    CHK(cudaEventCreate(&start));
    CHK(cudaEventCreate(&stop));

    scalar_copy_kernel<float><<<4096, 256>>>(d_src, d_dst, N);
    CHK(cudaDeviceSynchronize());

    int iters = 100;
    CHK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
        scalar_copy_kernel<float><<<4096, 256>>>(d_src, d_dst, N);
    CHK(cudaEventRecord(stop));
    CHK(cudaEventSynchronize(stop));

    float ms;
    CHK(cudaEventElapsedTime(&ms, start, stop));
    printf("%-32s | %7.3f ms | %8.1f GB/s\n",
           "scalar_copy<float>", ms / iters,
           (2.0 * N * sizeof(float) * iters) / (ms * 1e6));

    CHK(cudaEventDestroy(start));
    CHK(cudaEventDestroy(stop));
    CHK(cudaFree(d_src));
    CHK(cudaFree(d_dst));

    run<float, 1>("vector_copy<float, VEC=1>", N);
    run<float, 2>("vector_copy<float, VEC=2>", N);
    run<float, 4>("vector_copy<float, VEC=4>", N);

    int N_half = N;
    run<__half, 4>("vector_copy<half,  VEC=4>", N_half);
    run<__half, 8>("vector_copy<half,  VEC=8>", N_half);

    return 0;
}
