/*
 * softmax_naive.cuh — Naive 3-pass Softmax
 *
 * 最简单的 softmax 实现，每行由单个线程串行处理：
 *   Pass 1: 遍历该行所有元素找 max
 *   Pass 2: 计算 exp(x - max) 并累加 sum
 *   Pass 3: 归一化写入 output
 *
 * 3 次 global memory read + 1 次 write，对 small cols 场景延迟尚可，
 * 但对大 cols 场景带宽利用率极低。作为 baseline 供对比。
 *
 * 仅支持 float，单 block 多线程（每线程处理一行）。
 */

#pragma once
#include <cuda_runtime.h>
#include <cfloat>

namespace cuda_softmax {

__global__ void naive_softmax_kernel(const float* X, float* P, int rows, int cols) {
    int row = threadIdx.x + blockDim.x * blockIdx.x;

    if (row < rows) {
        // Pass 1: find max
        float x_max = -INFINITY;
        for (int col = 0; col < cols; col++) {
            int i = row * cols + col;
            x_max = fmaxf(X[i], x_max);
        }

        // Pass 2: compute exp sum (denominator)
        float norm = 0.0f;
        for (int col = 0; col < cols; col++) {
            int i = row * cols + col;
            norm += expf(X[i] - x_max);
        }

        // Pass 3: normalize and write
        for (int col = 0; col < cols; col++) {
            int i = row * cols + col;
            P[i] = expf(X[i] - x_max) / norm;
        }
    }
}

inline void launch_naive_softmax(const float* d_input, float* d_output,
                                  int rows, int cols,
                                  cudaStream_t stream = 0) {
    int threads = 1024;
    int blocks = (rows + threads - 1) / threads;
    naive_softmax_kernel<<<blocks, threads, 0, stream>>>(d_input, d_output, rows, cols);
}

} // namespace cuda_softmax
