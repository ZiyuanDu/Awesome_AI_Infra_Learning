/*
 * scan_v0_naive.cuh — Naive Hillis-Steele Parallel Scan
 *
 * 最直观的并行 scan 实现：每个线程负责一个元素，log N 步迭代。
 * 每步中线程从 offset 距离处读取前驱元素并累加。
 * 工作复杂度 O(N log N)，单 block 内完成。
 *
 * 限制: 仅支持 N ≤ 1024（单 block 最大线程数）
 */

#pragma once
#include <cuda_runtime.h>

namespace cuda_scan {

__global__ void scan_v0_naive(const float* in, float* out, int n) {
    extern __shared__ float buf[];

    int tid = threadIdx.x;
    int pout = 0, pin = 1;

    buf[pout * n + tid] = (tid > 0) ? in[tid - 1] : 0.0f;
    __syncthreads();

    for (int offset = 1; offset < n; offset <<= 1) {
        pout = 1 - pout;
        pin  = 1 - pout;
        buf[pout * n + tid] = buf[pin * n + tid];

        if (tid >= offset) {
            buf[pout * n + tid] += buf[pin * n + tid - offset];
        }
        __syncthreads();
    }
    out[tid] = buf[pout * n + tid];
}

inline void launch_v0(const float* in, float* out, int n) {
    scan_v0_naive<<<1, n, 2 * n * sizeof(float)>>>(in, out, n);
}

} // namespace cuda_scan
