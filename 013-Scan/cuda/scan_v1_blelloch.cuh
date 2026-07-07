/*
 * scan_v1_blelloch.cuh — Single-block Blelloch Work-efficient Scan
 *
 * 实现经典的 Blelloch 两阶段 scan:
 *   Phase 1 (Upsweep):   树状归约，在根节点得到全局总和
 *   Phase 2 (Downsweep): 根清零后向下传播，生成 exclusive scan
 *
 * 每个线程处理 2 个元素以提高计算效率。
 * 使用 bank-conflict padding（来自 reduce.cuh）消除 shared memory 冲突。
 *
 * 工作复杂度 O(N)，但仅支持单 block（N ≤ 2048，受 shared memory 限制）。
 */

#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include "reduce.cuh"

namespace cuda_scan {

__global__ void scan_v1_blelloch(const float* in, float* out, int original_n, int n2) {
    extern __shared__ float temp[];

    int thid = threadIdx.x;
    int offset = 1;

    // 每个线程处理两个元素，提高线程本身的计算效率
    int ai = thid;
    int bi = thid + (n2 >> 1);

    temp[pad(ai)] = (ai < original_n) ? in[ai] : 0.0f;
    temp[pad(bi)] = (bi < original_n) ? in[bi] : 0.0f;

    // ==================== Phase 1: Upsweep ====================
    for (int d = n2 >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int left_idx  = offset * (2 * thid + 1) - 1;
            int right_idx = offset * (2 * thid + 2) - 1;

            temp[pad(right_idx)] += temp[pad(left_idx)];
        }
        offset <<= 1;
    }

    // ==================== Phase 2: Downsweep ====================
    if (thid == 0) {
        temp[pad(n2 - 1)] = 0.0f;
    }

    for (int d = 1; d < n2; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int left_idx  = offset * (2 * thid + 1) - 1;
            int right_idx = offset * (2 * thid + 2) - 1;

            float t         = temp[pad(left_idx)];
            temp[pad(left_idx)]  = temp[pad(right_idx)];
            temp[pad(right_idx)] += t;
        }
    }
    __syncthreads();

    if (ai < original_n) out[ai] = temp[pad(ai)];
    if (bi < original_n) out[bi] = temp[pad(bi)];
}

inline void launch_v1(const float* in, float* out, int n) {
    if (n <= 0) return;

    int n2 = 1;
    while (n2 < n) n2 <<= 1;

    if (n2 > 2048) {
        printf("Error: Single block scan only supports up to 2048 elements.\n");
        return;
    }

    int threads   = (n2 / 2 > 0) ? (n2 / 2) : 1;
    int padded_n  = n2 + (n2 >> LOG_NUM_BANKS);

    scan_v1_blelloch<<<1, threads, padded_n * sizeof(float)>>>(in, out, n, n2);
}

} // namespace cuda_scan
