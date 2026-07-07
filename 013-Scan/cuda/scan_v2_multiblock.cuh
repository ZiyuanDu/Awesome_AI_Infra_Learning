/*
 * scan_v2_multiblock.cuh — Multi-block Recursive Blelloch Scan
 *
 * 突破单 block 的 2048 元素限制，支持任意 N:
 *   1. 将数据分成多个 tile，每个 tile 做段内 scan（scan_v2_tile）
 *   2. 收集每个 tile 的总和到 aggs[]
 *   3. 递归地对 aggs[] 本身做 scan → 得到每个 tile 的全局前缀偏移
 *   4. 将前缀偏移加回各个 tile（scan_v2_add）
 *
 * 这是经典的三层递归分解: tile → recursive → add-back
 */

#pragma once
#include <cuda_runtime.h>
#include "reduce.cuh"
#include "scan_v1_blelloch.cuh"

namespace cuda_scan {

constexpr int V2_BLK     = 1024;                      // tile 大小
constexpr int V2_THR     = V2_BLK / 2;                // 每 tile 线程数（每线程 2 元素）
constexpr int PAD_V2_BLK = V2_BLK + (V2_BLK >> LOG_NUM_BANKS);

__global__ void scan_v2_tile(const float* in, float* out, float* aggs, int N) {
    // 1. 加载 2 个元素到共享内存 (带 padding)
    // 2. Upsweep：树状归约，算出段总和
    // 3. 把总和存到 aggs[blockIdx.x]，并把根节点清零
    // 4. Downsweep：向下传播，得段内 exclusive scan
    // 5. 写回 out
    __shared__ float tmp[PAD_V2_BLK];

    int tid = threadIdx.x;

    // 每线程负责的两个本地索引
    int li = tid;
    int ri = tid + V2_BLK / 2;

    // 对应的全局索引
    int gi = blockIdx.x * V2_BLK + li;
    int gj = blockIdx.x * V2_BLK + ri;

    tmp[pad(li)] = (gi < N) ? in[gi] : 0.0f;
    tmp[pad(ri)] = (gj < N) ? in[gj] : 0.0f;

    int offset = 1;
    // Upsweep
    for (int d = V2_BLK >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int left  = offset * (2 * tid + 1) - 1;
            int right = offset * (2 * tid + 2) - 1;
            tmp[pad(right)] += tmp[pad(left)];
        }
        offset <<= 1;
    }

    __syncthreads();
    if (tid == 0) {
        aggs[blockIdx.x] = tmp[pad(V2_BLK - 1)];   // 存下 tile 总和
        tmp[pad(V2_BLK - 1)] = 0.0f;                // 根节点清零
    }

    // Downsweep
    for (int d = 1; d < V2_BLK; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int left  = offset * (2 * tid + 1) - 1;
            int right = offset * (2 * tid + 2) - 1;

            float t = tmp[pad(left)];
            tmp[pad(left)]  = tmp[pad(right)];
            tmp[pad(right)] += t;
        }
    }
    __syncthreads();

    if (gi < N) out[gi] = tmp[pad(li)];
    if (gj < N) out[gj] = tmp[pad(ri)];
}

__global__ void scan_v2_add(const float* aggs, float* out, int N) {
    // 把 aggs[blockIdx.x] 加到 out 数组上，段内所有元素都加同一个值。
    int tid = threadIdx.x;
    int gi = blockIdx.x * V2_BLK + tid;
    int gj = gi + V2_BLK / 2;
    float add = aggs[blockIdx.x];          // 该 tile 需要加上的前缀

    if (gi < N) out[gi] += add;
    if (gj < N) out[gj] += add;
}

inline void launch_v2_prefix(float* data, int n) {
    // 终止条件：问题已缩小到一个 tile 能容纳
    if (n <= V2_BLK) {
        float* dummy;
        cudaMalloc(&dummy, sizeof(float)); // 不需要 aggs，但 kernel 需要这个参数，给个 dummy
        scan_v2_tile<<<1, V2_THR>>>(data, data, dummy, n);
        cudaFree(dummy);
        return;
    }

    // 1. 分块：计算需要多少个完整 tile
    int tiles = (n + V2_BLK - 1) / V2_BLK;
    float* totals;
    cudaMalloc(&totals, tiles * sizeof(float));

    // 2. 对每个 tile 做段内扫描，同时收集每个 tile 的总和到 totals
    scan_v2_tile<<<tiles, V2_THR>>>(data, data, totals, n);
    // 3. 递归！对 totals 本身求前缀和
    launch_v2_prefix(totals, tiles);
    // 4. 将 totals 中的前缀广播回各个 tile
    scan_v2_add<<<tiles, V2_THR>>>(totals, data, n);

    cudaFree(totals);
}

inline void launch_v2(const float* d_in, float* d_out, int n) {
    // 先拷贝输入到输出数组，之后就地计算
    cudaMemcpy(d_out, d_in, n * sizeof(float), cudaMemcpyDeviceToDevice);
    launch_v2_prefix(d_out, n);
}

} // namespace cuda_scan
