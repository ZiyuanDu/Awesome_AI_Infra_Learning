/*
 * scan_opt.cuh — Optimized Multi-level Scan
 *
 * 利用 GPU 的层级并行性，从细粒度到粗粒度逐步优化:
 *
 *   Level 0 — Warp 级:     使用 __shfl_up_sync 完成 warp 内 scan（纯寄存器）
 *   Level 1 — 单 Warp:     N ≤ 32，无 shared memory
 *   Level 2 — 单 Block:    1024 线程，warp 间通过 shared memory 传递部分和
 *   Level 3 — Device 级:   每个线程处理 8 个元素（向量化），tile 递归分解
 *
 * Level 3 是目前性能最高的版本:
 *   - 寄存器内并行（8 items/thread）
 *   - warp shuffle 通信
 *   - shared memory warp 间归约
 *   - 递归多 block 分解
 */

#pragma once
#include <cuda_runtime.h>
#include "reduce.cuh"

namespace cuda_scan {

constexpr int OPT_BLOCK = 1024;
constexpr int OPT_WARPS = OPT_BLOCK / 32;
constexpr int OPT_ITEMS = 8;
constexpr int OPT_TILE  = OPT_BLOCK * OPT_ITEMS;

// Level 1: single warp (N ≤ 32)
__global__ void scan_opt_warp(const float* in, float* out, int N) {
    int tid = threadIdx.x;
    if (tid < N) out[tid] = warp_exclusive_scan((tid < N) ? in[tid] : 0.0f);
}

inline void launch_opt_warp(const float* in, float* out, int n) {
    scan_opt_warp<<<1, 32>>>(in, out, n);
}

// Level 2: single block
__global__ void scan_opt_block(const float* in, float* out, int N) {
    __shared__ float agg[OPT_WARPS];
    int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    float val = (tid < N) ? in[tid] : 0.0f;

    val = warp_inclusive_scan(val);
    if (lane == 31) agg[wid] = val;
    __syncthreads();

    if (tid == 0) {
        float acc = 0;
#pragma unroll
        for (int w = 0; w < OPT_WARPS; w++) {
            float s = agg[w];
            agg[w] = acc;
            acc += s;
        }
    }
    __syncthreads();

    float excl = __shfl_up_sync(0xffffffff, val, 1);
    if (lane == 0) excl = 0.0f;
    if (tid < N) out[tid] = excl + agg[wid];
}

inline void launch_opt_block(const float* in, float* out, int n) {
    scan_opt_block<<<1, OPT_BLOCK>>>(in, out, n);
}

// Level 3: tile scan — 每线程处理 OPT_ITEMS 个元素
__global__ void scan_opt_tile(const float* in, float* out, float* aggs, int N) {
    __shared__ float s_agg[OPT_WARPS];
    int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    int base = blockIdx.x * OPT_TILE + tid * OPT_ITEMS;

    float items[OPT_ITEMS], sum = 0.0f;
#pragma unroll
    for (int i = 0; i < OPT_ITEMS; i++) {
        int idx = base + i;
        float v = (idx < N) ? in[idx] : 0.0f;
        sum += v;
        items[i] = sum;
    }

    float val = warp_inclusive_scan(sum);
    if (lane == 31) s_agg[wid] = val;
    __syncthreads();

    if (tid == 0) {
        float acc = 0;
#pragma unroll
        for (int w = 0; w < OPT_WARPS; w++) {
            float s = s_agg[w];
            s_agg[w] = acc;
            acc += s;
        }
        aggs[blockIdx.x] = acc;
    }
    __syncthreads();

    float prefix = s_agg[wid];
    float excl   = __shfl_up_sync(0xffffffff, val, 1);
    if (lane == 0) excl = 0.0f;
    prefix += excl;

#pragma unroll
    for (int i = 0; i < OPT_ITEMS; i++) {
        int idx = base + i;
        if (idx < N) out[idx] = prefix + (i > 0 ? items[i - 1] : 0.0f);
    }
}

__global__ void scan_opt_apply(const float* aggs, float* out, int N) {
    int tid = threadIdx.x, base = blockIdx.x * OPT_TILE + tid * OPT_ITEMS;
    float add = aggs[blockIdx.x];
#pragma unroll
    for (int i = 0; i < OPT_ITEMS; i++) {
        int idx = base + i;
        if (idx < N) out[idx] += add;
    }
}

inline void launch_opt_prefix(float* data, int n) {
    if (n <= OPT_TILE) {
        float* dummy;
        cudaMalloc(&dummy, sizeof(float));
        scan_opt_tile<<<1, OPT_BLOCK>>>(data, data, dummy, n);
        cudaFree(dummy);
        return;
    }
    int nb = (n + OPT_TILE - 1) / OPT_TILE;
    float* totals;
    cudaMalloc(&totals, nb * sizeof(float));
    scan_opt_tile<<<nb, OPT_BLOCK>>>(data, data, totals, n);
    launch_opt_prefix(totals, nb);
    scan_opt_apply<<<nb, OPT_BLOCK>>>(totals, data, n);
    cudaFree(totals);
}

inline void launch_opt_device(const float* in, float* out, int n) {
    int tiles = (n + OPT_TILE - 1) / OPT_TILE;
    float* aggs;
    cudaMalloc(&aggs, tiles * sizeof(float));
    scan_opt_tile<<<tiles, OPT_BLOCK>>>(in, out, aggs, n);
    launch_opt_prefix(aggs, tiles);
    scan_opt_apply<<<tiles, OPT_BLOCK>>>(aggs, out, n);
    cudaFree(aggs);
}

} // namespace cuda_scan
