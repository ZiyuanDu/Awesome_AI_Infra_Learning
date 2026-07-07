/*
 * reduce.cuh — Scan 公共原语
 *
 * 提供 warp 级 scan（Hillis-Steele 风格的 inclusive/exclusive）
 * 以及共享内存 bank conflict 规避用的 padding 工具。
 *
 * 所有 scan kernel 共享这些基础组件，与 014-Norm 的 reduce.cuh 保持一致的命名风格。
 */

#pragma once
#include <cuda_runtime.h>

namespace cuda_scan {

// ---- Bank-conflict padding ------------------------------------------------
// 32 个 bank，每 bank 4 字节。不加 padding 时 stride-32 访问会全部落到同一 bank。
// pad(i) 在索引 i 上插入一个偏移，使得逻辑上相邻的元素在物理 shared memory 中
// 落在不同的 bank，从而消除 bank conflict。
constexpr int NUM_BANKS     = 32;
constexpr int LOG_NUM_BANKS = 5;

__device__ __forceinline__ int pad(int i) {
    return i + (i >> LOG_NUM_BANKS);
}

// ---- Warp-level inclusive scan (Hillis-Steele) ----------------------------
// 使用 __shfl_up_sync 在 warp 内完成 O(log WARP_SIZE) 步的树状累加。
// 返回值 val 是当前 lane 及之前所有 lane 的元素之和（inclusive）。
__device__ __forceinline__ float warp_inclusive_scan(float val) {
    int lane = threadIdx.x & 31;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        float tmp = __shfl_up_sync(0xffffffff, val, off);
        if (lane >= off) val += tmp;
    }
    return val;
}

// ---- Warp-level exclusive scan --------------------------------------------
// exclusive scan = inclusive scan 右移一位，lane 0 填 0。
__device__ __forceinline__ float warp_exclusive_scan(float val) {
    float incl = warp_inclusive_scan(val);
    float excl = __shfl_up_sync(0xffffffff, incl, 1);
    return (threadIdx.x & 31) ? excl : 0.0f;
}

// ---- Warp-level inclusive scan (generic) ----------------------------------
// 与上相同，但保持原有名字以便兼容现有代码。
__device__ __forceinline__ float opt_warp_incl(float val) {
    return warp_inclusive_scan(val);
}

__device__ __forceinline__ float opt_warp_excl(float val) {
    return warp_exclusive_scan(val);
}

} // namespace cuda_scan
