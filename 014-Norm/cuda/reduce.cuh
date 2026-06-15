#pragma once
#include <cuda_runtime.h>

#define WARP_SIZE 32

namespace cuda_norm {

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
#pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <const int NUM_THREADS, typename T>
__device__ __forceinline__ T block_reduce_sum(T val) {
    static_assert(NUM_THREADS % WARP_SIZE == 0, "NUM_THREADS 必须是 32 的倍数");
    static_assert(NUM_THREADS <= 1024, "NUM_THREADS 不能超过 1024");

    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    __shared__ T s_vals[NUM_WARPS];

    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    // 第一步: warp 内规约
    val = warp_reduce_sum(val);
    // 每个 warp 的 lane0 将结果写入 shared memory
    if (lane == 0) s_vals[warp] = val;
    __syncthreads();

    // 第二步: warp0 从 shared memory 读取各 warp 结果，再做一次 warp 规约
    val = (lane < NUM_WARPS) ? s_vals[lane] : T(0);
    if (warp == 0) val = warp_reduce_sum(val);
    return val;
}

} // namespace cuda_norm
