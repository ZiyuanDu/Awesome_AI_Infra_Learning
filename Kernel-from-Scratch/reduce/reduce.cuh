#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>

// All 32 lanes participate; every lane receives the result.
template <bool MAXIMUM>
__device__ __forceinline__ float warp_reduce(float value) {
    #pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, value, offset);
        value = MAXIMUM ? fmaxf(value, other) : value + other;
    }
    return value;
}

// Use a 1D block with 32, 64, ..., 1024 threads. All threads must call this.
template <bool MAXIMUM>
__device__ __forceinline__ float block_reduce(float value) {
    __shared__ float partial[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    value = warp_reduce<MAXIMUM>(value);
    if (blockDim.x == 32) return value;
    if (lane == 0) partial[warp] = value;
    __syncthreads();
    float identity = MAXIMUM ? -CUDART_INF_F : 0.0f;
    value = lane < blockDim.x / 32 ? partial[lane] : identity;
    value = warp_reduce<MAXIMUM>(value);
    __syncthreads();  // Safe to reuse partial in the next reduction.
    return value;
}

__device__ __forceinline__ float block_sum(float value) {
    return block_reduce<false>(value);
}

__device__ __forceinline__ float block_max(float value) {
    return block_reduce<true>(value);
}
