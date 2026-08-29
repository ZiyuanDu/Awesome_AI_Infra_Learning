#pragma once
#include <cuda_runtime.h>
#include <math.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))

__device__ __forceinline__ float warpReduceSum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

__device__ __forceinline__ float warpReduceMax(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    return v;
}

template <int BLOCK>
__device__ __forceinline__ float blockReduceSum(float v) {
    __shared__ float smem[BLOCK / 32];
    __shared__ float res;
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    v = warpReduceSum(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();

    if (wid == 0) {
        v = (lane < BLOCK / 32) ? smem[lane] : 0.f;
        v = warpReduceSum(v);
        if (lane == 0) res = v;
    }
    __syncthreads();
    return res;
}

template <int BLOCK>
__device__ __forceinline__ float blockReduceMax(float v) {
    __shared__ float smem[BLOCK / 32];
    __shared__ float res;
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    v = warpReduceMax(v);
    if (lane == 0) smem[wid] = v;
    __syncthreads();

    if (wid == 0) {
        v = (lane < BLOCK / 32) ? smem[lane] : -INFINITY;
        v = warpReduceMax(v);
        if (lane == 0) res = v;
    }
    __syncthreads();
    return res;
}


__device__ __forceinline__ float4 load4(const float* p, int ld, int r, int c, int nr, int nc) {
    int i = r * ld + c;
    // r 和 c 都没有超过nr和nc边界，并且i是4的倍数，(i & 3) 表示i的最低2位是0，说明是4的倍数
    if (r < nr && c + 3 < nc && (i & 3) == 0) {
        return reinterpret_cast<const float4*>(p + i);
    }

    // 否则，需要补零
    float x = (r < nr && c + 0 < nc) ? p[i + 0] : 0.f;
    float y = (r < nr && c + 1 < nc) ? p[i + 1] : 0.f;
    float z = (r < nr && c + 2 < nc) ? p[i + 2] : 0.f;
    float w = (r < nr && c + 3 < nc) ? p[i + 3] : 0.f;
    return make_float4(x, y, z, w);
}

__device__ __forceinline__ void store4(float* p, int ld, int r, int c, float4 val, int nr, int nc) {
    int i = r * ld + c;
    if (r < nr && c + 3 < nc && (i & 3) == 0) {
        *reinterpret_cast<float4*>(p + i) = val;
        return;
    }
    if (r < nr && c + 0 < nc)
        p[i + 0] = val.x;
    if (r < nr && c + 1 < nc)
        p[i + 1] = val.y;
    if (r < nr && c + 2 < nc)
        p[i + 2] = val.z;
    if (r < nr && c + 3 < nc)
        p[i + 3] = val.w;
}