#pragma once
#include <cuda_runtime.h>
#include <math.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))


// =============================================================================
// Reduce
// =============================================================================
__device__ __forceinline__ float warpReduceSum(float v) {
#pragma unroll
    for (size_t offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, offset);
    return v;
}

__device__ __forceinline__ float warpReduceMax(float v) {
#pragma unroll
    for (size_t offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
    return v;
}

template <size_t BLOCK>
__device__ __forceinline__ float blockReduceSum(float v) {
    __shared__ float warp_sums[BLOCK / 32];
    const size_t lane = threadIdx.x & 31;
    const size_t warp = threadIdx.x >> 5;

    v = warpReduceSum(v);

    // 只让 每个 warp 的 lane0 写入
    if (lane == 0) {
        warp_sums[warp] = v;
    }
    __syncthreads();

    // 只让 warp0 读取
    if (warp == 0) {
        v = (lane < BLOCK / 32) ? warp_sums[lane] : 0.f;
        v = warpReduceSum(v);
    }
    return v;
}

template <int BLOCK>
__device__ __forceinline__ float blockReduceMax(float v) {
    __shared__ float warp_maxs[BLOCK / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    v = warpReduceMax(v);
    if (lane == 0)
        warp_maxs[warp] = v;
    __syncthreads();

    if (warp == 0) {
        v = (lane < BLOCK / 32) ? warp_maxs[lane] : -INFINITY;
        v = warpReduceMax(v);
    }
    return v;
}

// =============================================================================
// Scan
// =============================================================================

__device__ __forceinline__ float warpScanInclusive(float v) {
    const size_t lane = threadIdx.x & 31;

    #pragma unroll
    for (size_t offset = 1; offset < 32; offset <<= 1) {
        float other = __shfl_up_sync(0xffffffff, v, offset);
        // 只对比它大的数字做累加
        if (lane >= offset) {
            v += other;
        }
    }
    return v;
}

template <int BLOCK>
__device__ __forceinline__ float blockScanInclusive(float v) {
    __shared__ float arrays[BLOCK / 32];
    const size_t lane = threadIdx.x & 31;
    const size_t warp = threadIdx.x >> 5;

    v = warpScanInclusive(v);
    // 最后一个线程具有完整的累加结果
    if (lane == 31)
        arrays[warp] = v;
    __syncthreads();

    //
    if (warp == 0) {
        float w = (lane < BLOCK / 32) ? arrays[lane] : 0.f;
        w = warpScanInclusive(w);
        if (lane < BLOCK / 32)
            arrays[lane] = w;
    }
    __syncthreads();

    if (warp > 0)
        v += arrays[warp - 1];
    return v;
}

template <int BLOCK>
__device__ __forceinline__ float blockScanExclusive(float v) {
    return blockScanInclusive<BLOCK>(v) - v;
}

// =============================================================================
// Online Softmax
// =============================================================================
__device__ __forceinline__ void onlineMerge(float& m, float& l, float m2, float l2) {
    float nm = fmaxf(m, m2);
    float a = (m == nm) ? 1.f : __expf(m - nm);
    float b = (m2 == nm) ? 1.f : __expf(m2 - nm);
    l = l * a + l2 * b;
    m = nm;
}

__device__ __forceinline__ void onlineUpdate(float& m, float& l, float x) {
    onlineMerge(m, l, x, 1.f);
}

__device__ __forceinline__ void warpOnlineReduce(float& m, float& l) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        onlineMerge(m, l, __shfl_xor_sync(0xffffffff, m, offset),
                    __shfl_xor_sync(0xffffffff, l, offset));
}

template <int BLOCK>
__device__ __forceinline__ void blockOnlineReduce(float& m, float& l) {
    __shared__ float warp_m[BLOCK / 32], warp_l[BLOCK / 32];
    __shared__ float block_m, block_l;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    float tm = warpReduceMax(m);
    if (lane == 0)
        warp_m[warp] = tm;
    __syncthreads();
    if (warp == 0) {
        tm = (lane < BLOCK / 32) ? warp_m[lane] : -INFINITY;
        tm = warpReduceMax(tm);
        if (lane == 0)
            block_m = tm;
    }
    __syncthreads();
    const float m_g = block_m;

    float tl = l * ((m == m_g) ? 1.f : __expf(m - m_g));
    tl = warpReduceSum(tl);
    if (lane == 0)
        warp_l[warp] = tl;
    __syncthreads();
    if (warp == 0) {
        tl = (lane < BLOCK / 32) ? warp_l[lane] : 0.f;
        tl = warpReduceSum(tl);
        if (lane == 0)
            block_l = tl;
    }
    __syncthreads();
    m = m_g;
    l = block_l;
}

// =============================================================================
// Vector load / store helpers
// =============================================================================

__device__ __forceinline__ float4 load4(const float* p, int ld, int r, int c, int nr, int nc) {
    const int i = r * ld + c;
    if (r < nr && c + 3 < nc && (i & 3) == 0)
        return __ldg(reinterpret_cast<const float4*>(p + i));
    return make_float4((r < nr && c < nc) ? p[i] : 0.f, (r < nr && c + 1 < nc) ? p[i + 1] : 0.f,
                       (r < nr && c + 2 < nc) ? p[i + 2] : 0.f, (r < nr && c + 3 < nc) ? p[i + 3] : 0.f);
}

__device__ __forceinline__ void store4(float* p, int ld, int r, int c, float4 v, int nr, int nc) {
    const int i = r * ld + c;
    if (r < nr && c + 3 < nc && (i & 3) == 0) {
        *reinterpret_cast<float4*>(p + i) = v;
        return;
    }
    if (r < nr && c < nc)
        p[i] = v.x;
    if (r < nr && c + 1 < nc)
        p[i + 1] = v.y;
    if (r < nr && c + 2 < nc)
        p[i + 2] = v.z;
    if (r < nr && c + 3 < nc)
        p[i + 3] = v.w;
}

__device__ __forceinline__ float4 load4(const float* p, int i, int n) {
    return load4(p, n, 0, i, 1, n);
}
__device__ __forceinline__ void store4(float* p, int i, int n, float4 v) {
    store4(p, n, 0, i, v, 1, n);
}
