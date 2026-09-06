#pragma once
#include <cuda_runtime.h>
#include <math.h>

#define CEIL(a, b) (((a) + (b) - 1) / (b))


struct sum_op {
    __device__ __forceinline__ float operator()(float a, float b) const { return a + b; }
    __device__ __forceinline__ static float init() { return 0.f; }
};

struct max_op {
    __device__ __forceinline__ float operator()(float a, float b) const { return fmaxf(a, b); }
    __device__ __forceinline__ static float init() { return -INFINITY; }
};

template <typename Op>
__device__ __forceinline__ float warpReduce(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = Op{}(v, __shfl_xor_sync(0xffffffff, v, off));
    return v;
}

__device__ __forceinline__ float warpReduceSum(float v) {
    return warpReduce<sum_op>(v);
}
__device__ __forceinline__ float warpReduceMax(float v) {
    return warpReduce<max_op>(v);
}

__device__ __forceinline__ float warpScanInclusive(float v) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        float n = __shfl_up_sync(0xffffffff, v, off);
        if (lane >= off)
            v += n;
    }
    return v;
}

template <int BLOCK, typename Op>
__device__ __forceinline__ float blockReduce(float v) {
    __shared__ float smem[BLOCK / 32];
    __shared__ float res;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    v = warpReduce<Op>(v);
    if (lane == 0)
        smem[wid] = v;
    __syncthreads();

    if (wid == 0) {
        v = (lane < BLOCK / 32) ? smem[lane] : Op::init();
        v = warpReduce<Op>(v);
        if (lane == 0)
            res = v;
    }
    __syncthreads();
    return res;
}

template <int BLOCK>
__device__ __forceinline__ float blockReduceSum(float v) {
    return blockReduce<BLOCK, sum_op>(v);
}
template <int BLOCK>
__device__ __forceinline__ float blockReduceMax(float v) {
    return blockReduce<BLOCK, max_op>(v);
}
template <int BLOCK>
__device__ __forceinline__ float blockScanInclusive(float v) {
    __shared__ float wsum[BLOCK / 32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    v = warpScanInclusive(v);
    if (lane == 31)
        wsum[wid] = v;
    __syncthreads();

    if (wid == 0) {
        float w = (lane < BLOCK / 32) ? wsum[lane] : 0.f;
        w = warpScanInclusive(w);
        if (lane < BLOCK / 32)
            wsum[lane] = w;
    }
    __syncthreads();

    if (wid > 0)
        v += wsum[wid - 1];
    return v;
}
template <int BLOCK>
__device__ __forceinline__ float blockScanExclusive(float v) {
    return blockScanInclusive<BLOCK>(v) - v;
}

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
    for (int off = 16; off > 0; off >>= 1)
        onlineMerge(m, l, __shfl_xor_sync(0xffffffff, m, off),
                    __shfl_xor_sync(0xffffffff, l, off));
}

template <int BLOCK>
__device__ __forceinline__ void blockOnlineReduce(float& m, float& l) {
    __shared__ float sm[BLOCK / 32], sl[BLOCK / 32], bm, bl;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    warpOnlineReduce(m, l);
    if (lane == 0) {
        sm[wid] = m;
        sl[wid] = l;
    }
    __syncthreads();

    if (wid == 0) {
        m = (lane < BLOCK / 32) ? sm[lane] : -INFINITY;
        l = (lane < BLOCK / 32) ? sl[lane] : 0.f;
        warpOnlineReduce(m, l);
        if (lane == 0) {
            bm = m;
            bl = l;
        }
    }
    __syncthreads();
    m = bm;
    l = bl;
}

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

/* 1D：load4(p, i, n) / store4(p, i, n, v) */
__device__ __forceinline__ float4 load4(const float* p, int i, int n) {
    return load4(p, n, 0, i, 1, n);
}
__device__ __forceinline__ void store4(float* p, int i, int n, float4 v) {
    store4(p, n, 0, i, v, 1, n);
}
