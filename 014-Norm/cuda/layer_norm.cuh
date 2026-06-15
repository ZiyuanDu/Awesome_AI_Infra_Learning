/*
 * layer_norm.cuh — Layer Normalization
 *
 * 公式:  y = (x - mean) * rsqrt(var + ε) * gamma + beta
 *
 * 累加器为 float2 (sum, sq_sum)，两个分量独立规约后计算 mean 和 inv_std。
 * gamma 和 beta 是长度为 K 的 per-element 可学习参数。
 */

#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include "norm_kernel.cuh"

namespace cuda_norm {

// ---------------------------------------------------------------------------
// float2 辅助函数 — warp/block 级求和规约
// ---------------------------------------------------------------------------
__device__ __forceinline__ float2 f2_add(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ float2 f2_shfl_xor(float2 val, int mask) {
    return make_float2(__shfl_xor_sync(0xffffffff, val.x, mask),
                       __shfl_xor_sync(0xffffffff, val.y, mask));
}

__device__ __forceinline__ float2 f2_warp_reduce_sum(float2 val) {
#pragma unroll
    for (int mask = 16; mask >= 1; mask >>= 1)
        val = f2_add(val, f2_shfl_xor(val, mask));
    return val;
}

template <int NUM_THREADS>
__device__ __forceinline__ float2 f2_block_reduce_sum(float2 val) {
    constexpr int NUM_WARPS = NUM_THREADS / 32;
    __shared__ float2 s_vals[NUM_WARPS];

    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    val = f2_warp_reduce_sum(val);
    if (lane == 0) s_vals[warp] = val;
    __syncthreads();

    val = (lane < NUM_WARPS) ? s_vals[lane] : make_float2(0.0f, 0.0f);
    if (warp == 0) val = f2_warp_reduce_sum(val);
    return val;
}

// ---------------------------------------------------------------------------
// LayerNorm 统计量: float2 (sum, sq_sum) → {mean, inv_std}
// ---------------------------------------------------------------------------
struct LayerNormStats {
    using accum_t = float2;
    using stat_t  = float2;  // {mean, inv_std}

    __device__ static void init(accum_t& a) {
        a = make_float2(0.0f, 0.0f);
    }

    template <typename T>
    __device__ static void accumulate(accum_t& a, const T* vals, int n) {
#pragma unroll
        for (int i = 0; i < n; ++i) {
            float v = static_cast<float>(vals[i]);
            a.x += v;        // sum
            a.y += v * v;    // sq_sum
        }
    }

    __device__ static accum_t warp_reduce(accum_t a, int /*group_width*/) {
        return f2_warp_reduce_sum(a);
    }

    template <int BLOCK_SIZE>
    __device__ static accum_t block_reduce(accum_t a) {
        return f2_block_reduce_sum<BLOCK_SIZE>(a);
    }

    __device__ static float2 compute(accum_t a, int K, float eps) {
        float mean = a.x / static_cast<float>(K);
        float var  = a.y / static_cast<float>(K) - mean * mean;
        var = fmaxf(var, 0.0f);  // 防止浮点误差导致负方差
        return make_float2(mean, rsqrtf(var + eps));
    }

    template <typename T>
    __device__ static void normalize(T* vals, float2 stat, int n) {
#pragma unroll
        for (int i = 0; i < n; ++i) {
            float v = static_cast<float>(vals[i]);
            vals[i] = static_cast<T>((v - stat.x) * stat.y);
        }
    }
};

// ---------------------------------------------------------------------------
// Dispatch: 根据 K 大小和数据类型选择最优 kernel 配置
// ---------------------------------------------------------------------------

template <typename T, int pack_size>
struct LayerNormWarpDispatch {
    static void launch(cudaStream_t stream,
                       const T* x, T* y, const T* gamma, const T* beta,
                       int N, int K, float eps) {
        constexpr int thread_group_width = 32;
        constexpr int groups_per_block  = 4;

        int cols_per_thread = (K + thread_group_width - 1) / thread_group_width;
        if (cols_per_thread < pack_size)
            cols_per_thread = pack_size;
        else
            cols_per_thread = ((cols_per_thread + pack_size - 1) / pack_size) * pack_size;

        dim3 block(thread_group_width, groups_per_block);
        int grid = (N + groups_per_block - 1) / groups_per_block;

        DirectLoad<T, float> load(x, K);
        AffineStore<float, T, true, true> store(y, K, gamma, beta);

#define LAUNCH_WARP(CPT) \
        NormWarpImpl<DirectLoad<T,float>, AffineStore<float,T,true,true>, \
                     float, LayerNormStats, pack_size, CPT, thread_group_width, 1> \
            <<<grid, block, 0, stream>>>(load, store, N, K, eps);

        if (cols_per_thread <= pack_size)        { LAUNCH_WARP(pack_size) }
        else if (cols_per_thread <= pack_size*2)  { LAUNCH_WARP(pack_size*2) }
        else if (cols_per_thread <= pack_size*4)  { LAUNCH_WARP(pack_size*4) }
        else if (cols_per_thread <= pack_size*8)  { LAUNCH_WARP(pack_size*8) }
        else if (cols_per_thread <= pack_size*16) { LAUNCH_WARP(pack_size*16) }
        else { LAUNCH_WARP(pack_size*16) }
#undef LAUNCH_WARP
    }
};

template <typename T, int pack_size, int block_size>
static void launch_ln_smem(cudaStream_t stream,
                            const T* x, T* y, const T* gamma, const T* beta,
                            int N, int K, float eps) {
    DirectLoad<T, float> load(x, K);
    AffineStore<float, T, true, true> store(y, K, gamma, beta);

    int smem_bytes = K * sizeof(float);
    int grid = N;

    NormBlockSMemImpl<DirectLoad<T, float>, AffineStore<float, T, true, true>,
                      float, LayerNormStats, pack_size, block_size>
        <<<grid, block_size, smem_bytes, stream>>>(load, store, N, K, eps);
}

template <typename T, int pack_size>
static void layer_norm_dispatch(cudaStream_t stream,
                                 const T* x, T* y, const T* gamma, const T* beta,
                                 int N, int K, float eps) {
    if (K <= 1024) {
        LayerNormWarpDispatch<T, pack_size>::launch(stream, x, y, gamma, beta, N, K, eps);
    } else {
        launch_ln_smem<T, pack_size, 256>(stream, x, y, gamma, beta, N, K, eps);
    }
}

// ---------------------------------------------------------------------------
// 公开 host API
// ---------------------------------------------------------------------------
inline void layer_norm_forward(const void* x, void* y,
                                const void* gamma, const void* beta,
                                int N, int K, float eps,
                                bool is_fp16, cudaStream_t stream = 0) {
    if (is_fp16) {
        layer_norm_dispatch<__half, 8>(
            stream, (const __half*)x, (__half*)y,
            (const __half*)gamma, (const __half*)beta, N, K, eps);
    } else {
        layer_norm_dispatch<float, 4>(
            stream, (const float*)x, (float*)y,
            (const float*)gamma, (const float*)beta, N, K, eps);
    }
}

} // namespace cuda_norm
