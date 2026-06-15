/*
 * rms_norm.cuh — RMS Normalization
 *
 * 公式:  y = x * rsqrt(mean(x²) + ε) * gamma
 *
 * 使用 norm_kernel.cuh 中的通用骨架，实例化为 RMSNorm 专用统计量。
 * gamma 是长度为 K 的 per-element 权重向量（可学习参数）。
 */

#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include "norm_kernel.cuh"

namespace cuda_norm {

// ---------------------------------------------------------------------------
// RMSNorm 统计量: 线程局部平方和 → inv_rms
// ---------------------------------------------------------------------------
struct RMSNormStats {
    using accum_t = float;
    using stat_t  = float;   // inv_rms

    __device__ static void init(accum_t& a) { a = 0.0f; }

    template <typename T>
    __device__ static void accumulate(accum_t& a, const T* vals, int n) {
#pragma unroll
        for (int i = 0; i < n; ++i) {
            float v = static_cast<float>(vals[i]);
            a += v * v;
        }
    }

    __device__ static accum_t warp_reduce(accum_t a, int /*group_width*/) {
        return warp_reduce_sum(a);
    }

    template <int BLOCK_SIZE>
    __device__ static accum_t block_reduce(accum_t a) {
        return block_reduce_sum<BLOCK_SIZE>(a);
    }

    __device__ static float compute(accum_t a, int K, float eps) {
        return rsqrtf(a / static_cast<float>(K) + eps);
    }

    template <typename T>
    __device__ static void normalize(T* vals, float inv_rms, int n) {
#pragma unroll
        for (int i = 0; i < n; ++i) {
            vals[i] = static_cast<T>(static_cast<float>(vals[i]) * inv_rms);
        }
    }
};

// ---------------------------------------------------------------------------
// Dispatch: 根据 K 大小和数据类型选择最优 kernel 配置
// ---------------------------------------------------------------------------

template <typename T, int pack_size>
struct RMSNormWarpDispatch {
    static void launch(cudaStream_t stream,
                       const T* x, T* y, const T* gamma,
                       int N, int K, float eps) {
        constexpr int thread_group_width = 32;
        constexpr int groups_per_block  = 4;

        int cols_per_thread = (K + thread_group_width - 1) / thread_group_width;
        // 确保至少一个 pack，且对齐到 pack_size 的倍数
        if (cols_per_thread < pack_size)
            cols_per_thread = pack_size;
        else
            cols_per_thread = ((cols_per_thread + pack_size - 1) / pack_size) * pack_size;

        dim3 block(thread_group_width, groups_per_block);
        int grid = (N + groups_per_block - 1) / groups_per_block;

        DirectLoad<T, float> load(x, K);
        AffineStore<float, T, true, false> store(y, K, gamma, nullptr);

#define LAUNCH_WARP(CPT) \
        NormWarpImpl<DirectLoad<T,float>, AffineStore<float,T,true,false>, \
                     float, RMSNormStats, pack_size, CPT, thread_group_width, 1> \
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
static void launch_rms_smem(cudaStream_t stream,
                             const T* x, T* y, const T* gamma,
                             int N, int K, float eps) {
    DirectLoad<T, float> load(x, K);
    AffineStore<float, T, true, false> store(y, K, gamma, nullptr);

    int smem_bytes = K * sizeof(float);
    int grid = N;

    NormBlockSMemImpl<DirectLoad<T, float>, AffineStore<float, T, true, false>,
                      float, RMSNormStats, pack_size, block_size>
        <<<grid, block_size, smem_bytes, stream>>>(load, store, N, K, eps);
}

template <typename T, int pack_size>
static void rms_norm_dispatch(cudaStream_t stream,
                               const T* x, T* y, const T* gamma,
                               int N, int K, float eps) {
    if (K <= 1024) {
        RMSNormWarpDispatch<T, pack_size>::launch(stream, x, y, gamma, N, K, eps);
    } else {
        launch_rms_smem<T, pack_size, 256>(stream, x, y, gamma, N, K, eps);
    }
}

// ---------------------------------------------------------------------------
// 公开 host API
// ---------------------------------------------------------------------------
inline void rms_norm_forward(const void* x, void* y, const void* gamma,
                              int N, int K, float eps,
                              bool is_fp16, cudaStream_t stream = 0) {
    if (is_fp16) {
        rms_norm_dispatch<__half, 8>(
            stream, (const __half*)x, (__half*)y, (const __half*)gamma, N, K, eps);
    } else {
        rms_norm_dispatch<float, 4>(
            stream, (const float*)x, (float*)y, (const float*)gamma, N, K, eps);
    }
}

} // namespace cuda_norm
