/*
 * norm_kernel.cuh 
 *
 * 将"如何遍历数据、如何做规约"的策略与"算什么统计量、如何归一化"的公式解耦。
 * RMSNorm 和 LayerNorm 共享同一套 kernel 模板，仅通过 Stats类型参数来区分。
 *
 * Stats 类型需提供:
 *   accum_t              — 线程局部累加器类型
 *   stat_t               — 最终统计量类型 (RMSNorm=float, LayerNorm=float2)
 *   init(acc)            — 初始化累加器
 *   accumulate(acc, vals, n) — 将 n 个元素累积到累加器
 *   warp_reduce(acc, w)  — warp 内规约
 *   block_reduce<BLK>(acc)— block 内规约
 *   compute(acc, K, eps) — 从累加值计算最终统计量
 *   normalize(vals, stat, n) — 就地归一化（不含 affine）
 *
 * 两种策略，按 K (归一化维度) 自动选择:
 *
 *   K <= 1024 → NormWarpImpl
 *     一个 warp group 处理一行。一个 block 内含多个 warp group，
 *     各行之间完全并行。统计量只需 warp 内 reduce，不需要 __syncthreads。
 *     适合 N 很大、K 较小的场景（如 LLM 的小 hidden_size）。
 *
 *   K > 1024 → NormBlockSMemImpl
 *     整个 block 协作处理一行。第一趟: 读 global → 写 shared memory
 *     + 累加统计量。Block reduce 后 broadcast 统计量。第二趟:
 *     从 shared memory 读取 → 归一化 → 写 global。省去了第二次
 *     global memory 读取，对带宽 bound 的大 K 场景收益巨大。
 */

#pragma once
#include "reduce.cuh"
#include "io.cuh"

namespace cuda_norm {

// 策略一: WarpImpl  (K ≤ 1024)
template <typename LOAD, typename STORE, typename ComputeType, typename Stats,
          int pack_size, int cols_per_thread, int thread_group_width,
          int rows_per_access>
__global__ void NormWarpImpl(LOAD load, STORE store,
                              int rows, int cols, float eps) {
    static_assert(cols_per_thread >= pack_size &&
                  cols_per_thread % pack_size == 0,
                  "cols_per_thread 必须 >= pack_size 且是 pack_size 的倍数");
    static_assert(thread_group_width <= WARP_SIZE, "");
    constexpr int num_packs = cols_per_thread / pack_size;

    const int global_group_id = blockIdx.x * blockDim.y + threadIdx.y;
    const int num_groups      = gridDim.x * blockDim.y;
    const int lane            = threadIdx.x;

    ComputeType buf[rows_per_access][cols_per_thread];

    for (int base_row = global_group_id * rows_per_access;
         base_row < rows;
         base_row += num_groups * rows_per_access) {

        // 第 1 步: 加载数据 + 累加统计量
        typename Stats::accum_t acc[rows_per_access];
#pragma unroll
        for (int r = 0; r < rows_per_access; ++r) {
            Stats::init(acc[r]);
            ComputeType* row_buf = buf[r];
            const int row = base_row + r;
            if (row >= rows) continue;

#pragma unroll
            for (int p = 0; p < num_packs; ++p) {
                const int col = (p * thread_group_width + lane) * pack_size;
                load.template load<pack_size>(row_buf + p * pack_size, row, col);
                Stats::accumulate(acc[r], row_buf + p * pack_size, pack_size);
            }
        }

        // 第 2 步: warp 规约 → 计算统计量
        typename Stats::stat_t stat[rows_per_access];
#pragma unroll
        for (int r = 0; r < rows_per_access; ++r) {
            if (base_row + r >= rows) continue;
            acc[r] = Stats::warp_reduce(acc[r], thread_group_width);
            stat[r] = Stats::compute(acc[r], cols, eps);
        }

        // 第 3 步: 寄存器内就地归一化
#pragma unroll
        for (int r = 0; r < rows_per_access; ++r) {
            const int row = base_row + r;
            if (row >= rows) continue;
            Stats::normalize(buf[r], stat[r], cols_per_thread);
        }

        // 第 4 步: 写出 (AffineStore 在此融合 gamma/beta
#pragma unroll
        for (int r = 0; r < rows_per_access; ++r) {
            const int row = base_row + r;
            if (row >= rows) continue;
#pragma unroll
            for (int p = 0; p < num_packs; ++p) {
                const int col = (p * thread_group_width + lane) * pack_size;
                store.template store<pack_size>(buf[r] + p * pack_size, row, col);
            }
        }
    }
}

// 策略二: BlockSMemImpl  (K > 1024)
template <typename LOAD, typename STORE, typename ComputeType, typename Stats,
          int pack_size, int block_size>
__global__ void NormBlockSMemImpl(LOAD load, STORE store,
                                   int rows, int cols, float eps) {
    const int num_packs = cols / pack_size;

    extern __shared__ char smem_buf[];
    ComputeType* smem = reinterpret_cast<ComputeType*>(smem_buf);
    // smem 布局: [cols] 个 ComputeType，缓存原始 x 值

    __shared__ typename Stats::stat_t s_stat;

    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        // 第 1 步: 读 global → 写 smem + 累加统计量
        typename Stats::accum_t acc;
        Stats::init(acc);

        for (int p = threadIdx.x; p < num_packs; p += block_size) {
            ComputeType vals[pack_size];
            load.template load<pack_size>(vals, row, p * pack_size);
#pragma unroll
            for (int i = 0; i < pack_size; ++i) {
                smem[p * pack_size + i] = vals[i];
            }
            Stats::accumulate(acc, vals, pack_size);
        }

        // 第 2 步: block 规约 → 计算统计量 → broadcast
        // block_reduce 后仅 warp0 持有正确结果，需通过 shared memory 广播
        acc = Stats::template block_reduce<block_size>(acc);
        if (threadIdx.x == 0) s_stat = Stats::compute(acc, cols, eps);
        __syncthreads();

        typename Stats::stat_t stat = s_stat;

        // 第 3 步: 从 smem 读取 → 归一化 → 写 global
        for (int p = threadIdx.x; p < num_packs; p += block_size) {
            ComputeType vals[pack_size];
#pragma unroll
            for (int i = 0; i < pack_size; ++i) {
                vals[i] = smem[p * pack_size + i];
            }
            Stats::normalize(vals, stat, pack_size);
            store.template store<pack_size>(vals, row, p * pack_size);
        }
        __syncthreads();  // 保护 smem 供下一行复用
    }
}

} // namespace cuda_norm
