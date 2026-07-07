/*
 * softmax_online.cuh — Online Softmax / LogSoftmax
 *
 * 实现三种策略，根据 cols 大小自动选择:
 *
 *   Strategy 1 (cols < 1024):  Warp-level kernel
 *     数据在寄存器中，一次全局读 + 一次全局写，warp 内 shuffle 通信。
 *     对小 cols 场景已是最优。
 *
 *   Strategy 2 (cols ≥ 1024):  Block SMem Online
 *     整个 block 协作处理一行。Pass 1 读 global → 写 shared memory
 *     同时做 online (m,s) 跟踪。Block reduce 后 broadcast 统计量。
 *     Pass 2 从 shared memory 读取（省去第二次 global 读）。
 *     需要 smem = cols * sizeof(ComputeType) bytes。
 *
 *   Strategy 3 (cols ≥ 1024, smem 不足): Block Uncached Online
 *     Pass 1: 读 global → online (m,s) 跟踪 (不写 smem)
 *     Pass 2: 读 global → normalize → 写 global
 *     CACHE_OPT: Pass 2 逆序遍历，利用 L2 temporal locality。
 *
 * 改进点 (相比 naive 3-pass):
 *   - Online 算法将 max-finding 和 sum-accumulation 合并为一次遍历
 *   - 2 次 global read vs naive 的 3 次 (33% 减少)
 *   - CACHE_OPT 进一步提升大 row 的 L2 命中率
 */

#pragma once
#include <cub/cub.cuh>
#include <math_constants.h>
#include <assert.h>
#include <cuda.h>

#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif

#include "reduce.cuh"

namespace cuda_softmax {

// ---- Pack types for vectorised memory access --------------------------------

template<typename T, int N>
struct GetPackType {
  using type = typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type;
};

template<typename T, int N>
using PackType = typename GetPackType<T, N>::type;

template<typename T, int N>
union Pack {
  static_assert(sizeof(PackType<T, N>) == sizeof(T) * N, "");
  __device__ Pack() {}
  PackType<T, N> storage;
  T elem[N];
};

// ---- Direct load / store functors -------------------------------------------

template<typename SRC, typename DST>
struct DirectLoad {
  DirectLoad(const SRC* src, int64_t row_size) : src(src), row_size(row_size) {}
  template<int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) const {
    Pack<SRC, N> pack;
    const int64_t offset = (row * row_size + col) / N;
    pack.storage = *(reinterpret_cast<const PackType<SRC, N>*>(src) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) { dst[i] = static_cast<DST>(pack.elem[i]); }
  }
  const SRC* src;
  int64_t row_size;
};

template<typename SRC, typename DST>
struct DirectStore {
  DirectStore(DST* dst, int64_t row_size) : dst(dst), row_size(row_size) {}
  template<int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    Pack<DST, N> pack;
    const int64_t offset = (row * row_size + col) / N;
#pragma unroll
    for (int i = 0; i < N; ++i) { pack.elem[i] = static_cast<DST>(src[i]); }
    *(reinterpret_cast<PackType<DST, N>*>(dst) + offset) = pack.storage;
  }
  DST* dst;
  int64_t row_size;
};

// ---- Compute-type promotion -------------------------------------------------

template<typename T> struct DefaultComputeType { using type = T; };
template<> struct DefaultComputeType<__half> { using type = float; };
#if CUDA_VERSION >= 11000
template<> struct DefaultComputeType<__nv_bfloat16> { using type = float; };
#endif

// ---- Algorithm tag ----------------------------------------------------------

enum class Algorithm {
  kSoftmax = 0,
  kLogSoftmax = 1,
};

// ---- Occupancy-aware grid sizing -------------------------------------------

inline cudaError_t GetNumBlocks(int64_t block_size, int64_t max_blocks, int64_t waves,
                                int* num_blocks) {
  int dev;
  cudaError_t err = cudaGetDevice(&dev);
  if (err != cudaSuccess) return err;
  int sm_count;
  err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
  if (err != cudaSuccess) return err;
  int tpm;
  err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
  if (err != cudaSuccess) return err;
  *num_blocks = std::max<int>(1, std::min<int64_t>(max_blocks, sm_count * tpm / block_size * waves));
  return cudaSuccess;
}

// =========================================================================
// Strategy 1: Warp-level kernel  (cols <= 1024)
// =========================================================================

template<typename LOAD, typename STORE, typename ComputeType, int pack_size,
         int cols_per_thread, int thread_group_width, int rows_per_access,
         bool padding, Algorithm algorithm>
__global__ void SoftmaxWarpImpl(LOAD load, STORE store, const int64_t rows,
                                const int64_t cols) {
  static_assert(cols_per_thread % pack_size == 0, "");
  static_assert(thread_group_width <= kWarpSize, "");
  static_assert(kWarpSize % thread_group_width == 0, "");
  constexpr int num_packs = cols_per_thread / pack_size;
  assert(cols <= cols_per_thread * thread_group_width);
  ComputeType buf[rows_per_access][cols_per_thread];
  const int global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y;
  const int num_global_thread_group = gridDim.x * blockDim.y;
  const int lane_id = threadIdx.x;
  const int64_t step = num_global_thread_group * rows_per_access;

  for (int64_t row = global_thread_group_id * rows_per_access; row < rows; row += step) {
    ComputeType thread_max[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      thread_max[row_id] = -Inf<ComputeType>();
      ComputeType* row_buf = buf[row_id];
#pragma unroll
      for (int pack_id = 0; pack_id < num_packs; ++pack_id) {
        const int pack_offset = pack_id * pack_size;
        const int col = (pack_id * thread_group_width + lane_id) * pack_size;
        if (!padding || col < cols) {
          load.template load<pack_size>(row_buf + pack_offset, row + row_id, col);
#pragma unroll
          for (int i = 0; i < pack_size; ++i) {
            thread_max[row_id] = max(thread_max[row_id], row_buf[pack_offset + i]);
          }
        } else {
#pragma unroll
          for (int i = 0; i < pack_size; ++i) { row_buf[pack_offset + i] = -Inf<ComputeType>(); }
        }
      }
    }

    ComputeType warp_max[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      warp_max[row_id] = WarpAllReduce<MaxOp, ComputeType, thread_group_width>(thread_max[row_id]);
    }

    ComputeType thread_sum[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      thread_sum[row_id] = 0;
      ComputeType* row_buf = buf[row_id];
#pragma unroll
      for (int i = 0; i < cols_per_thread; ++i) {
        if (algorithm == Algorithm::kSoftmax) {
          row_buf[i] = Exp(row_buf[i] - warp_max[row_id]);
          thread_sum[row_id] += row_buf[i];
        } else {  // kLogSoftmax
          row_buf[i] -= warp_max[row_id];
          thread_sum[row_id] += Exp(row_buf[i]);
        }
      }
    }

    ComputeType warp_sum[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      warp_sum[row_id] = WarpAllReduce<SumOp, ComputeType, thread_group_width>(thread_sum[row_id]);
    }

#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      ComputeType* row_buf = buf[row_id];
#pragma unroll
      for (int i = 0; i < cols_per_thread; ++i) {
        if (algorithm == Algorithm::kSoftmax) {
          row_buf[i] = Div(row_buf[i], warp_sum[row_id]);
        } else {
          row_buf[i] -= Log(warp_sum[row_id]);
        }
      }
#pragma unroll
      for (int i = 0; i < num_packs; ++i) {
        const int col = (i * thread_group_width + lane_id) * pack_size;
        if (!padding || col < cols) {
          store.template store<pack_size>(row_buf + i * pack_size, row + row_id, col);
        }
      }
    }
  }
}

// ---- Warp path dispatch helpers ---------------------------------------------

template<typename LOAD, typename STORE, typename ComputeType, int pack_size,
         int cols_per_thread, int thread_group_width, int rows_per_access,
         bool padding, Algorithm algorithm>
inline cudaError_t LaunchSoftmaxWarpImpl(cudaStream_t stream, LOAD load, STORE store,
                                         const int64_t rows, const int64_t cols) {
  constexpr int block_size = 128;
  constexpr int waves = 32;
  static_assert(block_size % thread_group_width == 0, "");
  constexpr int thread_groups_per_block = block_size / thread_group_width;
  dim3 block_dim(thread_group_width, thread_groups_per_block);
  const int64_t num_blocks =
      (rows / rows_per_access + thread_groups_per_block - 1) / thread_groups_per_block;
  int grid_dim_x;
  cudaError_t err = GetNumBlocks(block_size, num_blocks, waves, &grid_dim_x);
  if (err != cudaSuccess) return err;
  SoftmaxWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread, thread_group_width,
                  rows_per_access, padding, algorithm>
      <<<grid_dim_x, block_dim, 0, stream>>>(load, store, rows, cols);
  return cudaPeekAtLastError();
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size,
         int cols_per_thread, int thread_group_width, int rows_per_access,
         Algorithm algorithm>
inline cudaError_t DispatchSoftmaxWarpImplPadding(cudaStream_t stream, LOAD load, STORE store,
                                                  const int64_t rows, const int64_t cols) {
  if (cols == cols_per_thread * thread_group_width) {
    return LaunchSoftmaxWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread,
                                 thread_group_width, rows_per_access, false, algorithm>(
        stream, load, store, rows, cols);
  } else {
    return LaunchSoftmaxWarpImpl<LOAD, STORE, ComputeType, pack_size, cols_per_thread,
                                 thread_group_width, rows_per_access, true, algorithm>(
        stream, load, store, rows, cols);
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, Algorithm algorithm>
typename std::enable_if<pack_size == 1, cudaError_t>::type
DispatchSoftmaxWarpImplCols(cudaStream_t stream, LOAD load, STORE store,
                            const int64_t rows, const int64_t cols) {
  if (cols <= 0) return cudaErrorInvalidValue;
#define DEFINE_ONE_ELIF(tgw)                                                      \
  else if (cols <= (tgw)*pack_size) {                                             \
    if (rows % 2 == 0) {                                                          \
      return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size,  \
                        pack_size, tgw, 2, algorithm>(stream, load, store, rows, cols); \
    } else {                                                                      \
      return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size,  \
                        pack_size, tgw, 1, algorithm>(stream, load, store, rows, cols); \
    }                                                                             \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                      \
  else if (cols <= (col)*kWarpSize) {                                             \
    return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size,    \
                      col, kWarpSize, 1, algorithm>(stream, load, store, rows, cols); \
  }
  DEFINE_ONE_ELIF(2)  DEFINE_ONE_ELIF(3)  DEFINE_ONE_ELIF(4)  DEFINE_ONE_ELIF(5)
  DEFINE_ONE_ELIF(6)  DEFINE_ONE_ELIF(7)  DEFINE_ONE_ELIF(8)  DEFINE_ONE_ELIF(9)
  DEFINE_ONE_ELIF(10) DEFINE_ONE_ELIF(11) DEFINE_ONE_ELIF(12) DEFINE_ONE_ELIF(13)
  DEFINE_ONE_ELIF(14) DEFINE_ONE_ELIF(15) DEFINE_ONE_ELIF(16) DEFINE_ONE_ELIF(17)
  DEFINE_ONE_ELIF(18) DEFINE_ONE_ELIF(19) DEFINE_ONE_ELIF(20) DEFINE_ONE_ELIF(21)
  DEFINE_ONE_ELIF(22) DEFINE_ONE_ELIF(23) DEFINE_ONE_ELIF(24) DEFINE_ONE_ELIF(25)
  DEFINE_ONE_ELIF(26) DEFINE_ONE_ELIF(27) DEFINE_ONE_ELIF(28) DEFINE_ONE_ELIF(29)
  DEFINE_ONE_ELIF(30) DEFINE_ONE_ELIF(31) DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
  else return cudaErrorInvalidValue;
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, Algorithm algorithm>
typename std::enable_if<pack_size == 2, cudaError_t>::type
DispatchSoftmaxWarpImplCols(cudaStream_t stream, LOAD load, STORE store,
                            const int64_t rows, const int64_t cols) {
  if (cols <= 0) return cudaErrorInvalidValue;
#define DEFINE_ONE_ELIF(tgw)                                                      \
  else if (cols <= (tgw)*pack_size) {                                             \
    if (rows % 2 == 0) {                                                          \
      return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size,  \
                        pack_size, tgw, 2, algorithm>(stream, load, store, rows, cols); \
    } else {                                                                      \
      return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size,  \
                        pack_size, tgw, 1, algorithm>(stream, load, store, rows, cols); \
    }                                                                             \
  }
  DEFINE_ONE_ELIF(1) DEFINE_ONE_ELIF(2) DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8) DEFINE_ONE_ELIF(16) DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                      \
  else if (cols <= (col)*kWarpSize) {                                             \
    return DispatchSoftmaxWarpImplPadding<LOAD, STORE, ComputeType, pack_size,    \
                      col, kWarpSize, 1, algorithm>(stream, load, store, rows, cols); \
  }
  DEFINE_ONE_ELIF(4)  DEFINE_ONE_ELIF(6)  DEFINE_ONE_ELIF(8)  DEFINE_ONE_ELIF(10)
  DEFINE_ONE_ELIF(12) DEFINE_ONE_ELIF(14) DEFINE_ONE_ELIF(16) DEFINE_ONE_ELIF(18)
  DEFINE_ONE_ELIF(20) DEFINE_ONE_ELIF(22) DEFINE_ONE_ELIF(24) DEFINE_ONE_ELIF(26)
  DEFINE_ONE_ELIF(28) DEFINE_ONE_ELIF(30) DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
  else return cudaErrorInvalidValue;
}

template<typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
struct DispatchSoftmaxWarpImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store,
                         const int64_t rows, const int64_t cols) {
    if (cols % 2 == 0) {
      return DispatchSoftmaxWarpImplCols<LOAD, STORE, ComputeType, 2, algorithm>(
          stream, load, store, rows, cols);
    } else {
      return DispatchSoftmaxWarpImplCols<LOAD, STORE, ComputeType, 1, algorithm>(
          stream, load, store, rows, cols);
    }
  }
};

template<typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm>
inline cudaError_t DispatchSoftmaxWarpImpl(cudaStream_t stream, LOAD load, STORE store,
                                           const int64_t rows, const int64_t cols) {
  return DispatchSoftmaxWarpImplPackSize<LOAD, STORE, ComputeType, algorithm>()(
      stream, load, store, rows, cols);
}

// =========================================================================
// Strategy 2: Block-level with shared memory — ONLINE algorithm
// =========================================================================

template<typename LOAD, typename STORE, typename ComputeType, int pack_size,
         int block_size, Algorithm algorithm, bool CACHE_OPT>
__global__ void SoftmaxBlockSMemOnlineImpl(LOAD load, STORE store, const int64_t rows,
                                           const int64_t cols) {
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  auto* buf = reinterpret_cast<ComputeType*>(shared_buf);
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = cols / pack_size;

  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    // --- Pass 1: load to shared memory, online (m,s) tracking ---
    OnlinePair<ComputeType> acc = {-Inf<ComputeType>(), 0};
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        buf[i * num_packs + pack_id] = pack[i];
        ComputeType m_new = max(acc.m, pack[i]);
        acc.s = acc.s * Exp(acc.m - m_new) + Exp(pack[i] - m_new);
        acc.m = m_new;
      }
    }
    OnlinePair<ComputeType> result = BlockAllReduceOnline<ComputeType, block_size>(acc);
    const ComputeType row_max = result.m;
    const ComputeType row_sum = result.s;

    // --- Pass 2: normalise from smem and store ---
    if (CACHE_OPT) {
      for (int pack_id = num_packs - 1 - tid; pack_id >= 0; pack_id -= block_size) {
        ComputeType pack[pack_size];
#pragma unroll
        for (int i = 0; i < pack_size; ++i) {
          ComputeType x = buf[i * num_packs + pack_id];
          if (algorithm == Algorithm::kSoftmax) {
            pack[i] = Div(Exp(x - row_max), row_sum);
          } else {
            pack[i] = (x - row_max) - Log(row_sum);
          }
        }
        store.template store<pack_size>(pack, row, pack_id * pack_size);
      }
    } else {
      for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
        ComputeType pack[pack_size];
#pragma unroll
        for (int i = 0; i < pack_size; ++i) {
          ComputeType x = buf[i * num_packs + pack_id];
          if (algorithm == Algorithm::kSoftmax) {
            pack[i] = Div(Exp(x - row_max), row_sum);
          } else {
            pack[i] = (x - row_max) - Log(row_sum);
          }
        }
        store.template store<pack_size>(pack, row, pack_id * pack_size);
      }
    }
  }
}

// ---- Strategy 2 dispatch ----------------------------------------------------

template<typename LOAD, typename STORE, typename ComputeType, int pack_size,
         int block_size, Algorithm algorithm, bool CACHE_OPT>
inline cudaError_t LaunchSoftmaxBlockSMemOnlineImpl(cudaStream_t stream, LOAD load,
                                                    STORE store, int smem,
                                                    const int64_t rows, const int64_t cols) {
  constexpr int waves = 32;
  int grid_dim_x;
  cudaError_t err = GetNumBlocks(block_size, rows, waves, &grid_dim_x);
  if (err != cudaSuccess) return err;
  SoftmaxBlockSMemOnlineImpl<LOAD, STORE, ComputeType, pack_size, block_size,
                              algorithm, CACHE_OPT>
      <<<grid_dim_x, block_size, smem, stream>>>(load, store, rows, cols);
  return cudaPeekAtLastError();
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size,
         Algorithm algorithm, bool CACHE_OPT>
inline cudaError_t TryDispatchSoftmaxBlockSMemOnlineBlockSize(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
    const int64_t cols, bool* success) {
  constexpr int block_size_conf_1 = 128;
  constexpr int block_size_conf_2 = 256;
  constexpr int block_size_conf_3 = 512;
  constexpr int block_size_conf_4 = 1024;
  const size_t smem = cols * sizeof(ComputeType);

  int max_active_blocks_conf_1;
  cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks_conf_1,
      SoftmaxBlockSMemOnlineImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_1,
                                  algorithm, CACHE_OPT>,
      block_size_conf_1, smem);
  if (err != cudaSuccess) return err;
  if (max_active_blocks_conf_1 <= 0) { *success = false; return cudaSuccess; }

  int max_active_blocks_conf_4;
  err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks_conf_4,
      SoftmaxBlockSMemOnlineImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_4,
                                  algorithm, CACHE_OPT>,
      block_size_conf_4, smem);
  if (err != cudaSuccess) return err;
  if (max_active_blocks_conf_4 == max_active_blocks_conf_1) {
    *success = true;
    return LaunchSoftmaxBlockSMemOnlineImpl<LOAD, STORE, ComputeType, pack_size,
                                            block_size_conf_4, algorithm, CACHE_OPT>(
        stream, load, store, smem, rows, cols);
  }

  int max_active_blocks_conf_3;
  err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks_conf_3,
      SoftmaxBlockSMemOnlineImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_3,
                                  algorithm, CACHE_OPT>,
      block_size_conf_3, smem);
  if (err != cudaSuccess) return err;
  if (max_active_blocks_conf_3 == max_active_blocks_conf_1) {
    *success = true;
    return LaunchSoftmaxBlockSMemOnlineImpl<LOAD, STORE, ComputeType, pack_size,
                                            block_size_conf_3, algorithm, CACHE_OPT>(
        stream, load, store, smem, rows, cols);
  }

  int max_active_blocks_conf_2;
  err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_active_blocks_conf_2,
      SoftmaxBlockSMemOnlineImpl<LOAD, STORE, ComputeType, pack_size, block_size_conf_2,
                                  algorithm, CACHE_OPT>,
      block_size_conf_2, smem);
  if (err != cudaSuccess) return err;
  if (max_active_blocks_conf_2 == max_active_blocks_conf_1) {
    *success = true;
    return LaunchSoftmaxBlockSMemOnlineImpl<LOAD, STORE, ComputeType, pack_size,
                                            block_size_conf_2, algorithm, CACHE_OPT>(
        stream, load, store, smem, rows, cols);
  }

  *success = true;
  return LaunchSoftmaxBlockSMemOnlineImpl<LOAD, STORE, ComputeType, pack_size,
                                          block_size_conf_1, algorithm, CACHE_OPT>(
      stream, load, store, smem, rows, cols);
}

template<typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm,
         bool CACHE_OPT>
struct TryDispatchSoftmaxBlockSMemOnlinePackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store,
                         const int64_t rows, const int64_t cols, bool* success) {
    if (cols % 2 == 0) {
      return TryDispatchSoftmaxBlockSMemOnlineBlockSize<LOAD, STORE, ComputeType, 2,
                                                        algorithm, CACHE_OPT>(
          stream, load, store, rows, cols, success);
    } else {
      return TryDispatchSoftmaxBlockSMemOnlineBlockSize<LOAD, STORE, ComputeType, 1,
                                                        algorithm, CACHE_OPT>(
          stream, load, store, rows, cols, success);
    }
  }
};

template<typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm,
         bool CACHE_OPT>
inline cudaError_t TryDispatchSoftmaxBlockSMemOnline(cudaStream_t stream, LOAD load,
                                                     STORE store, const int64_t rows,
                                                     const int64_t cols, bool* success) {
  return TryDispatchSoftmaxBlockSMemOnlinePackSize<LOAD, STORE, ComputeType, algorithm,
                                                    CACHE_OPT>()(
      stream, load, store, rows, cols, success);
}

template<typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm,
         bool CACHE_OPT>
inline cudaError_t DispatchSoftmaxBlockSMemOnline(cudaStream_t stream, LOAD load,
                                                  STORE store, const int64_t rows,
                                                  const int64_t cols) {
  bool success;
  cudaError_t err = TryDispatchSoftmaxBlockSMemOnline<LOAD, STORE, ComputeType, algorithm,
                                                       CACHE_OPT>(
      stream, load, store, rows, cols, &success);
  if (err != cudaSuccess) return err;
  if (!success) {
    return cudaErrorInvalidValue;  // caller will try uncached
  }
  return cudaSuccess;
}

// =========================================================================
// Strategy 3: Block-level uncached — ONLINE algorithm
// =========================================================================

template<typename LOAD, typename STORE, typename ComputeType, int pack_size,
         int block_size, Algorithm algorithm, bool CACHE_OPT>
__global__ void SoftmaxBlockUncachedOnlineImpl(LOAD load, STORE store,
                                               const int64_t rows, const int64_t cols) {
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = cols / pack_size;

  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    // --- Pass 1: online (m,s) accumulation in a single sweep ---
    OnlinePair<ComputeType> acc = {-Inf<ComputeType>(), 0};
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        ComputeType m_new = max(acc.m, pack[i]);
        acc.s = acc.s * Exp(acc.m - m_new) + Exp(pack[i] - m_new);
        acc.m = m_new;
      }
    }
    OnlinePair<ComputeType> result = BlockAllReduceOnline<ComputeType, block_size>(acc);
    const ComputeType row_max = result.m;
    const ComputeType row_sum = result.s;

    // --- Pass 2: re-read, normalise, and store ---
    if (CACHE_OPT) {
      for (int pack_id = num_packs - 1 - tid; pack_id >= 0; pack_id -= block_size) {
        ComputeType pack[pack_size];
        load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
        for (int i = 0; i < pack_size; ++i) {
          if (algorithm == Algorithm::kSoftmax) {
            pack[i] = Div(Exp(pack[i] - row_max), row_sum);
          } else {
            pack[i] = (pack[i] - row_max) - Log(row_sum);
          }
        }
        store.template store<pack_size>(pack, row, pack_id * pack_size);
      }
    } else {
      for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
        ComputeType pack[pack_size];
        load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
        for (int i = 0; i < pack_size; ++i) {
          if (algorithm == Algorithm::kSoftmax) {
            pack[i] = Div(Exp(pack[i] - row_max), row_sum);
          } else {
            pack[i] = (pack[i] - row_max) - Log(row_sum);
          }
        }
        store.template store<pack_size>(pack, row, pack_id * pack_size);
      }
    }
  }
}

// ---- Strategy 3 dispatch ----------------------------------------------------

template<typename LOAD, typename STORE, typename ComputeType, int pack_size,
         Algorithm algorithm, bool CACHE_OPT>
inline cudaError_t LaunchSoftmaxBlockUncachedOnlineImpl(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols) {
  constexpr int block_size = 1024;
  constexpr int waves = 32;
  int grid_dim_x;
  cudaError_t err = GetNumBlocks(block_size, rows, waves, &grid_dim_x);
  if (err != cudaSuccess) return err;
  SoftmaxBlockUncachedOnlineImpl<LOAD, STORE, ComputeType, pack_size, block_size,
                                  algorithm, CACHE_OPT>
      <<<grid_dim_x, block_size, 0, stream>>>(load, store, rows, cols);
  return cudaPeekAtLastError();
}

template<typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm,
         bool CACHE_OPT>
struct DispatchSoftmaxBlockUncachedOnlinePackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store,
                         const int64_t rows, const int64_t cols) {
    if (cols % 2 == 0) {
      return LaunchSoftmaxBlockUncachedOnlineImpl<LOAD, STORE, ComputeType, 2,
                                                   algorithm, CACHE_OPT>(
          stream, load, store, rows, cols);
    } else {
      return LaunchSoftmaxBlockUncachedOnlineImpl<LOAD, STORE, ComputeType, 1,
                                                   algorithm, CACHE_OPT>(
          stream, load, store, rows, cols);
    }
  }
};

template<typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm,
         bool CACHE_OPT>
inline cudaError_t DispatchSoftmaxBlockUncachedOnline(cudaStream_t stream, LOAD load,
                                                      STORE store, const int64_t rows,
                                                      const int64_t cols) {
  return DispatchSoftmaxBlockUncachedOnlinePackSize<LOAD, STORE, ComputeType, algorithm,
                                                     CACHE_OPT>()(
      stream, load, store, rows, cols);
}

// =========================================================================
// Top-level dispatch
// =========================================================================

template<typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm,
         bool CACHE_OPT>
inline typename std::enable_if<!std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchSoftmax(cudaStream_t stream, LOAD load, STORE store,
                const int64_t rows, const int64_t cols) {
  if (cols < 1024) {
    return DispatchSoftmaxWarpImpl<LOAD, STORE, ComputeType, algorithm>(
        stream, load, store, rows, cols);
  } else {
    bool success;
    cudaError_t err = TryDispatchSoftmaxBlockSMemOnline<LOAD, STORE, ComputeType,
                                                         algorithm, CACHE_OPT>(
        stream, load, store, rows, cols, &success);
    if (err != cudaSuccess) return err;
    if (!success) {
      return DispatchSoftmaxBlockUncachedOnline<LOAD, STORE, ComputeType, algorithm,
                                                 CACHE_OPT>(
          stream, load, store, rows, cols);
    }
    return cudaSuccess;
  }
}

template<typename LOAD, typename STORE, typename ComputeType, Algorithm algorithm,
         bool CACHE_OPT>
inline typename std::enable_if<std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchSoftmax(cudaStream_t stream, LOAD load, STORE store,
                const int64_t rows, const int64_t cols) {
  return DispatchSoftmaxBlockUncachedOnline<LOAD, STORE, ComputeType, algorithm, CACHE_OPT>(
      stream, load, store, rows, cols);
}

// =========================================================================
// Convenience wrappers
// =========================================================================

template<typename T, bool CACHE_OPT = true>
cudaError_t LaunchSoftmax(cudaStream_t stream, const T* input, T* output,
                          int64_t rows, int64_t cols) {
  using ComputeType = typename DefaultComputeType<T>::type;
  DirectLoad<T, ComputeType> load(input, cols);
  DirectStore<ComputeType, T> store(output, cols);
  return DispatchSoftmax<decltype(load), decltype(store), ComputeType,
                         Algorithm::kSoftmax, CACHE_OPT>(
      stream, load, store, rows, cols);
}

template<typename T, bool CACHE_OPT = true>
cudaError_t LaunchLogSoftmax(cudaStream_t stream, const T* input, T* output,
                             int64_t rows, int64_t cols) {
  using ComputeType = typename DefaultComputeType<T>::type;
  DirectLoad<T, ComputeType> load(input, cols);
  DirectStore<ComputeType, T> store(output, cols);
  return DispatchSoftmax<decltype(load), decltype(store), ComputeType,
                         Algorithm::kLogSoftmax, CACHE_OPT>(
      stream, load, store, rows, cols);
}

} // namespace cuda_softmax
