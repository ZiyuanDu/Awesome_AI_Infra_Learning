#pragma once
#include <cuda_runtime.h>
#include <mma.h>
#include "common.cuh"

template <int BM = 128, int BN = 128, int BK = 8, int K_STAGE = 2>
__global__ void __launch_bounds__(256, 2)
sgemm_v5(const float* __restrict__ A,
         const float* __restrict__ B,
         float* __restrict__ C,
         int M, int N, int K) {
  using namespace gemm;
  constexpr int WM = 16, WN = 16, WK = 8;
  constexpr int WARP_M = 2, WARP_N = 4;
  constexpr int WTM = BM / (WARP_M * WM);
  constexpr int WTN = BN / (WARP_N * WN);
  constexpr int SA = BK, SB = BN;
  constexpr int NT = 256;
  constexpr int FA = (BM * BK / 4 + NT - 1) / NT;
  constexpr int FB = (BK * BN / 4 + NT - 1) / NT;

  extern __shared__ float smem[];
  auto sA = reinterpret_cast<float (*)[SA]>(smem);
  auto sB = reinterpret_cast<float (*)[SB]>(smem + K_STAGE * BM * SA);

  // grid 重排（GROUP_M 提升 L2 命中）
  constexpr int GROUP_M = 8;
  int pid = blockIdx.y * gridDim.x + blockIdx.x;
  int num_pid_n = gridDim.x;
  int group_id = pid / (GROUP_M * num_pid_n);
  int pid_in_group = pid % (GROUP_M * num_pid_n);
  int bx = pid_in_group % num_pid_n;
  int by = group_id * GROUP_M + pid_in_group / num_pid_n;

  int tid = threadIdx.x;
  int warp_id = tid / warpSize;
  int warp_m = warp_id / WARP_N, warp_n = warp_id % WARP_N;

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WM, WN, WK, float>
      c_frag[WTM][WTN];
  #pragma unroll
  for (int i = 0; i < WTM; ++i)
    #pragma unroll
    for (int j = 0; j < WTN; ++j)
      nvcuda::wmma::fill_fragment(c_frag[i][j], 0.0f);

  int tiles = (K + BK - 1) / BK;
  float4 rA[FA];

  auto load_stage = [&](int gk, int sa_off, int sb_off) {
    load_tile<BM, BK, NT, FA>(A, K, by * BM, gk, M, K, tid, rA);
    load_b_async<BK, BN, NT, FB, SB>(B, N, gk, bx * BN, K, tid, sB + sb_off);
    cp_async_commit();
    store_tile<BM, BK, NT, FA, SA>(sA + sa_off, tid, rA);  // A 立即落 shared
  };

  // prologue：灌满前 K_STAGE-1 个 stage
  #pragma unroll
  for (int bk = 0; bk < K_STAGE - 1 && bk < tiles; ++bk)
    load_stage(bk * BK, bk * BM, bk * BK);

  // 主循环：预取下一片 → 等就绪 → 算当前片
  for (int bk = K_STAGE - 1; bk < tiles; ++bk) {
    int ls = bk % K_STAGE;
    int cs = (bk - (K_STAGE - 1)) % K_STAGE;
    load_stage(bk * BK, ls * BM, ls * BK);

    cp_async_wait<K_STAGE - 2>();
    __syncthreads();
    wmma_compute_tile<WM, WN, WK, WTM, WTN, BK, SA, SB>(
        smem + cs * BM * SA, smem + K_STAGE * BM * SA + cs * BK * SB,
        warp_m, warp_n, c_frag);
    __syncthreads();
  }

  // drain：算流水线里剩余的 stage
  cp_async_wait_all();
  __syncthreads();
  for (int t = tiles > K_STAGE - 1 ? tiles - K_STAGE + 1 : 0; t < tiles; ++t) {
    int cs = t % K_STAGE;
    wmma_compute_tile<WM, WN, WK, WTM, WTN, BK, SA, SB>(
        smem + cs * BM * SA, smem + K_STAGE * BM * SA + cs * BK * SB,
        warp_m, warp_n, c_frag);
  }

  // 写回 C
  #pragma unroll
  for (int i = 0; i < WTM; ++i)
    #pragma unroll
    for (int j = 0; j < WTN; ++j) {
      int row = by * BM + warp_m * WTM * WM + i * WM;
      int col = bx * BN + warp_n * WTN * WN + j * WN;
      if (row < M && col < N)
        nvcuda::wmma::store_matrix_sync(
            C + row * N + col, c_frag[i][j], N, nvcuda::wmma::mem_row_major);
    }
}

inline void launch_sgemm_v5(const float* A, const float* B, float* C,
                            int M, int N, int K) {
  constexpr int BM = 128, BN = 128, BK = 8, K_STAGE = 2;
  constexpr size_t smem_sz = K_STAGE * (BM * BK + BK * BN) * sizeof(float);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  cudaFuncSetAttribute(sgemm_v5<BM, BN, BK, K_STAGE>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem_sz);
  sgemm_v5<BM, BN, BK, K_STAGE><<<grid, 256, smem_sz>>>(A, B, C, M, N, K);
}
