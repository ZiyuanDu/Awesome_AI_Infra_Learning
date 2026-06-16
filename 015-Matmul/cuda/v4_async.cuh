#pragma once
#include <cuda_runtime.h>
#include "common.cuh"

template <int BM = 128, int BN = 128, int BK = 16,
          int TM = 8, int TN = 8, int OFFSET = 4>
__global__ void __launch_bounds__(256, 2)
sgemm_v4(const float* __restrict__ A,
         const float* __restrict__ B,
         float* __restrict__ C,
         int M, int N, int K) {
  using namespace gemm;
  constexpr int NT  = (BM / TM) * (BN / TN);
  constexpr int FA  = (BM * BK / 4 + NT - 1) / NT;
  constexpr int FB  = (BK * BN / 4 + NT - 1) / NT;
  constexpr int ALD = BM + OFFSET, BLD = BN + OFFSET;

  static_assert(BM % TM == 0 && BN % TN == 0, "block tile 必须整除 thread tile");
  static_assert(BK % 4 == 0 && BN % 4 == 0, "BK/BN 需被 4 整除以用 float4 / cp.async");
  static_assert(TM % 4 == 0 && TN % 4 == 0, "TM/TN 需被 4 整除以 float4 读 smem");

  __shared__ float sA[2][BK][ALD];   // A 转置：sA[buf][k][m]
  __shared__ float sB[2][BK][BLD];

  const int bx = blockIdx.x, by = blockIdx.y;
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;
  const int tiles = (K + BK - 1) / BK;

  float4 rA[FA];
  float accum[TM][TN] = {{0.f}};
  int wb = 0, rb = 0;

  // prologue：搬入 tile 0（A 走寄存器转置，B 走 cp.async）
  load_tile<BM, BK, NT, FA>(A, K, by * BM, 0, M, K, tid, rA);
  store_a_T<BM, BK, NT, FA, ALD>(sA[0], tid, rA);
  load_b_async<BK, BN, NT, FB, BLD>(B, N, 0, bx * BN, K, tid, sB[0]);
  cp_async_commit();
  cp_async_wait_all();
  __syncthreads();

  // 主循环：B 异步预取与计算 overlap
  for (int bk = 1; bk < tiles; ++bk) {
    wb ^= 1;
    load_tile<BM, BK, NT, FA>(A, K, by * BM, bk * BK, M, K, tid, rA);
    load_b_async<BK, BN, NT, FB, BLD>(B, N, bk * BK, bx * BN, K, tid, sB[wb]);
    cp_async_commit();

    compute_tile<BK, ALD, BLD, TM, TN>(sA[rb], sB[rb], ty, tx, accum);

    store_a_T<BM, BK, NT, FA, ALD>(sA[wb], tid, rA);
    cp_async_wait_all();
    __syncthreads();
    rb ^= 1;
  }

  // epilogue：算最后一片
  compute_tile<BK, ALD, BLD, TM, TN>(sA[rb], sB[rb], ty, tx, accum);

  // 写回 C
  #pragma unroll
  for (int m = 0; m < TM; ++m) {
    int gc_row = by * BM + ty * TM + m;
    if (gc_row >= M) continue;
    #pragma unroll
    for (int n = 0; n < TN; n += 4) {
      int gc_col = bx * BN + tx * TN + n;
      if (gc_col < N)
        *(float4*)&C[gc_row * N + gc_col] =
            make_float4(accum[m][n], accum[m][n + 1],
                        accum[m][n + 2], accum[m][n + 3]);
    }
  }
}

template <int BM = 128, int BN = 128, int BK = 16,
          int TM = 8, int TN = 8, int OFFSET = 4>
inline void launch_sgemm_v4(const float* A, const float* B, float* C,
                            int M, int N, int K) {
  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_v4<BM, BN, BK, TM, TN, OFFSET><<<grid, block>>>(A, B, C, M, N, K);
}
