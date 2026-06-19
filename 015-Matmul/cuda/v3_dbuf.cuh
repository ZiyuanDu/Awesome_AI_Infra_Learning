#pragma once
#include <cuda_runtime.h>
#include "common.cuh"


template <int BM = 64, int BN = 64, int BK = 16,
          int TM = 8, int TN = 4, int OFFSET = 0>
__global__ void sgemm_v3(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float* __restrict__ C,
                         int M, int N, int K) {
  using namespace gemm;

  static_assert(BM % TM == 0 && BN % TN == 0, "BM/TM, BN/TN must be integer");
  static_assert(BK % 4 == 0 && BN % 4 == 0, "BK/BN must be multiple of 4");
  static_assert(TM % 4 == 0 && TN % 4 == 0, "TM/TN must be multiple of 4");

  constexpr int NT = (BM / TM) * (BN / TN);

  constexpr int ALD = BM + OFFSET;
  constexpr int BLD = BN + OFFSET;
  __shared__ float SA[2][BK][ALD];
  __shared__ float SB[2][BK][BLD];

  constexpr int FA = (BM * BK / 4 + NT - 1) / NT;
  constexpr int FB = (BK * BN / 4 + NT - 1) / NT;
  float4 rA[FA], rB[FB];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = ty * blockDim.x + tx;

  float sum[TM][TN] = {{0.f}};

  int wb = 0, rb = 0;
  const int tiles = (K + BK - 1) / BK;

  load_tile<BM, BK, NT, FA>(A, K, blockIdx.y * BM, 0, M, K, tid, rA);
  load_tile<BK, BN, NT, FB>(B, N, 0, blockIdx.x * BN, K, N, tid, rB);
  store_a_T<BM, BK, NT, FA, ALD>(SA[0], tid, rA);        // [NEW 2] 转置写
  store_b  <BK, BN, NT, FB, BLD>(SB[0], tid, rB);
  __syncthreads();

  for (int tile = 1; tile < tiles; ++tile) {
    int gk = tile * BK;
    wb ^= 1;

    // Step 1: 预取下一 tile 到寄存器
    load_tile<BM, BK, NT, FA>(A, K, blockIdx.y * BM, gk, M, K, tid, rA);
    load_tile<BK, BN, NT, FB>(B, N, gk, blockIdx.x * BN, K, N, tid, rB);

    // Step 2: 计算当前缓冲 rb（与 Step 1 的 global load 重叠）
    //         compute_tile 内 A/B 均用 float4 读取
    compute_tile<BK, ALD, BLD, TM, TN>(SA[rb], SB[rb], ty, tx, sum);

    // Step 3: 寄存器 → Shared Memory wb
    store_a_T<BM, BK, NT, FA, ALD>(SA[wb], tid, rA);     // [NEW 2]
    store_b  <BK, BN, NT, FB, BLD>(SB[wb], tid, rB);
    __syncthreads();
    rb ^= 1;
  }

  // Epilogue: 计算最后一片
  compute_tile<BK, ALD, BLD, TM, TN>(SA[rb], SB[rb], ty, tx, sum);

  #pragma unroll
  for (int m = 0; m < TM; ++m) {
    int row = blockIdx.y * BM + ty * TM + m;
    if (row >= M) continue;
    #pragma unroll
    for (int n = 0; n < TN; n += 4) {
      int col = blockIdx.x * BN + tx * TN + n;
      if (col < N)
        *(float4*)&C[row * N + col] =
            make_float4(sum[m][n], sum[m][n + 1],
                        sum[m][n + 2], sum[m][n + 3]);
    }
  }
}

template <int BM = 64, int BN = 64, int BK = 16,
          int TM = 8, int TN = 4, int OFFSET = 0>
inline void launch_sgemm_v3(const float* A, const float* B, float* C,
                            int M, int N, int K) {
  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_v3<BM, BN, BK, TM, TN, OFFSET><<<grid, block>>>(A, B, C, M, N, K);
}
