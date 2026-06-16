#pragma once
#include <cuda_runtime.h>

template <int BM = 128, int BN = 128, int BK = 8, int TM = 8, int TN = 8>
__global__ void sgemm_v1_2(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M, int N, int K) {
  constexpr int NT = (BM / TM) * (BN / TN);
  __shared__ float sA[BM][BK], sB[BK][BN];

  const int bx = blockIdx.x, by = blockIdx.y;
  const int tx = threadIdx.x, ty = threadIdx.y; 
  const int tid = ty * (BN / TN) + tx;

  float accum[TM][TN] = {{0.f}};

  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    const int gk = bk * BK;

    #pragma unroll
    for (int i = tid; i < BM * BK; i += NT) {
      int r = i / BK, c = i % BK;
      int gr = by * BM + r, gc = gk + c;
      sA[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.f;
    }
    #pragma unroll
    for (int i = tid; i < BK * BN; i += NT) {
      int r = i / BN, c = i % BN;
      int gr = gk + r, gc = bx * BN + c;
      sB[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.f;
    }
    __syncthreads();

    // 内层外积：先把 sA/sB 读进寄存器，再做 TM*TN 次 FFMA
    #pragma unroll
    for (int k = 0; k < BK; ++k) {
      float a_reg[TM], b_reg[TN];
      #pragma unroll
      for (int m = 0; m < TM; ++m) a_reg[m] = sA[ty * TM + m][k];
      #pragma unroll
      for (int n = 0; n < TN; ++n) b_reg[n] = sB[k][tx * TN + n];
      #pragma unroll
      for (int m = 0; m < TM; ++m)
        #pragma unroll
        for (int n = 0; n < TN; ++n)
          accum[m][n] += a_reg[m] * b_reg[n];
    }
    __syncthreads();
  }

  // 写回 TM×TN 块
  #pragma unroll
  for (int m = 0; m < TM; ++m) {
    int row = by * BM + ty * TM + m;
    if (row >= M) continue;
    #pragma unroll
    for (int n = 0; n < TN; ++n) {
      int col = bx * BN + tx * TN + n;
      if (col < N) C[row * N + col] = accum[m][n];
    }
  }
}

template <int BM = 128, int BN = 128, int BK = 16, int TM = 8, int TN = 4>
inline void launch_sgemm_v1_2(const float* A, const float* B, float* C,
                              int M, int N, int K) {
  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_v1_2<BM, BN, BK, TM, TN><<<grid, block>>>(A, B, C, M, N, K);
}
