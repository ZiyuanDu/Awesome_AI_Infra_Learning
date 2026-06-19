#pragma once
#include <cuda_runtime.h>

template <int BM = 64, int BN = 64, int BK = 16, int TM = 8, int TN = 4>
__global__ void sgemm_v1_2(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M, int N, int K) {
  constexpr int NT = (BM / TM) * (BN / TN);
  __shared__ float SA[BM][BK];
  __shared__ float SB[BK][BN];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = ty * blockDim.x + tx;

  float sum[TM][TN] = {{0.f}};

  for (int tile = 0; tile < (K + BK - 1) / BK; ++tile) {
    int gk = tile * BK;

    #pragma unroll
    for (int i = tid; i < BM * BK; i += NT) {
      int load_a_row = i / BK;
      int load_a_col = i % BK;
      int ga_row = blockIdx.y * BM + load_a_row;
      int ga_col = gk + load_a_col;
      SA[load_a_row][load_a_col] = (ga_row < M && ga_col < K) ? A[ga_row * K + ga_col] : 0.f;
    }

    #pragma unroll
    for (int i = tid; i < BK * BN; i += NT) {
      int load_b_row = i / BN;
      int load_b_col = i % BN;
      int gb_row = gk + load_b_row;
      int gb_col = blockIdx.x * BN + load_b_col;
      SB[load_b_row][load_b_col] = (gb_row < K && gb_col < N) ? B[gb_row * N + gb_col] : 0.f;
    }
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < BK; ++k) {
      float a_reg[TM], b_reg[TN];
      #pragma unroll
      for (int m = 0; m < TM; ++m) a_reg[m] = SA[ty * TM + m][k];
      #pragma unroll
      for (int n = 0; n < TN; ++n) b_reg[n] = SB[k][tx * TN + n];
      #pragma unroll
      for (int m = 0; m < TM; ++m)
        #pragma unroll
        for (int n = 0; n < TN; ++n)
          sum[m][n] += a_reg[m] * b_reg[n];
    }
    __syncthreads();
  }

  #pragma unroll
  for (int m = 0; m < TM; ++m) {
    int row = blockIdx.y * BM + ty * TM + m;
    if (row >= M) continue;
    #pragma unroll
    for (int n = 0; n < TN; ++n) {
      int col = blockIdx.x * BN + tx * TN + n;
      if (col < N) C[row * N + col] = sum[m][n];
    }
  }
}

template <int BM = 64, int BN = 64, int BK = 16, int TM = 8, int TN = 4>
inline void launch_sgemm_v1_2(const float* A, const float* B, float* C,
                              int M, int N, int K) {
  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_v1_2<BM, BN, BK, TM, TN><<<grid, block>>>(A, B, C, M, N, K);
}
