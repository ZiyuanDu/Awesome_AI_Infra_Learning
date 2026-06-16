#pragma once
#include <cuda_runtime.h>

template <int BM = 64, int BN = 64, int BK = 4, int TM = 16>
__global__ void sgemm_v1_1(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M, int N, int K) {
  __shared__ float sA[BM][BK], sB[BK][BN];

  const int bx = blockIdx.x, by = blockIdx.y;
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int tid = ty * BN + tx;

  const int col = bx * BN + tx;
  float accum[TM] = {0.f};

  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    const int gk = bk * BK;

    const int ar = tid / BK, ac = tid % BK;
    const int ga_row = by * BM + ar, ga_col = gk + ac;
    sA[ar][ac] = (ga_row < M && ga_col < K) ? A[ga_row * K + ga_col] : 0.f;

    const int br = tid / BN, bc = tid % BN;
    const int gb_row = gk + br, gb_col = bx * BN + bc;
    sB[br][bc] = (gb_row < K && gb_col < N) ? B[gb_row * N + bc + bx * BN] : 0.f;
    __syncthreads();

    
    #pragma unroll
    for (int k = 0; k < BK; ++k) {
      float b_val = sB[k][tx];
      #pragma unroll
      for (int m = 0; m < TM; ++m)
        accum[m] += sA[ty * TM + m][k] * b_val;
    }
    __syncthreads();
  }

  #pragma unroll
  for (int m = 0; m < TM; ++m) {
    int row = by * BM + ty * TM + m;
    if (row < M && col < N) C[row * N + col] = accum[m];
  }
}

template <int BM = 64, int BN = 64, int BK = 4, int TM = 16>
inline void launch_sgemm_v1_1(const float* A, const float* B, float* C,
                              int M, int N, int K) {
  dim3 block(BN, BM / TM);                             
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_v1_1<BM, BN, BK, TM><<<grid, block>>>(A, B, C, M, N, K);
}
