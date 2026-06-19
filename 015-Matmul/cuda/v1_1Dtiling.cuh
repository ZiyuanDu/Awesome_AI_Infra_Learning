#pragma once
#include <cuda_runtime.h>

template <int BM = 64, int BN = 64, int BK = 16, int TM = 16>
__global__ void sgemm_v1_1(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M, int N, int K) {
                            
  constexpr int NT = (BM / TM) * BN;
  __shared__ float SA[BM][BK];
  __shared__ float SB[BK][BN];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int col = blockIdx.x * BN + tx;

  float sum[TM] = {0.0f};

  int tid = ty * blockDim.x + tx;

  for (int tile = 0; tile < K; tile += BK) {

    #pragma unroll
    for (int i = tid; i < BM * BK; i += NT) {
      int load_a_row = i / BK;
      int load_a_col = i % BK;
      int ga_row = blockIdx.y * BM + load_a_row;
      int ga_col = tile + load_a_col;
      SA[load_a_row][load_a_col] = (ga_row < M && ga_col < K) ? A[ga_row * K + ga_col] : 0.0f;
    }

    #pragma unroll
    for (int i = tid; i < BK * BN; i += NT) {
      int load_b_row = i / BN;
      int load_b_col = i % BN;
      int gb_row = tile + load_b_row;
      int gb_col = blockIdx.x * BN + load_b_col;
      SB[load_b_row][load_b_col] = (gb_row < K && gb_col < N) ? B[gb_row * N + gb_col] : 0.0f;
    }

    __syncthreads();

    
    #pragma unroll
    for (int k = 0; k < BK; k++) {
      float b_val = SB[k][tx];
      
      #pragma unroll
      for (int m = 0; m < TM; m++) {
        sum[m] += SA[ty * TM + m][k] * b_val;
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int m = 0; m < TM; m++) {
    int row = blockIdx.y * BM + ty * TM + m;
    
    if (row < M && col < N) {
      C[row * N + col] = sum[m];
    }
  }
}

template <int BM = 64, int BN = 64, int BK = 16, int TM = 16>
inline void launch_sgemm_v1_1(const float* A, const float* B, float* C,
                              int M, int N, int K) {
  dim3 block(BN, BM / TM);                             
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_v1_1<BM, BN, BK, TM><<<grid, block>>>(A, B, C, M, N, K);
}