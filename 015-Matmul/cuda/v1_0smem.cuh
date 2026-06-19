#pragma once
#include <cuda_runtime.h>
__global__ void sgemm_v1(const float* A, const float* B, float* C, int M, int N, int K) {
  constexpr int BM = 32, BN = 32, BK = 32;
  __shared__ float SA[BM][BK], SB[BK][BN];

  // 单个Block内部的偏移量
  int ty = threadIdx.y;
  int tx = threadIdx.x;

  // C[row][col]
  int row = blockIdx.y * BM + ty;
  int col = blockIdx.x * BN + tx;

  float sum = 0.0f;

  for (int tile = 0; tile < K; tile += BK) {
    int ga = tile + tx;
    SA[ty][tx] = (row < M && ga < K) ? A[row * K + ga] : 0.0f;
    int gb = tile + ty;
    SB[ty][tx] = (gb < K && col < N) ? B[gb * N + col] : 0.0f;
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < BK; k++) {
      sum += SA[ty][k] * SB[k][tx];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }

}


inline void launch_sgemm_v1(const float* A, const float* B, float* C, int M, int N, int K) {
  dim3 block(32, 32);
  dim3 grid((N + 31) / 32, (M + 31) / 32);
  sgemm_v1<<<grid, block>>>(A, B, C, M, N, K);
}