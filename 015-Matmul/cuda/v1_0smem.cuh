#pragma once
#include <cuda_runtime.h>

__global__ void sgemm_v1(const float* A, const float* B, float* C, int M, int N, int K) {
  constexpr int BM = 32, BN = 32, BK = 32;
  __shared__ float sA[BM][BK], sB[BK][BN];

  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = by * BM + ty, col = bx * BN + tx;
  float sum = 0.f;

  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    // 存储到Shared memory
    int gk = bk * BK;
    sA[ty][tx] = (row < M && gk + tx < K) ? A[row * K + (gk + tx)] : 0.f;
    sB[ty][tx] = (gk + ty < K && col < N) ? B[(gk + ty) * N + col] : 0.f;
    __syncthreads();
    
    // 计算方式同v0
    for (int k = 0; k < BK; ++k)
      sum += sA[ty][k] * sB[k][tx];
    __syncthreads();
  }
  // 写回
  if (row < M && col < N) C[row * N + col] = sum;
}

inline void launch_sgemm_v1(const float* A, const float* B, float* C, int M, int N, int K) {
  dim3 block(32, 32);
  dim3 grid((N + 31) / 32, (M + 31) / 32);
  sgemm_v1<<<grid, block>>>(A, B, C, M, N, K);
}
