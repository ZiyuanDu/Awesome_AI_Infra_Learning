#pragma once
#include <cuda_runtime.h>

__global__ void sgemm_v1(const float* A, const float* B, float* C, int M, int N, int K) {
  constexpr int BM = 32, BN = 32, BK = 32;
  __shared__ float sA[BM][BK], sB[BK][BN];

  int ty = threadIdx.y;
  int tx = threadIdx.x;
  
  // 以输出矩阵C为视角
  int row = blockIdx.y * BM + ty;
  int col = blockIdx.x * BN + tx;
  
  float sum = 0.f;

  // A和B分别沿着K滑动，每次滑动长度为BK
  for (int k_tile = 0; k_tile < K; k_tile += BK) {
    // 从HBM放到Shared Memory中
    int ga = k_tile + tx;
    sA[ty][tx] = (row < M && ga < K) ? A[row * K + ga] : 0.f;

    int gb = k_tile + ty;
    sB[ty][tx] = (gb < K && col < N) ? B[gb * N + col] : 0.f;
    
    __syncthreads();
    // 从Shared memory中取值计算
    for (int k = 0; k < BK; ++k) {
      sum += sA[ty][k] * sB[k][tx];
    }
    __syncthreads();
  }
  // 写回
  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

inline void launch_sgemm_v1(const float* A, const float* B, float* C, int M, int N, int K) {
  dim3 block(32, 32);
  dim3 grid((N + 31) / 32, (M + 31) / 32);
  sgemm_v1<<<grid, block>>>(A, B, C, M, N, K);
}