#pragma once
#include <cuda_runtime.h>

__global__ void sgemm_v0(const float* A, const float* B, float* C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < N) {
    float sum = 0.f;
    for (int k = 0; k < K; ++k)
      sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
  }
}

inline void launch_sgemm_v0(const float* A, const float* B, float* C, int M, int N, int K) {
  dim3 block(16, 16);
  dim3 grid((N + 15) / 16, (M + 15) / 16);
  sgemm_v0<<<grid, block>>>(A, B, C, M, N, K);
}
