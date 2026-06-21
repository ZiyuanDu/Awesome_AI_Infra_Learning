#pragma once
#include <cuda_runtime.h>

// v1: im2col + SGEMM — 将卷积降级为矩阵乘法 (当前仅支持 N=1)
// Step1: im2col 将 input[1][C][H][W] 展开为 col[CRS][OH*OW]
// Step2: SGEMM: output[K][OH*OW] = weight[K][CRS] * col[CRS][OH*OW]
// Note: sgemm输出layout为[K][OH*OW], 与v0的[1][K][OH][OW]在内存中一致(for N=1)

__global__ void im2col_kernel(const float* __restrict__ input,
                               float* __restrict__ col,
                               int N, int C, int H, int W, int R, int S) {
  int OH = H - R + 1, OW = W - S + 1, CRS = C * R * S;
  int out_spatial = N * OH * OW;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= out_spatial) return;

  int n = idx / (OH * OW), resid = idx % (OH * OW);
  int oh = resid / OW, ow = resid % OW;
  int in_base = n * C * H * W;

  #pragma unroll
  for (int crs = 0; crs < CRS; ++crs) {
    int c = crs / (R * S), rs = crs % (R * S);
    int r = rs / S, s = rs % S;
    col[crs * out_spatial + idx] =
        input[in_base + c * H * W + (oh + r) * W + (ow + s)];
  }
}

// 32×32 shared-memory tiled SGEMM: C[M][N] = A[M][K] * B[K][N]
// A = weight[K][CRS], B = col[CRS][out_spatial]
__global__ void sgemm_32x32(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int N, int K) {
  constexpr int BM = 32, BN = 32, BK = 32;
  __shared__ float sA[BM][BK], sB[BK][BN];

  int tx = threadIdx.x, ty = threadIdx.y;

  int row = blockIdx.y * BM + ty;
  int col = blockIdx.x * BN + tx;

  float sum = 0.f;

  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    int gk = bk * BK;

    // 协作加载 A[ty][gk+tx] 和 B[gk+ty][tx]
    int ak = gk + tx;
    sA[ty][tx] = (row < M && ak < K) ? A[row * K + ak] : 0.f;

    int bk_row = gk + ty;
    sB[ty][tx] = (bk_row < K && col < N) ? B[bk_row * N + col] : 0.f;
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < BK; ++k)
      sum += sA[ty][k] * sB[k][tx];
    __syncthreads();
  }

  if (row < M && col < N)
    C[row * N + col] = sum;
}

inline void launch_conv_v1(const float* input, const float* weight, float* output,
                            float* d_col, int N, int C, int H, int W, int K, int R, int S) {
  int OH = H - R + 1, OW = W - S + 1, CRS = C * R * S;
  int out_spatial = N * OH * OW;

  // Step 1: im2col
  im2col_kernel<<<(out_spatial + 255) / 256, 256>>>(
      input, d_col, N, C, H, W, R, S);

  // Step 2: SGEMM: output[K][out_spatial] = weight[K][CRS] * col[CRS][out_spatial]
  dim3 block(32, 32);
  dim3 grid((out_spatial + 31) / 32, (K + 31) / 32);
  sgemm_32x32<<<grid, block>>>(weight, d_col, output, K, out_spatial, CRS);
}
