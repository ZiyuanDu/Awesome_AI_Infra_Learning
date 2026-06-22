#pragma once
#include <cuda_runtime.h>

// v1: im2col + optimized SGEMM — 将卷积降级为矩阵乘法 (仅支持N=1)
// Step1: im2col: input[1][C][H][W] -> col[CRS][OH*OW]
// Step2: SGEMM:  output[K][OH*OW] = weight[K][CRS] * col[CRS][OH*OW]
//         使用 float4 向量化 + 64×64 shared memory tiling + warp tiling

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

// Optimized SGEMM: float4 loads + 64×64 tiles + warp tiling (TM=8,TN=4)
// C[M][N] = A[M][K] * B[K][N]
template <int BM = 64, int BN = 64, int BK = 16, int TM = 8, int TN = 4>
__global__ void sgemm_tiled(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int N, int K) {
  static_assert(BK % 4 == 0 && BN % 4 == 0 && TN % 4 == 0,
                "BK/BN/TN must be multiple of 4 for float4");

  constexpr int NT = (BM / TM) * (BN / TN);
  __shared__ float sA[BM][BK], sB[BK][BN];

  int tx = threadIdx.x, ty = threadIdx.y;
  int tid = ty * blockDim.x + tx;

  float sum[TM][TN] = {{0.f}};

  for (int tile = 0; tile < (K + BK - 1) / BK; ++tile) {
    int gk = tile * BK;

    // float4 协作加载 A[BM][BK]
    #pragma unroll
    for (int i = tid; i < BM * BK / 4; i += NT) {
      int r = i / (BK / 4), c = (i % (BK / 4)) * 4;
      int gr = blockIdx.y * BM + r, gc = gk + c;
      *(float4*)&sA[r][c] = (gr < M && gc < K)
          ? *(const float4*)&A[gr * K + gc]
          : make_float4(0, 0, 0, 0);
    }

    // float4 协作加载 B[BK][BN]
    #pragma unroll
    for (int i = tid; i < BK * BN / 4; i += NT) {
      int r = i / (BN / 4), c = (i % (BN / 4)) * 4;
      int gr = gk + r, gc = blockIdx.x * BN + c;
      *(float4*)&sB[r][c] = (gr < K && gc < N)
          ? *(const float4*)&B[gr * N + gc]
          : make_float4(0, 0, 0, 0);
    }
    __syncthreads();

    // warp tiling: TM×TN per thread, float4读B
    #pragma unroll
    for (int k = 0; k < BK; ++k) {
      #pragma unroll
      for (int n = 0; n < TN; n += 4) {
        float4 bv = *(float4*)&sB[k][tx * TN + n];
        #pragma unroll
        for (int m = 0; m < TM; ++m) {
          float av = sA[ty * TM + m][k];
          sum[m][n + 0] += av * bv.x;
          sum[m][n + 1] += av * bv.y;
          sum[m][n + 2] += av * bv.z;
          sum[m][n + 3] += av * bv.w;
        }
      }
    }
    __syncthreads();
  }

  // float4 写回 C[BM][BN]
  #pragma unroll
  for (int m = 0; m < TM; ++m) {
    int row = blockIdx.y * BM + ty * TM + m;
    if (row >= M) continue;
    #pragma unroll
    for (int n = 0; n < TN; n += 4) {
      int col = blockIdx.x * BN + tx * TN + n;
      if (col < N)
        *(float4*)&C[row * N + col] =
            make_float4(sum[m][n], sum[m][n + 1], sum[m][n + 2], sum[m][n + 3]);
    }
  }
}

template <int BM = 64, int BN = 64, int BK = 16, int TM = 8, int TN = 4>
inline void launch_sgemm_tiled(const float* A, const float* B, float* C,
                                int M, int N, int K) {
  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_tiled<BM, BN, BK, TM, TN><<<grid, block>>>(A, B, C, M, N, K);
}

// ---- Basic 32×32 SGEMM (fallback when CRS mod 4 ≠ 0) ----
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
    int ak = gk + tx, bk_row = gk + ty;
    sA[ty][tx] = (row < M && ak < K) ? A[row * K + ak] : 0.f;
    sB[ty][tx] = (bk_row < K && col < N) ? B[bk_row * N + col] : 0.f;
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < BK; ++k) sum += sA[ty][k] * sB[k][tx];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = sum;
}

// ---- 组装: im2col + SGEMM ----
// CRS为4的倍数时用optimized SGEMM (float4+tiling), 否则fallback到32×32
inline void launch_conv_v1(const float* input, const float* weight, float* output,
                            float* d_col, int N, int C, int H, int W, int K, int R, int S) {
  int OH = H - R + 1, OW = W - S + 1, CRS = C * R * S;
  int out_spatial = N * OH * OW;

  im2col_kernel<<<(out_spatial + 255) / 256, 256>>>(
      input, d_col, N, C, H, W, R, S);

  if (CRS % 4 == 0) {
    // 快速路径: CRS对齐, 使用float4优化SGEMM
    launch_sgemm_tiled(weight, d_col, output, K, out_spatial, CRS);
  } else {
    // CRS非4倍数时float4会misaligned, fallback到标量SGEMM
    dim3 block(32, 32);
    dim3 grid((out_spatial + 31) / 32, (K + 31) / 32);
    sgemm_32x32<<<grid, block>>>(weight, d_col, output, K, out_spatial, CRS);
  }
}
