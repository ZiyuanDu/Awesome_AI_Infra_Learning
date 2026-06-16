#pragma once
#include <cuda_runtime.h>

template <int BM = 128, int BN = 128, int BK = 8, int TM = 8, int TN = 8>
__global__ void sgemm_v2(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float* __restrict__ C,
                         int M, int N, int K) {
                          
  constexpr int FLOAT4      = 4;
  constexpr int BLOCK_DIM_X = BN / TN;
  constexpr int BLOCK_DIM_Y = BM / TM;
  constexpr int NUM_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y;
  constexpr int A_TPR       = BK / FLOAT4;            // 搬 sA 每行的 float4 数
  constexpr int B_TPR       = BN / FLOAT4;            // 搬 sB 每行的 float4 数
  constexpr int A_ROW_STRIDE = NUM_THREADS / A_TPR;   // 一趟覆盖的 A 行数
  constexpr int B_ROW_STRIDE = NUM_THREADS / B_TPR;   // 一趟覆盖的 B 行数


  __shared__ float sA[BM][BK];
  __shared__ float sB[BK][BN];

  const int bx = blockIdx.x, by = blockIdx.y;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;

  const int tr = threadIdx.y * TM;
  const int tc = threadIdx.x * TN;

  // 搬运身份：tid 在 tile 内的起始坐标
  const int load_a_row = tid / A_TPR;
  const int load_a_col = (tid % A_TPR) * FLOAT4;
  const int load_b_row = tid / B_TPR;
  const int load_b_col = (tid % B_TPR) * FLOAT4;

  const int gb_col = bx * BN + load_b_col;

  float accum[TM][TN] = {{0.f}};

  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    const int gk = bk * BK;

    // ---- 搬 A 片：沿行方向多趟覆盖 ----
    #pragma unroll
    for (int off = 0; off < BM; off += A_ROW_STRIDE) {
      const int a_row  = load_a_row + off;
      const int ga_row = by * BM + a_row;
      const int ga_col = gk + load_a_col;
      if (ga_row < M && ga_col < K)
        *(float4*)&sA[a_row][load_a_col] =
            *(const float4*)&A[ga_row * K + ga_col];
      else
        *(float4*)&sA[a_row][load_a_col] = make_float4(0, 0, 0, 0);
    }

    // ---- 搬 B 片：沿行方向多趟覆盖 ----
    #pragma unroll
    for (int off = 0; off < BK; off += B_ROW_STRIDE) {
      const int b_row  = load_b_row + off;
      const int gb_row = gk + b_row;
      if (gb_row < K && gb_col < N)
        *(float4*)&sB[b_row][load_b_col] =
            *(const float4*)&B[gb_row * N + gb_col];
      else
        *(float4*)&sB[b_row][load_b_col] = make_float4(0, 0, 0, 0);
    }
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < BK; ++k) {
      #pragma unroll
      for (int n = 0; n < TN; n += FLOAT4) {
        float4 bv = *(float4*)&sB[k][tc + n];
        #pragma unroll
        for (int m = 0; m < TM; ++m) {
          float av = sA[tr + m][k];
          accum[m][n + 0] += av * bv.x;
          accum[m][n + 1] += av * bv.y;
          accum[m][n + 2] += av * bv.z;
          accum[m][n + 3] += av * bv.w;
        }
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int m = 0; m < TM; ++m) {
    const int gc_row = by * BM + tr + m;
    if (gc_row >= M) continue;
    #pragma unroll
    for (int n = 0; n < TN; n += FLOAT4) {
      const int gc_col = bx * BN + tc + n;
      if (gc_col < N)
        *(float4*)&C[gc_row * N + gc_col] =
            make_float4(accum[m][n], accum[m][n + 1],
                        accum[m][n + 2], accum[m][n + 3]);
    }
  }
}

template <int BM = 128, int BN = 128, int BK = 16, int TM = 8, int TN = 8>
inline void launch_sgemm_v2(const float* A, const float* B, float* C,
                            int M, int N, int K) {
  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_v2<BM, BN, BK, TM, TN><<<grid, block>>>(A, B, C, M, N, K);
}
