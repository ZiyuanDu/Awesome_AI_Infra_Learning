#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <mma.h>

namespace gemm {

// global -> 寄存器(float4)：grid-stride 读 ROWS×COLS 的 tile，越界填 0
template <int ROWS, int COLS, int NT, int NF>
__device__ __forceinline__ void load_tile(
    const float* __restrict__ src, int ld, int row_base, int col_base,
    int bound_row, int bound_col, int tid, float4 (&reg)[NF]) {
  constexpr int F4 = ROWS * COLS / 4, PR = COLS / 4;
  #pragma unroll
  for (int i = 0; i < NF; ++i) {
    int idx = i * NT + tid;
    if (idx < F4) {
      int r = idx / PR, c = (idx % PR) * 4;
      int gr = row_base + r, gc = col_base + c;
      reg[i] = (gr < bound_row && gc < bound_col)
                   ? *(const float4*)&src[gr * ld + gc]
                   : make_float4(0, 0, 0, 0);
    }
  }
}

// 寄存器 -> shared：dst[r][c] = reg
template <int ROWS, int COLS, int NT, int NF, int LD>
__device__ __forceinline__ void store_tile(
    float (*dst)[LD], int tid, const float4 (&reg)[NF]) {
  constexpr int F4 = ROWS * COLS / 4, PR = COLS / 4;
  #pragma unroll
  for (int i = 0; i < NF; ++i) {
    int idx = i * NT + tid;
    if (idx < F4) {
      int r = idx / PR, c = (idx % PR) * 4;
      *(float4*)&dst[r][c] = reg[i];
    }
  }
}

// 寄存器 -> shared，转置写：sA[c][r] = A[r][c]，让计算端能 float4 读 A
template <int BM, int BK, int NT, int NF, int LD>
__device__ __forceinline__ void store_a_T(
    float (*sA)[LD], int tid, const float4 (&reg)[NF]) {
  constexpr int F4 = BM * BK / 4, PR = BK / 4;
  #pragma unroll
  for (int i = 0; i < NF; ++i) {
    int idx = i * NT + tid;
    if (idx < F4) {
      int r = idx / PR, c = (idx % PR) * 4;
      sA[c + 0][r] = reg[i].x; sA[c + 1][r] = reg[i].y;
      sA[c + 2][r] = reg[i].z; sA[c + 3][r] = reg[i].w;
    }
  }
}

// 寄存器 -> shared，B 直存
template <int BK, int BN, int NT, int NF, int LD>
__device__ __forceinline__ void store_b(
    float (*sB)[LD], int tid, const float4 (&reg)[NF]) {
  store_tile<BK, BN, NT, NF, LD>(sB, tid, reg);
}

// CUDA-core 路径：shared -> 寄存器 -> FMA
template <int BK, int ALD, int BLD, int TM, int TN>
__device__ __forceinline__ void compute_tile(
    const float (*sA)[ALD], const float (*sB)[BLD],
    int ty, int tx, float (&accum)[TM][TN]) {
  #pragma unroll
  for (int k = 0; k < BK; ++k) {
    float av[TM], bv[TN];
    #pragma unroll
    for (int m = 0; m < TM; m += 4)
      *(float4*)(av + m) = *(const float4*)&sA[k][ty * TM + m];
    #pragma unroll
    for (int n = 0; n < TN; n += 4)
      *(float4*)(bv + n) = *(const float4*)&sB[k][tx * TN + n];
    #pragma unroll
    for (int m = 0; m < TM; ++m)
      #pragma unroll
      for (int n = 0; n < TN; ++n)
        accum[m][n] = __fmaf_rn(av[m], bv[n], accum[m][n]);
  }
}

}  // namespace gemm
