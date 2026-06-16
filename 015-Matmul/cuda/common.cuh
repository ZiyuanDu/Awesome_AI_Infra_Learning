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

// 寄存器 -> shared，直存（非转置）：dst[r][c] = reg
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

// 寄存器 -> shared，B 直存（v3 用）
template <int BK, int BN, int NT, int NF, int LD>
__device__ __forceinline__ void store_b(
    float (*sB)[LD], int tid, const float4 (&reg)[NF]) {
  store_tile<BK, BN, NT, NF, LD>(sB, tid, reg);
}

// global B -> shared via cp.async，越界用 src-size=0 零填充（不读源、不留垃圾）
template <int BK, int BN, int NT, int NF, int LD>
__device__ __forceinline__ void load_b_async(
    const float* __restrict__ B, int N, int k_base, int col_base, int K,
    int tid, float (*sB)[LD]) {
  constexpr int F4 = BK * BN / 4, PR = BN / 4;
  #pragma unroll
  for (int i = 0; i < NF; ++i) {
    int idx = i * NT + tid;
    if (idx < F4) {
      int r = idx / PR, c = (idx % PR) * 4;
      int gk = k_base + r, gc = col_base + c;
      bool ok = (gk < K && gc < N);
      const float* src = ok ? (B + gk * N + gc) : B;
      int bytes = ok ? 16 : 0;
      uint32_t dst = __cvta_generic_to_shared(&sB[r][c]);
      asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
                   ::"r"(dst), "l"(src), "r"(bytes));
    }
  }
}

// CUDA-core 路径：shared -> 寄存器 -> FMA（v3/v4 用）
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

// TensorCore 路径：从 shared 取 A/B fragment，累加一片 BK 进 c_frag（v5 用）
template <int WM, int WN, int WK, int WTM, int WTN, int BK, int SA, int SB>
__device__ __forceinline__ void wmma_compute_tile(
    const float* base_a, const float* base_b, int warp_m, int warp_n,
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WM, WN, WK, float>
        (&c_frag)[WTM][WTN]) {
  namespace w = nvcuda::wmma;
  #pragma unroll
  for (int wk = 0; wk < BK / WK; ++wk) {
    w::fragment<w::matrix_a, WM, WN, WK, w::precision::tf32, w::row_major> a_frag[WTM];
    #pragma unroll
    for (int i = 0; i < WTM; ++i) {
      int row = warp_m * WTM * WM + i * WM;
      w::load_matrix_sync(a_frag[i], base_a + row * SA + wk * WK, SA);
    }
    w::fragment<w::matrix_b, WM, WN, WK, w::precision::tf32, w::row_major> b_frag[WTN];
    #pragma unroll
    for (int j = 0; j < WTN; ++j) {
      int col = warp_n * WTN * WN + j * WN;
      w::load_matrix_sync(b_frag[j], base_b + wk * WK * SB + col, SB);
    }
    #pragma unroll
    for (int i = 0; i < WTM; ++i)
      #pragma unroll
      for (int j = 0; j < WTN; ++j)
        w::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
  }
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n" ::);
}

}  // namespace gemm
