#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace conv {

// global -> 寄存器(float4): grid-stride 读取 ROWS×COLS 的 tile, 越界填 0
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

// 寄存器 -> shared: dst[r][c] = reg
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

// 将输入窗口 (IH×IW) 加载到 shared memory, 每个线程加载一部分
// 用于卷积的 shared memory input tiling
template <int IH, int IW, int NT>
__device__ __forceinline__ void load_input_window(
    const float* __restrict__ input, int H, int W,
    int in_base_row, int in_base_col,
    float (*s_inp)[IW], int tid) {
  int n_elems = IH * IW;
  for (int i = tid; i < n_elems; i += NT) {
    int r = i / IW, c = i % IW;
    int gr = in_base_row + r, gc = in_base_col + c;
    s_inp[r][c] = (gr < H && gc < W) ? input[gr * W + gc] : 0.f;
  }
}

}  // namespace conv
