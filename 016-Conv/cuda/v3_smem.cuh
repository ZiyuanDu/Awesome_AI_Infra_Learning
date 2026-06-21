#pragma once
#include <cuda_runtime.h>

// v3: Shared memory input tiling + per-channel iteration
// 将输入窗口(TH+R-1)×(TW+S-1)加载到shared memory, block内线程共享input数据
// 逐个channel加载、计算，避免过大的shared memory开销
// 每个线程累加TM个output channel (2D thread tiling in K dimension)

template <int TH = 8, int TW = 8, int BK = 8, int TM = 4>
__global__ void conv_v3(const float* __restrict__ input,
                         const float* __restrict__ weight,
                         float* __restrict__ output,
                         int N, int C, int H, int W, int K, int R, int S) {
  constexpr int IH = TH + R - 1;  // input window height
  constexpr int IW = TW + S - 1;  // input window width
  constexpr int NT = TH * TW;     // 线程数

  __shared__ float s_inp[IH][IW];

  int OH = H - R + 1, OW = W - S + 1;
  int num_k_blocks = (K + BK - 1) / BK;

  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;
  int tid = ty * blockDim.x + tx;

  int ow = bx * TW + tx;
  int oh = by * TH + ty;

  // grid.z = num_k_blocks * N
  int n = blockIdx.z / num_k_blocks;
  int k_start = (blockIdx.z % num_k_blocks) * BK;

  // 每个线程为TM个output channel维护累加寄存器
  float sum[TM] = {0.f};

  // 输入窗口在global memory中的起始位置
  int in_base_row = by * TH;
  int in_base_col = bx * TW;

  // 逐个channel加载input窗口到shared memory
  for (int c = 0; c < C; ++c) {
    // 协作加载input[c][in_base_row:in_base_row+IH][in_base_col:in_base_col+IW] -> s_inp
    int n_elems = IH * IW;
    for (int i = tid; i < n_elems; i += NT) {
      int iy = i / IW;
      int ix = i % IW;
      int giy = in_base_row + iy;
      int gix = in_base_col + ix;
      if (giy < H && gix < W && n < N)
        s_inp[iy][ix] = input[n * C * H * W + c * H * W + giy * W + gix];
      else
        s_inp[iy][ix] = 0.f;
    }
    __syncthreads();

    // 使用缓存的input数据，对R×S×TM做累加
    if (ow < OW && oh < OH && n < N) {
      #pragma unroll
      for (int r = 0; r < R; ++r) {
        #pragma unroll
        for (int s = 0; s < S; ++s) {
          float iv = s_inp[ty + r][tx + s];
          #pragma unroll
          for (int ki = 0; ki < TM; ++ki) {
            int k = k_start + ki;
            if (k < K)
              sum[ki] += iv * weight[k * C * R * S + c * R * S + r * S + s];
          }
        }
      }
    }
    __syncthreads();
  }

  // 写回结果
  #pragma unroll
  for (int ki = 0; ki < TM; ++ki) {
    int k = k_start + ki;
    if (k < K && ow < OW && oh < OH && n < N)
      output[n * K * OH * OW + k * OH * OW + oh * OW + ow] = sum[ki];
  }
}

template <int TH = 8, int TW = 8, int BK = 8, int TM = 4>
inline void launch_conv_v3(const float* input, const float* weight, float* output,
                            int N, int C, int H, int W, int K, int R, int S) {
  static_assert(TM <= BK, "TM must be <= BK");
  int OH = H - R + 1, OW = W - S + 1;
  int num_k_blocks = (K + BK - 1) / BK;
  dim3 block(TW, TH);
  dim3 grid((OW + TW - 1) / TW, (OH + TH - 1) / TH, num_k_blocks * N);
  conv_v3<TH, TW, BK, TM><<<grid, block>>>(input, weight, output, N, C, H, W, K, R, S);
}
