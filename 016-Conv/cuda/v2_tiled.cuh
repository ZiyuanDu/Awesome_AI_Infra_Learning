#pragma once
#include <cuda_runtime.h>

// v2: Tiled convolution — K维度分块 + 寄存器累加
// 每个block处理 output 的一个 TILE_H×TILE_W 空间区域
// 在K维度分块(BK), 每个线程累加BK个output channel的部分和
// 通过寄存器复用减少weight的重复读取

template <int TH = 8, int TW = 8, int BK = 4>
__global__ void conv_v2(const float* __restrict__ input,
                         const float* __restrict__ weight,
                         float* __restrict__ output,
                         int N, int C, int H, int W, int K, int R, int S) {
  int OH = H - R + 1, OW = W - S + 1;
  int num_k_blocks = (K + BK - 1) / BK;

  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;
  int ow = bx * TW + tx, oh = by * TH + ty;

  // grid.z = num_k_blocks * N, flat over (n, k_block)
  int n = blockIdx.z / num_k_blocks;
  int k_start = (blockIdx.z % num_k_blocks) * BK;

  float sum[BK] = {0.f};

  if (ow < OW && oh < OH && n < N) {
    int in_base = n * C * H * W;
    #pragma unroll
    for (int c = 0; c < C; ++c) {
      int in_c = in_base + c * H * W;
      #pragma unroll
      for (int r = 0; r < R; ++r) {
        int in_r = in_c + (oh + r) * W;
        #pragma unroll
        for (int s = 0; s < S; ++s) {
          float iv = input[in_r + (ow + s)];
          #pragma unroll
          for (int ki = 0; ki < BK; ++ki) {
            int k = k_start + ki;
            if (k < K)
              sum[ki] += iv * weight[k * C * R * S + c * R * S + r * S + s];
          }
        }
      }
    }
  }

  #pragma unroll
  for (int ki = 0; ki < BK; ++ki) {
    int k = k_start + ki;
    if (k < K && ow < OW && oh < OH && n < N)
      output[n * K * OH * OW + k * OH * OW + oh * OW + ow] = sum[ki];
  }
}

template <int TH = 8, int TW = 8, int BK = 4>
inline void launch_conv_v2(const float* input, const float* weight, float* output,
                            int N, int C, int H, int W, int K, int R, int S) {
  int OH = H - R + 1, OW = W - S + 1;
  int num_k_blocks = (K + BK - 1) / BK;
  dim3 block(TW, TH);
  dim3 grid((OW + TW - 1) / TW, (OH + TH - 1) / TH, num_k_blocks * N);
  conv_v2<TH, TW, BK><<<grid, block>>>(input, weight, output, N, C, H, W, K, R, S);
}
