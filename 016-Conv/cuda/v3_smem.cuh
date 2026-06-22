#pragma once
#include <cuda_runtime.h>

// v3: BC-channel tiled direct convolution — shared memory for input + weight
// 核心思路: 同时缓存BC个输入channel + BK×BC×R×S的weight到shared memory
//           sync次数从2*C降到2*ceil(C/BC), 实现10-20x sync减少
//           所有inner loop数据来自shared memory, 无global memory访问
// R,S作为template参数: 编译器可优化2D shared memory索引

template <int TH = 8, int TW = 8, int BK = 16, int BC = 16, int R = 3, int S = 3>
__global__ void conv_v3(const float* __restrict__ input,
                         const float* __restrict__ weight,
                         float* __restrict__ output,
                         int N, int C, int H, int W, int K) {
  constexpr int IH = TH + R - 1;       // input window height
  constexpr int IW = TW + S - 1;       // input window width
  constexpr int CRS = R * S;
  constexpr int NT = TH * TW;           // threads per block

  // 双缓冲shared memory: input[BC][IH][IW] + weight[BK][BC][R][S]
  __shared__ float s_inp[BC][IH][IW];
  __shared__ float s_wt[BK][BC][R][S];

  int OH = H - R + 1, OW = W - S + 1;
  int num_k_blocks = (K + BK - 1) / BK;

  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;
  int tid = ty * blockDim.x + tx;

  int ow = bx * TW + tx;
  int oh = by * TH + ty;
  int n  = blockIdx.z / num_k_blocks;
  int k_start = (blockIdx.z % num_k_blocks) * BK;

  // 每个线程维护BK个output channel的累加器
  float sum[BK] = {0.f};
  int in_base_row = by * TH;
  int in_base_col = bx * TW;
  int in_nchw = n * C * H * W;

  // 沿C维度分BC个channel一组, 大幅减少sync次数
  for (int cb = 0; cb < C; cb += BC) {
    int cur_bc = min(BC, C - cb);  // 实际有效的channel数 (最后一组可能不足)

    // --- Step 1: 协作加载 s_inp[0:cur_bc][IH][IW] ---
    int n_inp = cur_bc * IH * IW;
    #pragma unroll 1
    for (int i = tid; i < n_inp; i += NT) {
      int ic = i / (IH * IW);
      int iy = (i / IW) % IH;
      int ix = i % IW;
      int giy = in_base_row + iy;
      int gix = in_base_col + ix;
      s_inp[ic][iy][ix] = (giy < H && gix < W)
          ? input[in_nchw + (cb + ic) * H * W + giy * W + gix] : 0.f;
    }

    // --- Step 2: 协作加载 s_wt[0:BK][0:cur_bc][R][S] ---
    int n_wt = BK * cur_bc * CRS;
    #pragma unroll 1
    for (int i = tid; i < n_wt; i += NT) {
      int ibk = i / (cur_bc * CRS);
      int ic  = (i / CRS) % cur_bc;
      int irs = i % CRS;
      int ir = irs / S, is = irs % S;
      int k = k_start + ibk;
      s_wt[ibk][ic][ir][is] = (k < K && (cb + ic) < C)
          ? weight[k * C * CRS + (cb + ic) * CRS + ir * S + is] : 0.f;
    }
    __syncthreads();

    // --- Step 3: 纯shared memory计算 (无global memory访问!) ---
    if (ow < OW && oh < OH && n < N) {
      #pragma unroll
      for (int ic = 0; ic < cur_bc; ++ic) {
        #pragma unroll
        for (int r = 0; r < R; ++r) {
          #pragma unroll
          for (int s = 0; s < S; ++s) {
            float iv = s_inp[ic][ty + r][tx + s];
            #pragma unroll
            for (int ki = 0; ki < BK; ++ki) {
              sum[ki] += iv * s_wt[ki][ic][r][s];
            }
          }
        }
      }
    }
    __syncthreads();
  }

  // --- Step 4: 写回 ---
  #pragma unroll
  for (int ki = 0; ki < BK; ++ki) {
    int k = k_start + ki;
    if (k < K && ow < OW && oh < OH && n < N)
      output[n * K * OH * OW + k * OH * OW + oh * OW + ow] = sum[ki];
  }
}

// ---- 3×3 kernel 专用 (最常见) ----
inline void launch_conv_v3_3x3(const float* input, const float* weight, float* output,
                                int N, int C, int H, int W, int K) {
  constexpr int TH = 8, TW = 8, BK = 16, BC = 16, R = 3, S = 3;
  int OH = H - R + 1, OW = W - S + 1;
  int num_k_blocks = (K + BK - 1) / BK;
  dim3 block(TW, TH);
  dim3 grid((OW + TW - 1) / TW, (OH + TH - 1) / TH, num_k_blocks * N);
  conv_v3<TH, TW, BK, BC, R, S><<<grid, block>>>(
      input, weight, output, N, C, H, W, K);
}

// ---- 7×7 kernel 专用 ----
inline void launch_conv_v3_7x7(const float* input, const float* weight, float* output,
                                int N, int C, int H, int W, int K) {
  constexpr int TH = 8, TW = 8, BK = 16, BC = 4, R = 7, S = 7;
  int OH = H - R + 1, OW = W - S + 1;
  int num_k_blocks = (K + BK - 1) / BK;
  dim3 block(TW, TH);
  dim3 grid((OW + TW - 1) / TW, (OH + TH - 1) / TH, num_k_blocks * N);
  conv_v3<TH, TW, BK, BC, R, S><<<grid, block>>>(
      input, weight, output, N, C, H, W, K);
}
