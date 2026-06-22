#pragma once
#include <cuda_runtime.h>

// ============================================================================
// v4: Winograd F(2×2, 3×3) — BC-channel tiling + 权值共享缓存
// 每线程输出 2×2 tile (4 pixels), BK output channel, 消除4×冗余
// 共享内存: s_inp[BC][IH][IW] + s_wt[BK][BC][16]
// sync: 2*ceil(C/BC) (与v3相同)
// 算术: 每(c,k)对 16 FMAs vs 传统36 FMAs → 2.25×减少
// ============================================================================

__global__ void wino_wt_transform(const float* __restrict__ weight,
                                   float* __restrict__ wino_wt,
                                   int K, int C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= K * C) return;
  int k = idx / C, c = idx % C;
  float g[3][3];
  #pragma unroll
  for (int r = 0; r < 3; ++r)
    #pragma unroll
    for (int s = 0; s < 3; ++s)
      g[r][s] = weight[k * C * 9 + c * 9 + r * 3 + s];
  float t[4][3];
  #pragma unroll
  for (int j = 0; j < 3; ++j) {
    t[0][j] = g[0][j]; t[1][j] = (g[0][j]+g[1][j]+g[2][j])*0.5f;
    t[2][j] = (g[0][j]-g[1][j]+g[2][j])*0.5f; t[3][j] = g[2][j];
  }
  float* u = wino_wt + idx * 16;
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    u[i*4+0] = t[i][0]; u[i*4+1] = (t[i][0]+t[i][1]+t[i][2])*0.5f;
    u[i*4+2] = (t[i][0]-t[i][1]+t[i][2])*0.5f; u[i*4+3] = t[i][2];
  }
}

template <int TH = 16, int TW = 16, int BK = 16, int BC = 8>
__global__ void conv_v4_wino(const float* __restrict__ input,
                              const float* __restrict__ wino_wt,
                              float* __restrict__ output,
                              int N, int C, int H, int W, int K) {
  constexpr int IH = TH + 2, IW = TW + 2;
  constexpr int TWINO = TH / 2 * TW / 2;   // Winograd tile count / block
  constexpr int NT = TWINO;                 // 1 thread per Winograd 2×2 tile

  __shared__ float s_inp[BC][IH][IW];
  __shared__ float s_wt [BK][BC][16];

  int OH = H - 2, OW = W - 2;
  int num_k_blocks = (K + BK - 1) / BK;

  int bx = blockIdx.x, by = blockIdx.y;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int tile_y = threadIdx.y, tile_x = threadIdx.x;
  int sr = tile_y * 2, sc = tile_x * 2;  // s_inp起始

  int n = blockIdx.z / num_k_blocks;
  int k_start = (blockIdx.z % num_k_blocks) * BK;
  int in_base_row = by * TH, in_base_col = bx * TW;

  float sum[BK][16] = {{0.f}};
  int out_base_row = by * TH + sr;
  int out_base_col = bx * TW + sc;
  int in_nchw = n * C * H * W;

  for (int cb = 0; cb < C; cb += BC) {
    int cur_bc = min(BC, C - cb);

    // Load s_inp[cur_bc][IH][IW]
    #pragma unroll 1
    for (int i = tid; i < cur_bc * IH * IW; i += NT) {
      int ic = i / (IH * IW), iy = (i / IW) % IH, ix = i % IW;
      int giy = in_base_row + iy, gix = in_base_col + ix;
      s_inp[ic][iy][ix] = (giy < H && gix < W)
          ? input[in_nchw + (cb+ic) * H * W + giy * W + gix] : 0.f;
    }

    // Load s_wt[BK][cur_bc][16]
    #pragma unroll 1
    for (int i = tid; i < BK * cur_bc * 16; i += NT) {
      int ibk = i / (cur_bc*16), ic = (i/16)%cur_bc, j = i%16;
      int k = k_start + ibk;
      s_wt[ibk][ic][j] = (k < K)
          ? wino_wt[(k * C + (cb+ic)) * 16 + j] : 0.f;
    }
    __syncthreads();

    if (out_base_row+1 < OH && out_base_col+1 < OW && n < N) {
      for (int ic = 0; ic < cur_bc; ++ic) {
        float d[4][4], t[4][4], V[16];
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
          #pragma unroll
          for (int q = 0; q < 4; ++q)
            d[r][q] = s_inp[ic][sr+r][sc+q];
          t[r][0] = d[r][0]-d[r][2]; t[r][1]=d[r][1]+d[r][2];
          t[r][2] = d[r][2]-d[r][1]; t[r][3]=d[r][1]-d[r][3];
        }
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
          V[ 0+q] = t[0][q]-t[2][q]; V[ 4+q] = t[1][q]+t[2][q];
          V[ 8+q] = t[2][q]-t[1][q]; V[12+q] = t[1][q]-t[3][q];
        }
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
          float vj = V[j];
          #pragma unroll
          for (int ki = 0; ki < BK; ++ki)
            sum[ki][j] += vj * s_wt[ki][ic][j];
        }
      }
    }
    __syncthreads();
  }

  if (out_base_row+1 < OH && out_base_col+1 < OW && n < N) {
    #pragma unroll
    for (int ki = 0; ki < BK; ++ki) {
      int k = k_start + ki;
      if (k >= K) continue;
      float* M = sum[ki];
      float ta0 = M[0]+M[1]+M[2],   ta1 = M[1]-M[2]-M[3];
      float ta2 = M[4]+M[5]+M[6],   ta3 = M[5]-M[6]-M[7];
      float ta4 = M[8]+M[9]+M[10],  ta5 = M[9]-M[10]-M[11];
      float ta6 = M[12]+M[13]+M[14],ta7 = M[13]-M[14]-M[15];
      float o00 = ta0+ta2+ta4, o10 = ta2-ta4-ta6;
      float o01 = ta1+ta3+ta5, o11 = ta3-ta5-ta7;
      output[n*K*OH*OW + k*OH*OW + (out_base_row+0)*OW + (out_base_col+0)] = o00;
      output[n*K*OH*OW + k*OH*OW + (out_base_row+1)*OW + (out_base_col+0)] = o10;
      output[n*K*OH*OW + k*OH*OW + (out_base_row+0)*OW + (out_base_col+1)] = o01;
      output[n*K*OH*OW + k*OH*OW + (out_base_row+1)*OW + (out_base_col+1)] = o11;
    }
  }
}

inline void launch_conv_v4_wino(const float* input, const float* weight, float* output,
                                 float* d_wino_wt, int N, int C, int H, int W, int K) {
  constexpr int TH = 16, TW = 16, BK = 16, BC = 8;
  wino_wt_transform<<<(K * C + 255) / 256, 256>>>(weight, d_wino_wt, K, C);
  int OH = H - 2, OW = W - 2;
  int num_k_blocks = (K + BK - 1) / BK;
  dim3 block(TW/2, TH/2);
  dim3 grid((OW + TW - 1) / TW, (OH + TH - 1) / TH, num_k_blocks * N);
  conv_v4_wino<TH, TW, BK, BC><<<grid, block>>>(
      input, d_wino_wt, output, N, C, H, W, K);
}
