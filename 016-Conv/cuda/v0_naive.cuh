#pragma once
#include <cuda_runtime.h>

// v0: Naive direct convolution (stride=1, pad=0)
// Each thread computes one output element, iterating C,R,S from global memory
// input[N][C][H][W] * weight[K][C][R][S] -> output[N][K][OH][OW]
__global__ void conv_v0(const float* __restrict__ input,
                        const float* __restrict__ weight,
                        float* __restrict__ output,
                        int N, int C, int H, int W, int K, int R, int S) {
  int OH = W - S + 1, OW = W - S + 1;
  int ow = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = blockIdx.y * blockDim.y + threadIdx.y;
  int nk = blockIdx.z;  // flat: n * K + k
  int n = nk / K, k = nk % K;

  if (ow < OW && oh < OH && n < N && k < K) {
    float sum = 0.f;
    for (int c = 0; c < C; ++c)
      for (int r = 0; r < R; ++r)
        for (int s = 0; s < S; ++s)
          sum += input[n * C * H * W + c * H * W + (oh + r) * W + (ow + s)] *
                 weight[k * C * R * S + c * R * S + r * S + s];
    output[n * K * OH * OW + k * OH * OW + oh * OW + ow] = sum;
  }
}

inline void launch_conv_v0(const float* input, const float* weight, float* output,
                            int N, int C, int H, int W, int K, int R, int S) {
  int OH = H - R + 1, OW = W - S + 1;
  dim3 block(16, 16);
  dim3 grid((OW + 15) / 16, (OH + 15) / 16, N * K);
  conv_v0<<<grid, block>>>(input, weight, output, N, C, H, W, K, R, S);
}
