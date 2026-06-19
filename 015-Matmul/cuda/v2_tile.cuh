#pragma once
#include <cuda_runtime.h>


template <int BM = 64, int BN = 64, int BK = 32, int TM = 8, int TN = 4>
__global__ void sgemm_v2(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float* __restrict__ C,
                         int M, int N, int K) {

  static_assert(BK % 4 == 0, "BK must be multiple of 4 for float4");
  static_assert(BN % 4 == 0, "BN must be multiple of 4 for float4");
  static_assert(TN % 4 == 0, "TN must be multiple of 4 for float4");

  constexpr int NT = (BM / TM) * (BN / TN);
  __shared__ float SA[BM][BK];
  __shared__ float SB[BK][BN];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = ty * blockDim.x + tx;

  float sum[TM][TN] = {{0.f}};

  for (int tile = 0; tile < (K + BK - 1) / BK; ++tile) {
    int gk = tile * BK;


    #pragma unroll
    for (int i = tid; i < BM * BK / 4; i += NT) {
      int load_a_row = i / (BK / 4);
      int load_a_col = (i % (BK / 4)) * 4;                  
      int ga_row = blockIdx.y * BM + load_a_row;
      int ga_col = gk + load_a_col;
      if (ga_row < M && ga_col < K)
        *(float4*)&SA[load_a_row][load_a_col] =              
            *(const float4*)&A[ga_row * K + ga_col];        
      else
        *(float4*)&SA[load_a_row][load_a_col] =              
            make_float4(0.f, 0.f, 0.f, 0.f);
    }

    #pragma unroll
    for (int i = tid; i < BK * BN / 4; i += NT) {
      int load_b_row = i / (BN / 4);
      int load_b_col = (i % (BN / 4)) * 4;                 
      int gb_row = gk + load_b_row;
      int gb_col = blockIdx.x * BN + load_b_col;
      if (gb_row < K && gb_col < N)
        *(float4*)&SB[load_b_row][load_b_col] =             
            *(const float4*)&B[gb_row * N + gb_col];         
      else
        *(float4*)&SB[load_b_row][load_b_col] =             
            make_float4(0.f, 0.f, 0.f, 0.f);
    }
    __syncthreads();


    #pragma unroll
    for (int k = 0; k < BK; ++k) {
      #pragma unroll
      for (int n = 0; n < TN; n += 4) {                     
        float4 b_val = *(float4*)&SB[k][tx * TN + n];       
        #pragma unroll
        for (int m = 0; m < TM; ++m) {
          float a_val = SA[ty * TM + m][k];
          sum[m][n + 0] += a_val * b_val.x;                 
          sum[m][n + 1] += a_val * b_val.y;
          sum[m][n + 2] += a_val * b_val.z;
          sum[m][n + 3] += a_val * b_val.w;
        }
      }
    }
    __syncthreads();
  }


  #pragma unroll
  for (int m = 0; m < TM; ++m) {
    int row = blockIdx.y * BM + ty * TM + m;
    if (row >= M) continue;
    #pragma unroll
    for (int n = 0; n < TN; n += 4) {                      
      int col = blockIdx.x * BN + tx * TN + n;
      if (col < N)
        *(float4*)&C[row * N + col] =                        
            make_float4(sum[m][n],     sum[m][n + 1],
                        sum[m][n + 2], sum[m][n + 3]);
    }
  }
}

template <int BM = 64, int BN = 64, int BK = 32, int TM = 8, int TN = 4>
inline void launch_sgemm_v2(const float* A, const float* B, float* C,
                            int M, int N, int K) {
  dim3 block(BN / TN, BM / TM);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  sgemm_v2<BM, BN, BK, TM, TN><<<grid, block>>>(A, B, C, M, N, K);
}
