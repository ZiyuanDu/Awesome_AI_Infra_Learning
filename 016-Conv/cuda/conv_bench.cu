#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#include "bench.cuh"
#include "v0_naive.cuh"
#include "v1_im2col.cuh"
#include "v2_tiled.cuh"
#include "v3_smem.cuh"

int main() {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("  CUDA Conv2D — v0 to v3 optimization journey\n");
  printf("  GPU: %s  |  SMs: %d  |  sm_%d%d\n",
         prop.name, prop.multiProcessorCount, prop.major, prop.minor);

  constexpr int ITERS = 100;

  auto make_data = [](int N, int C, int H, int W, int K, int R, int S,
                      std::vector<float>& h_in, std::vector<float>& h_wt) {
    h_in.resize(N * C * H * W); h_wt.resize(K * C * R * S);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& v : h_in) v = dist(rng);
    for (auto& v : h_wt) v = dist(rng);
  };

  auto run = [&](const char* title, int N, int C, int H, int W, int K, int R, int S,
                  auto&& body) {
    int OH = H - R + 1, OW = W - S + 1;
    printf("--- %s  (%.1f GFLOPs) ---\n", title,
           2.0 * N * K * OH * OW * C * R * S * 1e-9);
    std::vector<float> h_in, h_wt, h_ref;
    make_data(N, C, H, W, K, R, S, h_in, h_wt);
    printf("  %-24s  %9s  %8s  %s\n", "kernel", "time", "GFLOPS", "vs ref");

    auto first = [&](const char* name, auto&& launch) {
      bench_conv_first(name, N, C, H, W, K, R, S, ITERS, h_in, h_wt, h_ref, launch);
    };
    auto bench = [&](const char* name, auto&& launch) {
      bench_conv(name, N, C, H, W, K, R, S, ITERS, h_in, h_wt, h_ref.data(), launch);
    };
    body(first, bench);
    printf("\n");
  };

  // 小卷积：常见于ResNet等网络的中间层
  run("Conv N=1 C=64 H=56 W=56 K=64 R=3 S=3",
      1, 64, 56, 56, 64, 3, 3, [&](auto first, auto bench) {
    int N = 1, C = 64, H = 56, W = 56, K = 64, R = 3, S = 3;
    int CRS = C * R * S, OH = H - R + 1, OW = W - S + 1;
    int out_spatial = N * OH * OW;

    float* d_col;
    CUDA_CHECK(cudaMalloc(&d_col, CRS * out_spatial * sizeof(float)));
    auto launch_v1 = [d_col](const float* in, const float* wt, float* out,
                              int N, int C, int H, int W, int K, int R, int S) {
      launch_conv_v1(in, wt, out, d_col, N, C, H, W, K, R, S);
    };

    first("v0-naive",          launch_conv_v0);
    bench("v1-im2col+sgemm",   launch_v1);
    bench("v2-tiled",          launch_conv_v2<>);
    bench("v3-smem",           launch_conv_v3<>);
    CUDA_CHECK(cudaFree(d_col));
  });

  // 大卷积：输入通道多，常见于网络第一层
  run("Conv N=1 C=3 H=224 W=224 K=64 R=7 S=7",
      1, 3, 224, 224, 64, 7, 7, [&](auto first, auto bench) {
    int N = 1, C = 3, H = 224, W = 224, K = 64, R = 7, S = 7;
    int CRS = C * R * S, OH = H - R + 1, OW = W - S + 1;
    int out_spatial = N * OH * OW;

    float* d_col;
    CUDA_CHECK(cudaMalloc(&d_col, CRS * out_spatial * sizeof(float)));
    auto launch_v1 = [d_col](const float* in, const float* wt, float* out,
                              int N, int C, int H, int W, int K, int R, int S) {
      launch_conv_v1(in, wt, out, d_col, N, C, H, W, K, R, S);
    };

    first("v0-naive",          launch_conv_v0);
    bench("v1-im2col+sgemm",   launch_v1);
    bench("v2-tiled",          launch_conv_v2<>);
    bench("v3-smem",           launch_conv_v3<8, 8, 8, 4>);
    CUDA_CHECK(cudaFree(d_col));
  });

  // 深度可分离卷积风格：C=K=512, 大量channel
  run("Conv N=1 C=128 H=28 W=28 K=128 R=3 S=3",
      1, 128, 28, 28, 128, 3, 3, [&](auto first, auto bench) {
    int N = 1, C = 128, H = 28, W = 28, K = 128, R = 3, S = 3;
    int CRS = C * R * S, OH = H - R + 1, OW = W - S + 1;
    int out_spatial = N * OH * OW;

    float* d_col;
    CUDA_CHECK(cudaMalloc(&d_col, CRS * out_spatial * sizeof(float)));
    auto launch_v1 = [d_col](const float* in, const float* wt, float* out,
                              int N, int C, int H, int W, int K, int R, int S) {
      launch_conv_v1(in, wt, out, d_col, N, C, H, W, K, R, S);
    };

    first("v0-naive",          launch_conv_v0);
    bench("v1-im2col+sgemm",   launch_v1);
    bench("v2-tiled",          launch_conv_v2<8, 8, 8>);
    bench("v3-smem",           launch_conv_v3<8, 8, 8, 4>);
    CUDA_CHECK(cudaFree(d_col));
  });

  // Batch>1 convolution — v1(im2col) supports N=1 only (output layout differs)
  run("Conv N=4 C=32 H=32 W=32 K=64 R=3 S=3",
      4, 32, 32, 32, 64, 3, 3, [&](auto first, auto bench) {
    first("v0-naive",          launch_conv_v0);
    bench("v2-tiled",          launch_conv_v2<>);
    bench("v3-smem",           launch_conv_v3<>);
  });

  return 0;
}
