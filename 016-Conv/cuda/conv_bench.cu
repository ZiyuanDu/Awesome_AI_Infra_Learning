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
#include "v4_wino.cuh"

int main() {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("  CUDA Conv2D — v0-v3 + cuBLAS + cuDNN\n");
  printf("  GPU: %s  |  SMs: %d  |  sm_%d%d\n\n",
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

  auto run = [&](const char* title, int N, int C, int H, int W,
                  int K, int R, int S, auto&& body) {
    int OH = H - R + 1, OW = W - S + 1;
    double gflop_work = 2.0 * N * K * OH * OW * C * R * S * 1e-9;
    printf("--- %s  (%.2f GFLOPs) ---\n", title, gflop_work);
    std::vector<float> h_in, h_wt, h_ref;
    make_data(N, C, H, W, K, R, S, h_in, h_wt);
    printf("  %-22s  %9s  %8s  %s\n", "kernel", "time", "GFLOPS", "vs v0");
    auto first = [&](const char* name, auto&& launch) {
      bench_conv_first(name, N, C, H, W, K, R, S, ITERS, h_in, h_wt, h_ref, launch);
    };
    auto bench = [&](const char* name, auto&& launch) {
      bench_conv(name, N, C, H, W, K, R, S, ITERS, h_in, h_wt, h_ref.data(), launch);
    };
    body(first, bench);
    printf("\n");
  };

  // ---- Helpers ----
  auto make_launch_v1 = [](float* d_col) {
    return [d_col](const float* in, const float* wt, float* out,
                    int N, int C, int H, int W, int K, int R, int S) {
      launch_conv_v1(in, wt, out, d_col, N, C, H, W, K, R, S);
    };
  };

  auto v3_3x3 = [](const float* in, const float* wt, float* out,
                    int N, int C, int H, int W, int K, int, int) {
    launch_conv_v3_3x3(in, wt, out, N, C, H, W, K);
  };
  auto v3_7x7 = [](const float* in, const float* wt, float* out,
                    int N, int C, int H, int W, int K, int, int) {
    launch_conv_v3_7x7(in, wt, out, N, C, H, W, K);
  };

  // cuBLAS+im2col: capture pre-allocated d_col (避免benchmark loop内malloc)
  auto make_cublas_launch = [](float* d_col) {
    return [d_col](const float* in, const float* wt, float* out,
                    int N, int C, int H, int W, int K, int R, int S) {
      int OH = H - R + 1, OW = W - S + 1;
      int CRS = C * R * S, out_spatial = N * OH * OW;
      im2col_kernel<<<(out_spatial + 255) / 256, 256>>>(
          in, d_col, N, C, H, W, R, S);
      cublas_sgemm(wt, d_col, out, K, out_spatial, CRS);
    };
  };
  // ================================================================
  // Test 1: ResNet中间层 — 3×3 kernel, 64→64 channels
  // ================================================================
  {
    int N = 1, C = 64, H = 56, W = 56, K = 64, R = 3, S = 3;
    int OH = H - R + 1, OW = W - S + 1, CRS = C * R * S;
    float* d_col;  // shared col buffer for v1 + cuBLAS+im2col
    CUDA_CHECK(cudaMalloc(&d_col, CRS * OH * OW * sizeof(float)));

    auto v1 = make_launch_v1(d_col);
    auto cuBLAS_fp32 = make_cublas_launch(d_col);

    // v4 Winograd: 预计算transformed weight buffer
    float* d_wino_wt;
    CUDA_CHECK(cudaMalloc(&d_wino_wt, K * C * 16 * sizeof(float)));
    auto v4_wino = [d_wino_wt](const float* in, const float* wt, float* out,
                                int N, int C, int H, int W, int K, int, int) {
      launch_conv_v4_wino(in, wt, out, d_wino_wt, N, C, H, W, K);
    };

#ifdef HAS_CUDNN
    CudnnConvCtx cudnn_ctx(N, C, H, W, K, R, S);
    auto cudnn_launch = [&](const float* in, const float* wt, float* out,
                             int, int, int, int, int, int, int) {
      cudnn_ctx.forward(in, wt, out);
    };
#endif

    run("ResNet mid: N=1 C=64 H=56 W=56 K=64 R=3 S=3",
        N, C, H, W, K, R, S, [&](auto first, auto bench) {
      first("v0-naive",              launch_conv_v0);
      bench("v1-im2col+gemm(f4)",    v1);
      bench("v2-tiled(BK=4)",        launch_conv_v2<8, 8, 4>);
      bench("v3-smem(BC=16,BK=16)",  v3_3x3);
      bench("v4-wino(F2x2,3x3)",     v4_wino);
      bench("cuBLAS+im2col(FP32)",   cuBLAS_fp32);
#ifdef HAS_CUDNN
      bench("cuDNN(FP32)",           cudnn_launch);
#endif
    });
    CUDA_CHECK(cudaFree(d_col));
    CUDA_CHECK(cudaFree(d_wino_wt));
  }

  // ================================================================
  // Test 2: 网络第一层 — 7×7 kernel, 3→64 channels
  // ================================================================
  {
    int N = 1, C = 3, H = 224, W = 224, K = 64, R = 7, S = 7;
    int OH = H - R + 1, OW = W - S + 1, CRS = C * R * S;
    float* d_col;
    CUDA_CHECK(cudaMalloc(&d_col, CRS * OH * OW * sizeof(float)));

    auto v1 = make_launch_v1(d_col);
    auto cuBLAS_fp32 = make_cublas_launch(d_col);
#ifdef HAS_CUDNN
    CudnnConvCtx cudnn_ctx(N, C, H, W, K, R, S);
    auto cudnn_launch = [&](const float* in, const float* wt, float* out,
                             int, int, int, int, int, int, int) {
      cudnn_ctx.forward(in, wt, out);
    };
#endif

    run("First layer: N=1 C=3 H=224 W=224 K=64 R=7 S=7",
        N, C, H, W, K, R, S, [&](auto first, auto bench) {
      first("v0-naive",              launch_conv_v0);
      bench("v1-im2col+gemm(f4)",    v1);
      bench("v2-tiled(BK=4)",        launch_conv_v2<8, 8, 4>);
      bench("v3-smem(BC=4,BK=16)",   v3_7x7);
      bench("cuBLAS+im2col(FP32)",   cuBLAS_fp32);
#ifdef HAS_CUDNN
      bench("cuDNN(FP32)",           cudnn_launch);
#endif
    });
    CUDA_CHECK(cudaFree(d_col));
  }

  // ================================================================
  // Test 3: 深层卷积 — 128→128 channels, 3×3
  // ================================================================
  {
    int N = 1, C = 128, H = 28, W = 28, K = 128, R = 3, S = 3;
    int OH = H - R + 1, OW = W - S + 1, CRS = C * R * S;
    float* d_col;
    CUDA_CHECK(cudaMalloc(&d_col, CRS * OH * OW * sizeof(float)));

    auto v1 = make_launch_v1(d_col);
    auto cuBLAS_fp32 = make_cublas_launch(d_col);

    float* d_wino_wt;
    CUDA_CHECK(cudaMalloc(&d_wino_wt, K * C * 16 * sizeof(float)));
    auto v4_wino = [d_wino_wt](const float* in, const float* wt, float* out,
                                int N, int C, int H, int W, int K, int, int) {
      launch_conv_v4_wino(in, wt, out, d_wino_wt, N, C, H, W, K);
    };

#ifdef HAS_CUDNN
    CudnnConvCtx cudnn_ctx(N, C, H, W, K, R, S);
    auto cudnn_launch = [&](const float* in, const float* wt, float* out,
                             int, int, int, int, int, int, int) {
      cudnn_ctx.forward(in, wt, out);
    };
#endif

    run("Deep layer: N=1 C=128 H=28 W=28 K=128 R=3 S=3",
        N, C, H, W, K, R, S, [&](auto first, auto bench) {
      first("v0-naive",              launch_conv_v0);
      bench("v1-im2col+gemm(f4)",    v1);
      bench("v2-tiled(BK=4)",        launch_conv_v2<8, 8, 4>);
      bench("v3-smem(BC=16,BK=16)",  v3_3x3);
      bench("v4-wino(F2x2,3x3)",     v4_wino);
      bench("cuBLAS+im2col(FP32)",   cuBLAS_fp32);
#ifdef HAS_CUDNN
      bench("cuDNN(FP32)",           cudnn_launch);
#endif
    });
    CUDA_CHECK(cudaFree(d_col));
    CUDA_CHECK(cudaFree(d_wino_wt));
  }

  // ================================================================
  // Test 4: Batch>1 (v1不支持N>1)
  // ================================================================
  {
    int N = 4, C = 32, H = 32, W = 32, K = 64, R = 3, S = 3;
    run("Batch: N=4 C=32 H=32 W=32 K=64 R=3 S=3",
        N, C, H, W, K, R, S, [&](auto first, auto bench) {
      first("v0-naive",              launch_conv_v0);
      bench("v2-tiled(BK=4)",        launch_conv_v2<8, 8, 4>);
      bench("v3-smem(BC=16,BK=16)",  v3_3x3);
#ifdef HAS_CUDNN
      CudnnConvCtx cudnn_ctx(N, C, H, W, K, R, S);
      auto cudnn_launch = [&](const float* in, const float* wt, float* out,
                               int, int, int, int, int, int, int) {
        cudnn_ctx.forward(in, wt, out);
      };
      bench("cuDNN(FP32)",           cudnn_launch);
#endif
    });
  }

  return 0;
}
