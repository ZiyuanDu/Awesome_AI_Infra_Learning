#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#include "bench.cuh"
#include "v0_naive.cuh"
#include "v1_0smem.cuh"
#include "v1_1Dtiling.cuh"
#include "v1_2Dtiling.cuh"
#include "v2_tile.cuh"
#include "v3_dbuf.cuh"

int main() {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("  CUDA SGEMM — v0 to v6 optimization journey\n");
  printf("  GPU: %s  |  SMs: %d  |  sm_%d%d\n",
         prop.name, prop.multiProcessorCount, prop.major, prop.minor);

  constexpr int ITERS = 100;

  auto make_data = [](int M, int K, int N,
                      std::vector<float>& hA, std::vector<float>& hB) {
    hA.resize(M * K); hB.resize(K * N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& v : hA) v = dist(rng);
    for (auto& v : hB) v = dist(rng);
  };

  // 一个 case：建数据 → 打表头 → body 里第一个 first() 跑 ref，其余 bench() 对比
  auto run = [&](const char* title, int M, int N, int K, auto&& body) {
    printf("--- %s  (%.1f GFLOPs) ---\n", title, 2.0 * M * N * K * 1e-9);
    std::vector<float> hA, hB, h_ref;
    make_data(M, K, N, hA, hB);
    printf("  %-24s  %9s  %8s  %s\n", "kernel", "time", "TFLOPS", "vs ref");

    auto first = [&](const char* name, auto&& launch) {
      bench_matmul_first(name, M, N, K, ITERS, hA, hB, h_ref, launch);
    };
    auto bench = [&](const char* name, auto&& launch) {
      bench_matmul(name, M, N, K, ITERS, hA, hB, h_ref.data(), launch);
    };
    body(first, bench);
    printf("\n");
  };

  run("SGEMM M=N=K=4096", 1024, 1024, 1024, [&](auto first, auto bench) {
    first("v0-naive",          launch_sgemm_v0);
    bench("v1-smem",           launch_sgemm_v1);
    bench("v1-1Dtiling",       launch_sgemm_v1_1<>);
    bench("v1-2Dtiling",       launch_sgemm_v1_2<>);
    bench("v2-tile",           launch_sgemm_v2<>);
    bench("v3-dbuf",           launch_sgemm_v3<>);
    bench("cuBLAS(FP32)",      launch_cublas_sgemm_fp32);
    bench("v5-wmma",           launch_sgemm_v5);
  });

  return 0;
}
