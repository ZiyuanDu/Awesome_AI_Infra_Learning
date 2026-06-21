#pragma once
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>
#include <cublas_v2.h>

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(e)); \
      exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t st = (call); \
    if (st != CUBLAS_STATUS_SUCCESS) { \
      fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)st); \
      exit(1); \
    } \
} while(0)


inline float max_rel_err(const float* a, const float* b, int N) {
  double sum_sq = 0.0;
  for (int i = 0; i < N; ++i) sum_sq += (double)b[i] * b[i];
  float rms = sqrtf((float)(sum_sq / N));
  float floor_val = fmaxf(rms * 1e-3f, 1e-6f);

  float max_rel = 0.f;
  for (int i = 0; i < N; ++i) {
    float abs_e = fabsf(a[i] - b[i]);
    float rel_e = abs_e / fmaxf(fabsf(b[i]), floor_val);
    if (rel_e > max_rel) max_rel = rel_e;
  }
  return max_rel;
}

// ---- Benchmark harness for convolution ----
template<typename Launch>
inline void bench_conv(const char* name, int N, int C, int H, int W, int K, int R, int S,
                       int iters, const std::vector<float>& h_in, const std::vector<float>& h_wt,
                       const float* h_ref, Launch&& launch) {
  int OH = H - R + 1, OW = W - S + 1;
  size_t bytes_in = N * C * H * W * sizeof(float);
  size_t bytes_wt = K * C * R * S * sizeof(float);
  size_t bytes_out = N * K * OH * OW * sizeof(float);

  float *d_in, *d_wt, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes_in));
  CUDA_CHECK(cudaMalloc(&d_wt, bytes_wt));
  CUDA_CHECK(cudaMalloc(&d_out, bytes_out));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes_in, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_wt, h_wt.data(), bytes_wt, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out, 0, bytes_out));

  launch(d_in, d_wt, d_out, N, C, H, W, K, R, S);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i)
    launch(d_in, d_wt, d_out, N, C, H, W, K, R, S);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  double avg_ms = ms / iters;
  double gflops = (2.0 * N * K * OH * OW * C * R * S) * 1e-9 / (avg_ms / 1000.0);

  std::vector<float> out(N * K * OH * OW);
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, bytes_out, cudaMemcpyDeviceToHost));
  float err = max_rel_err(out.data(), h_ref, N * K * OH * OW);

  printf("  %-22s %8.4f ms  %7.2f GFLOPS  %.2e\n", name, avg_ms, gflops, err);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_wt)); CUDA_CHECK(cudaFree(d_out));
}

template<typename Launch>
inline void bench_conv_first(const char* name, int N, int C, int H, int W, int K, int R, int S,
                             int iters, const std::vector<float>& h_in,
                             const std::vector<float>& h_wt,
                             std::vector<float>& h_ref, Launch&& launch) {
  int OH = H - R + 1, OW = W - S + 1;
  size_t bytes_in = N * C * H * W * sizeof(float);
  size_t bytes_wt = K * C * R * S * sizeof(float);
  size_t bytes_out = N * K * OH * OW * sizeof(float);

  float *d_in, *d_wt, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, bytes_in));
  CUDA_CHECK(cudaMalloc(&d_wt, bytes_wt));
  CUDA_CHECK(cudaMalloc(&d_out, bytes_out));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes_in, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_wt, h_wt.data(), bytes_wt, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out, 0, bytes_out));

  launch(d_in, d_wt, d_out, N, C, H, W, K, R, S);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i)
    launch(d_in, d_wt, d_out, N, C, H, W, K, R, S);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  double avg_ms = ms / iters;
  double gflops = (2.0 * N * K * OH * OW * C * R * S) * 1e-9 / (avg_ms / 1000.0);

  h_ref.resize(N * K * OH * OW);
  CUDA_CHECK(cudaMemcpy(h_ref.data(), d_out, bytes_out, cudaMemcpyDeviceToHost));

  printf("  %-22s %8.4f ms  %7.2f GFLOPS  ref\n", name, avg_ms, gflops);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_wt)); CUDA_CHECK(cudaFree(d_out));
}

// cuBLAS wrapper for im2col path (used by v1)
class CublasHandle {
  cublasHandle_t h_;
public:
  CublasHandle() { cublasCreate(&h_); }
  ~CublasHandle() { cublasDestroy(h_); }
  cublasHandle_t get() const { return h_; }
};

static CublasHandle g_cublas;

inline void launch_cublas_sgemm(const float* A, const float* B, float* C,
                                 int M, int N, int K) {
  const float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(
      g_cublas.get(), CUBLAS_OP_N, CUBLAS_OP_N,
      N, M, K, &alpha,
      B, CUDA_R_32F, N,
      A, CUDA_R_32F, K,
      &beta,
      C, CUDA_R_32F, N,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT));
}
