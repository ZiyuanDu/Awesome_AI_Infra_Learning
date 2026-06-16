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

template<typename Launch>
inline void bench_matmul(const char* name, int M, int N, int K, int iters,
                         const std::vector<float>& hA,
                         const std::vector<float>& hB,
                         const float* h_ref, Launch&& launch) {
  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, M * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, K * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dC, 0, M * N * sizeof(float)));

  launch(dA, dB, dC, M, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i)
    launch(dA, dB, dC, M, N, K);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  double avg_ms = ms / iters;
  double tflops = (2.0 * M * N * K) * 1e-12 / (avg_ms / 1000.0);

  std::vector<float> out(M * N);
  CUDA_CHECK(cudaMemcpy(out.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
  float err = max_rel_err(out.data(), h_ref, M * N);

  printf("  %-22s %8.4f ms  %7.2f TFLOPS  %.2e\n", name, avg_ms, tflops, err);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
}

template<typename Launch>
inline void bench_matmul_first(const char* name, int M, int N, int K, int iters,
                               const std::vector<float>& hA,
                               const std::vector<float>& hB,
                               std::vector<float>& h_ref, Launch&& launch) {
  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, M * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, K * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dC, 0, M * N * sizeof(float)));

  launch(dA, dB, dC, M, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i)
    launch(dA, dB, dC, M, N, K);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  double avg_ms = ms / iters;
  double tflops = (2.0 * M * N * K) * 1e-12 / (avg_ms / 1000.0);

  h_ref.resize(M * N);
  CUDA_CHECK(cudaMemcpy(h_ref.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  printf("  %-22s %8.4f ms  %7.2f TFLOPS  ref\n", name, avg_ms, tflops);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
}

class CublasHandle {
  cublasHandle_t h_;
public:
  CublasHandle() { cublasCreate(&h_); }
  ~CublasHandle() { cublasDestroy(h_); }
  cublasHandle_t get() const { return h_; }
};

static CublasHandle g_cublas;

// ---- 真 FP32 基线：CUDA core，不走 TF32，对标 v2/v3/v4 ----
// cuBLAS 列主序：C_col(N,M) = α * B_col(N,K) * A_col(K,M)
inline void launch_cublas_sgemm_fp32(const float* A, const float* B, float* C,
                                     int M, int N, int K) {
  const float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(
      g_cublas.get(), CUBLAS_OP_N, CUBLAS_OP_N,
      N, M, K, &alpha,
      B, CUDA_R_32F, N,
      A, CUDA_R_32F, K,
      &beta,
      C, CUDA_R_32F, N,
      CUBLAS_COMPUTE_32F,            // 真 FP32 计算与累加
      CUBLAS_GEMM_DEFAULT));
}

// ---- TF32 基线：Tensor Core，对标 v5 ----
inline void launch_cublas_sgemm_tf32(const float* A, const float* B, float* C,
                                     int M, int N, int K) {
  const float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(
      g_cublas.get(), CUBLAS_OP_N, CUBLAS_OP_N,
      N, M, K, &alpha,
      B, CUDA_R_32F, N,
      A, CUDA_R_32F, K,
      &beta,
      C, CUDA_R_32F, N,
      CUBLAS_COMPUTE_32F_FAST_TF32,  // TF32 Tensor Core
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// ---- 兼容旧接口（可选，建议优先用上面两个显式入口）----
inline void launch_cublas_sgemm(const float* A, const float* B, float* C,
                                int M, int N, int K) {
  const float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasSgemm(g_cublas.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                           N, M, K, &alpha, B, N, A, K, &beta, C, N));
}

inline void set_cublas_tf32(bool enable) {
  cublasMath_t mode = enable ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
  CUBLAS_CHECK(cublasSetMathMode(g_cublas.get(), mode));
}
