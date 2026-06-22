#pragma once
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>
#include <cublas_v2.h>
#ifdef HAS_CUDNN
#include <cudnn.h>
#endif

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

#ifdef HAS_CUDNN
#define CUDNN_CHECK(call) do { \
    cudnnStatus_t st = (call); \
    if (st != CUDNN_STATUS_SUCCESS) { \
      fprintf(stderr, "cuDNN error %s:%d: %s\n", __FILE__, __LINE__, \
              cudnnGetErrorString(st)); \
      exit(1); \
    } \
} while(0)
#endif


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

// ============================================================================
// cuBLAS im2col + GEMM baseline (same algorithm as v1, but SGEMM by cuBLAS)
// ============================================================================
class CublasHandle {
  cublasHandle_t h_;
public:
  CublasHandle() { cublasCreate(&h_); }
  ~CublasHandle() { cublasDestroy(h_); }
  cublasHandle_t get() const { return h_; }
};

static CublasHandle g_cublas;

// cuBLAS GEMM: C[M][N] = A[M][K] * B[K][N], 使用cublasGemmEx
inline void cublas_sgemm(const float* A, const float* B, float* C,
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

// cuBLAS TF32 tensor core path — 用于对比TF32精度下的加速效果
inline void cublas_sgemm_tf32(const float* A, const float* B, float* C,
                               int M, int N, int K) {
  const float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(
      g_cublas.get(), CUBLAS_OP_N, CUBLAS_OP_N,
      N, M, K, &alpha,
      B, CUDA_R_32F, N,
      A, CUDA_R_32F, K,
      &beta,
      C, CUDA_R_32F, N,
      CUBLAS_COMPUTE_32F_FAST_TF32,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// ============================================================================
// cuDNN conv baseline (工业gold standard)
// ============================================================================
#ifdef HAS_CUDNN
class CudnnConvCtx {
  cudnnHandle_t handle_;
  cudnnTensorDescriptor_t in_desc_, out_desc_;
  cudnnFilterDescriptor_t wt_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnConvolutionFwdAlgo_t algo_;
  size_t ws_size_;
  void* ws_;
  float alpha_ = 1.0f, beta_ = 0.0f;

public:
  CudnnConvCtx(int N, int C, int H, int W, int K, int R, int S) {
    CUDNN_CHECK(cudnnCreate(&handle_));

    // Input: [N, C, H, W]
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc_));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(in_desc_, CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT, N, C, H, W));

    // Weight: [K, C, R, S]
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&wt_desc_));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(wt_desc_, CUDNN_DATA_FLOAT,
                                           CUDNN_TENSOR_NCHW, K, C, R, S));

    // Convolution descriptor
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        conv_desc_, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));

    // Output: [N, K, OH, OW]
    int OH = H - R + 1, OW = W - S + 1;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc_));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(out_desc_, CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT, N, K, OH, OW));

    // Auto-select best algorithm
    int requested = 1, returned = 0;
    cudnnConvolutionFwdAlgoPerf_t perf;
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
        handle_, in_desc_, wt_desc_, conv_desc_, out_desc_,
        1, &requested, &perf));
    algo_ = perf.algo;
    ws_size_ = perf.memory;
    if (ws_size_ > 0)
      CUDA_CHECK(cudaMalloc(&ws_, ws_size_));
  }

  ~CudnnConvCtx() {
    if (ws_) cudaFree(ws_);
    cudnnDestroyTensorDescriptor(in_desc_);
    cudnnDestroyTensorDescriptor(out_desc_);
    cudnnDestroyFilterDescriptor(wt_desc_);
    cudnnDestroyConvolutionDescriptor(conv_desc_);
    cudnnDestroy(handle_);
  }

  void forward(const float* in, const float* wt, float* out) {
    CUDNN_CHECK(cudnnConvolutionForward(
        handle_, &alpha_, in_desc_, in, wt_desc_, wt,
        conv_desc_, algo_, ws_, ws_size_, &beta_, out_desc_, out));
  }

  static void launch(const float* in, const float* wt, float* out,
                     int N, int C, int H, int W, int K, int R, int S,
                     CudnnConvCtx* self) {
    self->forward(in, wt, out);
  }
};
#endif  // HAS_CUDNN
