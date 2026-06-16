// ============================================================================
// cutlass_sgemm.cu — CUTLASS SGEMM benchmark for comparison
//
// Requires CUTLASS (https://github.com/NVIDIA/cutlass).
// Build: see cutlass/CMakeLists.txt
//
// Compares:
//   1. CUTLASS SGEMM (FP32)
//   2. CUTLASS TF32 Tensor Op (Ampere+)
//   3. cuBLAS (for reference)
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUTLASS headers
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(e)); \
      exit(1); \
    } \
} while(0)

// ---------------------------------------------------------------------------
// Error metric
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Benchmark helper
// ---------------------------------------------------------------------------
template<typename Launch>
void bench(const char* name, int M, int N, int K, int iters,
           const std::vector<float>& hA, const std::vector<float>& hB,
           const float* h_ref, Launch&& launch) {
  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, M * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, K * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

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

  printf("  %-30s %8.4f ms  %7.2f TFLOPS  %.2e\n", name, avg_ms, tflops, err);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
}

// ---------------------------------------------------------------------------
// cuBLAS
// ---------------------------------------------------------------------------
class CublasHandle {
  cublasHandle_t h_;
public:
  CublasHandle() { cublasCreate(&h_); }
  ~CublasHandle() { cublasDestroy(h_); }
  cublasHandle_t get() const { return h_; }
};

static CublasHandle g_cublas;

inline void launch_cublas(const float* A, const float* B, float* C,
                           int M, int N, int K) {
  float alpha = 1.0f, beta = 0.0f;
  cublasSgemm(g_cublas.get(), CUBLAS_OP_N, CUBLAS_OP_N,
              N, M, K, &alpha, B, N, A, K, &beta, C, N);
}

// ---------------------------------------------------------------------------
// CUTLASS SGEMM (FP32 — SIMT)
// ---------------------------------------------------------------------------
using CutlassSgemm = cutlass::gemm::device::Gemm<
    float,                           // ElementA
    cutlass::layout::RowMajor,       // LayoutA
    float,                           // ElementB
    cutlass::layout::RowMajor,       // LayoutB
    float,                           // ElementC
    cutlass::layout::RowMajor,       // LayoutC
    float,                           // Accumulator
    cutlass::arch::OpClassSimt,      // OpClass
    cutlass::arch::Sm80,            // ArchTag
    cutlass::gemm::GemmShape<128, 128, 8>,   // ThreadblockShape
    cutlass::gemm::GemmShape<64, 64, 8>,      // WarpShape
    cutlass::gemm::GemmShape<1, 1, 1>,         // InstructionShape
    cutlass::epilogue::thread::LinearCombination<
        float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2>;  // Stages

// ---------------------------------------------------------------------------
// CUTLASS SGEMM (TF32 Tensor Op — Ampere+)
// ---------------------------------------------------------------------------
using CutlassSgemmTF32 = cutlass::gemm::device::Gemm<
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<
        float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3>;  // Stages

// ---------------------------------------------------------------------------
// CUTLASS launch helpers
// ---------------------------------------------------------------------------
void launch_cutlass_sgemm(const float* A, const float* B, float* C,
                           int M, int N, int K) {
  CutlassSgemm gemm_op;
  typename CutlassSgemm::Arguments args(
      {M, N, K},          // GEMM dimensions
      {A, K},             // A (M×K, ld=K)
      {B, N},             // B (K×N, ld=N)
      {C, N},             // C (M×N, ld=N)
      {C, N},             // D = C (in-place accumulation)
      {1.0f, 0.0f}        // α, β
  );
  cutlass::Status status = gemm_op(args);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "CUTLASS SGEMM failed\n");
    exit(1);
  }
}

void launch_cutlass_tf32(const float* A, const float* B, float* C,
                          int M, int N, int K) {
  CutlassSgemmTF32 gemm_op;
  typename CutlassSgemmTF32::Arguments args(
      {M, N, K}, {A, K}, {B, N}, {C, N}, {C, N}, {1.0f, 0.0f});
  cutlass::Status status = gemm_op(args);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "CUTLASS TF32 failed\n");
    exit(1);
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("================================================================\n");
  printf("  CUTLASS SGEMM Benchmark\n");
  printf("  GPU: %s  |  SMs: %d  |  sm_%d%d\n",
         prop.name, prop.multiProcessorCount, prop.major, prop.minor);
  printf("================================================================\n\n");

  const int ITERS = 100;

  auto make_data = [](int M, int K, int N,
                      std::vector<float>& hA, std::vector<float>& hB) {
    hA.resize(M * K); hB.resize(K * N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& v : hA) v = dist(rng);
    for (auto& v : hB) v = dist(rng);
  };

  auto print_hdr = []() {
    printf("  %-30s %10s   %8s   %s\n", "kernel", "time", "TFLOPS", "max err");
  };

  // -------------------------------------------------------------------
  // Test 1: 2048² square
  // -------------------------------------------------------------------
  {
    int M = 2048, N = 2048, K = 2048;
    printf("--- SGEMM M=N=K=%d  (%.1f GFLOPs) ---\n",
           M, 2.0 * M * N * K * 1e-9);

    std::vector<float> hA, hB, h_ref;
    make_data(M, K, N, hA, hB);

    // Compute reference on GPU (via cuBLAS)
    {
      float *dA, *dB, *dC;
      CUDA_CHECK(cudaMalloc(&dA, M * K * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&dB, K * N * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dB, hB.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
      launch_cublas(dA, dB, dC, M, N, K);
      CUDA_CHECK(cudaDeviceSynchronize());
      h_ref.resize(M * N);
      CUDA_CHECK(cudaMemcpy(h_ref.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
    }

    print_hdr();
    bench("cuBLAS (TF32, baseline)", M,N,K, ITERS, hA,hB, h_ref.data(), launch_cublas);
    bench("CUTLASS SIMT (FP32)",     M,N,K, ITERS, hA,hB, h_ref.data(), launch_cutlass_sgemm);
    bench("CUTLASS TensorOp (TF32)", M,N,K, ITERS, hA,hB, h_ref.data(), launch_cutlass_tf32);
    printf("\n");
  }

  // -------------------------------------------------------------------
  // Test 2: 4096² large square
  // -------------------------------------------------------------------
  {
    int M = 4096, N = 4096, K = 4096;
    printf("--- SGEMM M=N=K=%d  (%.1f GFLOPs) ---\n",
           M, 2.0 * M * N * K * 1e-9);

    std::vector<float> hA, hB, h_ref;
    make_data(M, K, N, hA, hB);
    {
      float *dA, *dB, *dC;
      CUDA_CHECK(cudaMalloc(&dA, M * K * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&dB, K * N * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(float)));
      CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dB, hB.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
      launch_cublas(dA, dB, dC, M, N, K);
      CUDA_CHECK(cudaDeviceSynchronize());
      h_ref.resize(M * N);
      CUDA_CHECK(cudaMemcpy(h_ref.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
    }

    print_hdr();
    bench("cuBLAS (TF32)",         M,N,K, ITERS, hA,hB, h_ref.data(), launch_cublas);
    bench("CUTLASS TensorOp (TF32)", M,N,K, ITERS, hA,hB, h_ref.data(), launch_cutlass_tf32);
    printf("\n");
  }

  printf("Done.\n");
  return 0;
}
