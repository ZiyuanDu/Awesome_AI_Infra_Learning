cmake_minimum_required(VERSION 3.18)
project(cuda-sgemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "80" CACHE STRING "CUDA architectures (e.g. 80;89;90)")

find_package(CUDAToolkit REQUIRED)

# ================================================================
# matmul_bench — 性能对比可执行文件
# ================================================================
add_executable(matmul_bench matmul_bench.cu)
target_compile_options(matmul_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(matmul_bench PRIVATE CUDA::cublas CUDA::cudart)

add_custom_target(run_matmul
  COMMAND matmul_bench
  DEPENDS matmul_bench
  USES_TERMINAL)



add_custom_target(tune
  COMMAND tune_matmul 1024 1024 1024 ${CMAKE_BINARY_DIR}/tune_results.txt
  DEPENDS tune_matmul
  USES_TERMINAL
  COMMENT "Auto-tuning matmul kernels (M=N=K=1024)...")
