cmake_minimum_required(VERSION 3.18)
project(cuda-conv LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "80" CACHE STRING "CUDA architectures (e.g. 80;89;90)")

find_package(CUDAToolkit REQUIRED)

# cuDNN — try conda pip path first, then system
set(CUDNN_BASE "/home/xingyongwang/miniconda3/lib/python3.12/site-packages/nvidia/cudnn")
if(EXISTS "${CUDNN_BASE}/include/cudnn.h")
  message(STATUS "Found cuDNN at ${CUDNN_BASE}")
  set(CUDNN_INCLUDE_DIR "${CUDNN_BASE}/include")
  set(CUDNN_LIBRARY "${CUDNN_BASE}/lib/libcudnn.so.9")
  set(HAS_CUDNN TRUE)
else()
  find_path(CUDNN_INCLUDE_DIR cudnn.h)
  find_library(CUDNN_LIBRARY cudnn)
  if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
    set(HAS_CUDNN TRUE)
  endif()
endif()

add_executable(conv_bench conv_bench.cu)
target_compile_options(conv_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(conv_bench PRIVATE CUDA::cublas CUDA::cudart)

if(HAS_CUDNN)
  target_include_directories(conv_bench PRIVATE ${CUDNN_INCLUDE_DIR})
  target_link_libraries(conv_bench PRIVATE ${CUDNN_LIBRARY})
  target_compile_definitions(conv_bench PRIVATE HAS_CUDNN)
  message(STATUS "cuDNN comparison enabled")
else()
  message(STATUS "cuDNN not found — skipping cuDNN comparison")
endif()

add_custom_target(run_conv
  COMMAND conv_bench
  DEPENDS conv_bench
  USES_TERMINAL)
