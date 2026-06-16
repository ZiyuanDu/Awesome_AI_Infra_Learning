cmake_minimum_required(VERSION 3.18)
project(cutlass-sgemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "80" CACHE STRING "CUDA architectures")

find_package(CUDAToolkit REQUIRED)

# ---- CUTLASS include path ----
# Set CUTLASS_DIR to your CUTLASS checkout, or it will try common locations.
if(NOT DEFINED CUTLASS_DIR)
  set(CUTLASS_DIR "" CACHE PATH "Path to CUTLASS repository root")
endif()

if(CUTLASS_DIR)
  set(CUTLASS_INCLUDE_DIR ${CUTLASS_DIR}/include)
else()
  # Try common locations
  foreach(dir ${CMAKE_SOURCE_DIR}/../cutlass
              ${CMAKE_SOURCE_DIR}/../../cutlass
              $ENV{HOME}/cutlass
              /opt/cutlass)
    if(EXISTS ${dir}/include/cutlass/cutlass.h)
      set(CUTLASS_INCLUDE_DIR ${dir}/include)
      message(STATUS "Found CUTLASS at ${dir}")
      break()
    endif()
  endforeach()
endif()

if(NOT CUTLASS_INCLUDE_DIR)
  message(FATAL_ERROR
    "CUTLASS not found. Please set CUTLASS_DIR to your CUTLASS checkout.\n"
    "  git clone https://github.com/NVIDIA/cutlass.git\n"
    "  cmake -DCUTLASS_DIR=/path/to/cutlass ...")
endif()

add_executable(cutlass_sgemm cutlass_sgemm.cu)
target_include_directories(cutlass_sgemm PRIVATE ${CUTLASS_INCLUDE_DIR})
target_compile_options(cutlass_sgemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(cutlass_sgemm PRIVATE CUDA::cublas CUDA::cudart)

add_custom_target(run_cutlass
  COMMAND cutlass_sgemm
  DEPENDS cutlass_sgemm
  USES_TERMINAL)
