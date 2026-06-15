cmake_minimum_required(VERSION 3.18)
project(cuda_norm LANGUAGES CXX CUDA)

# ---- GPU architecture --------------------------------------------------
# Ada Lovelace: sm_89 (RTX 4090D, 4070 SUPER)
set(CMAKE_CUDA_ARCHITECTURES 89)

# ---- Compiler flags ----------------------------------------------------
# -O3 for host code; CUDA flags tuned for memory-bound norm kernels
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

# Let the L1 cache serve global loads (128-byte cache lines).
# Norm kernels re-read data → high L1 hit rate is critical for perf.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -dlcm=ca")

# Fast-math: rsqrt → __frsqrt_rn, fma contraction, etc.
# (comment out if you need strict IEEE compliance)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

# ---- Benchmarks --------------------------------------------------------
# Each benchmark is a standalone executable; build them all.

add_executable(reduce_bench bench/reduce_bench.cu)
add_executable(io_bench    bench/io_bench.cu)
add_executable(rms_norm    bench/rms_norm_bench.cu)
add_executable(layer_norm  bench/layer_norm_bench.cu)

# All benchmarks are header-only (include ../*.cuh), no extra link deps.

# ---- Convenience: build-and-run target for each benchmark ---------------
#   cmake --build build --target run_reduce
#   cmake --build build --target run_io
#   cmake --build build --target run_rms
#   cmake --build build --target run_layer
#   cmake --build build --target run_all       (runs everything)

add_custom_target(run_reduce
    COMMAND $<TARGET_FILE:reduce_bench>
    DEPENDS reduce_bench
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_custom_target(run_io
    COMMAND $<TARGET_FILE:io_bench>
    DEPENDS io_bench
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_custom_target(run_rms
    COMMAND $<TARGET_FILE:rms_norm>
    DEPENDS rms_norm
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_custom_target(run_layer
    COMMAND $<TARGET_FILE:layer_norm>
    DEPENDS layer_norm
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_custom_target(run_all
    COMMAND $<TARGET_FILE:reduce_bench>
    COMMAND $<TARGET_FILE:io_bench>
    COMMAND $<TARGET_FILE:rms_norm>
    COMMAND $<TARGET_FILE:layer_norm>
    DEPENDS reduce_bench io_bench rms_norm layer_norm
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
