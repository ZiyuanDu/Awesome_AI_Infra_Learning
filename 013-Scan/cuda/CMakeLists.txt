cmake_minimum_required(VERSION 3.18)
project(cuda_scan LANGUAGES CXX CUDA)

# ---- GPU architecture --------------------------------------------------
# Ada Lovelace: sm_89 (RTX 4090D, 4070 SUPER)
set(CMAKE_CUDA_ARCHITECTURES 89)

# ---- Compiler flags ----------------------------------------------------
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

# Let the L1 cache serve global loads (128-byte cache lines).
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -dlcm=ca")

# Fast-math: fma contraction, etc.
# (comment out if you need strict IEEE compliance)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

# ---- Benchmarks --------------------------------------------------------
# Single benchmark executable that tests all scan variants

add_executable(scan_bench bench/scan_bench.cu)

# All sources are header-only (include ../*.cuh), no extra link deps.

# ---- Convenience: build-and-run targets --------------------------------
#   cmake --build build --target run_scan
add_custom_target(run_scan
    COMMAND $<TARGET_FILE:scan_bench>
    DEPENDS scan_bench
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
