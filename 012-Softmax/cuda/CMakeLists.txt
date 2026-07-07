cmake_minimum_required(VERSION 3.18)
project(cuda_softmax LANGUAGES CXX CUDA)

# ---- GPU architecture --------------------------------------------------
set(CMAKE_CUDA_ARCHITECTURES 89)

# ---- Compiler flags ----------------------------------------------------
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

# L1 cache for global loads
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -dlcm=ca")

# Fast-math: __expf, __fdividef, __logf
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

# ---- Benchmark ---------------------------------------------------------
add_executable(softmax_bench bench/softmax_bench.cu)

target_compile_definitions(softmax_bench PRIVATE
    OF_SOFTMAX_USE_FAST_MATH
    CCCL_IGNORE_DEPRECATED_CPP_DIALECT
)

# ---- Convenience: build-and-run target ---------------------------------
#   cmake --build build --target run_softmax
add_custom_target(run_softmax
    COMMAND $<TARGET_FILE:softmax_bench>
    DEPENDS softmax_bench
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
