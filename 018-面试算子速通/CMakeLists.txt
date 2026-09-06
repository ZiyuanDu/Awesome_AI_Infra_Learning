cmake_minimum_required(VERSION 3.18)
project(leetgpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES "120" CACHE STRING "native GPU arch")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

function(leetgpu_bench dir)
    string(REPLACE "/" "_" tgt "${dir}")
    add_executable(${tgt} ${dir}/solve.cu ${dir}/bench.cu)
    target_include_directories(${tgt} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    )
    target_link_libraries(${tgt} PRIVATE CUDA::cudart)
    set_target_properties(${tgt} PROPERTIES
        OUTPUT_NAME bench
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${dir}
        CUDA_SEPARABLE_COMPILATION OFF
    )
endfunction()

leetgpu_bench(element_wise/01_vector_add)
leetgpu_bench(element_wise/02_color_inversion)
leetgpu_bench(element_wise/03_relu)
leetgpu_bench(element_wise/04_leaky_relu)
leetgpu_bench(element_wise/05_matrix_copy)
leetgpu_bench(element_wise/06_silu)
leetgpu_bench(element_wise/07_swiglu)
leetgpu_bench(element_wise/08_value_clipping)
leetgpu_bench(element_wise/09_rgb_to_grayscale)
leetgpu_bench(element_wise/10_sigmoid)
leetgpu_bench(permute/01_reverse_array)
leetgpu_bench(permute/02_matrix_transpose)
leetgpu_bench(permute/03_interleave_arrays)
leetgpu_bench(gemm/01_matrix_mul)
leetgpu_bench(gemm/02_spmv)
leetgpu_bench(gemm/03_gemm)
leetgpu_bench(stencil/01_conv1d)
leetgpu_bench(stencil/02_conv2d)
leetgpu_bench(stencil/03_conv3d)
leetgpu_bench(reduce/01_reduction)
leetgpu_bench(reduce/02_softmax)
leetgpu_bench(reduce/03_prefix_sum)
leetgpu_bench(reduce/04_dot_product)
leetgpu_bench(attention/01_naive)
leetgpu_bench(attention/02_online)
leetgpu_bench(attention/03_flash_v2)
