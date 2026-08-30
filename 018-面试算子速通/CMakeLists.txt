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
    add_executable(${dir} ${dir}/solve.cu ${dir}/bench.cu)
    target_include_directories(${dir} PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/${dir}
    )
    target_link_libraries(${dir} PRIVATE CUDA::cudart)
    set_target_properties(${dir} PROPERTIES
        OUTPUT_NAME bench
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${dir}
        CUDA_SEPARABLE_COMPILATION OFF
    )
endfunction()

leetgpu_bench(01_vector_add)
leetgpu_bench(02_matrix_transpose)
leetgpu_bench(03_matrix_mul)
leetgpu_bench(04_reverse_array)
