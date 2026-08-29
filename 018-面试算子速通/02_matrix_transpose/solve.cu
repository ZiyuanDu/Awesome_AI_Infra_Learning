#include "solve.h"

constexpr int TILE = 16;


__global__ void matrix_transpose_kernel(const float* __restrict__ in, float* __restrict__ out,
                                        int rows, int cols) {
    // 共享内存，避免访存冲突
    __shared__ float s[TILE][TILE + 1];

    // 当前线程在 block 内的 x 方向索引，由于每个线程负责一个 float4，所以是 threadIdx.x * 4，范围是 0~3
    const int tx = threadIdx.x; 
    // 当前线程在 block 内的 y 方向索引，范围是 0~15
    const int ty = threadIdx.y; 

    // 加载 in 矩阵的 float4 元素
    float4 v = load4(in, cols, blockIdx.y * TILE + ty, blockIdx.x * TILE + tx * 4, rows, cols);

    // 共享内存和 in 保持一致，写入时是：s[行][列]
    s[ty][tx * 4 + 0] = v.x;
    s[ty][tx * 4 + 1] = v.y;
    s[ty][tx * 4 + 2] = v.z;
    s[ty][tx * 4 + 3] = v.w;
    __syncthreads()


    v = make_float4(
        s[tx * 4 + 0][ty], 
        s[tx * 4 + 1][ty], 
        s[tx * 4 + 2][ty], 
        s[tx * 4 + 3][ty]
    );
    store4(out, rows, blockIdx.x * TILE + ty, blockIdx.y * TILE + tx * 4, v, cols, rows);
}

void solve(const float* input, float* output, int rows, int cols) {
    dim3 threads(TILE / 4, TILE);
    dim3 blocks(CEIL(cols, TILE), CEIL(rows, TILE));
    matrix_transpose_kernel<<<blocks, threads>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
