#include "common.cuh"

// 矩阵转置：out[col, row] = in[row, col]。
// 使用 TILE×TILE shared memory tile，读入转置方向后写出，合并访存。

constexpr int TILE = 16;

__global__ void matrix_transpose_kernel(const float* __restrict__ in, float* __restrict__ out,
                                        int rows, int cols) {
    // 行距 +1 避免 shared memory bank conflict。
    __shared__ float s[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < cols && y < rows)
        s[threadIdx.y][threadIdx.x] = in[y * cols + x];
    __syncthreads();

    // 读入 tile 后按转置索引写回，保证 global memory 合并访问。
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    if (x < rows && y < cols)
        out[y * rows + x] = s[threadIdx.x][threadIdx.y];
}

void solve(const float* input, float* output, int rows, int cols) {
    dim3 threads(TILE, TILE);
    dim3 blocks(CEIL(cols, TILE), CEIL(rows, TILE));
    matrix_transpose_kernel<<<blocks, threads>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
