#include "common.cuh"

constexpr int TILE = 16;

__global__ void matrix_transpose_kernel(const float* __restrict__ in, float* __restrict__ out,
                                        int rows, int cols) {
    __shared__ float s[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < cols && y < rows)
        s[threadIdx.y][threadIdx.x] = in[y * cols + x];
    __syncthreads();

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
