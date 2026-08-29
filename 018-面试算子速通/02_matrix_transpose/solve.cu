#include "solve.h"

constexpr int TILE = 16;
constexpr int VEC_SIZE = 4;


__global__ void matrix_transpose_kernel_naive(const float* __restrict__ in, float* __restrict__ out,
                                            int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows || col >= cols) return;

    int in_idx = row * cols + col;
    int out_idx = col * rows + row;

    out[out_idx] = in[in_idx];

}


__global__ void matrix_transpose_kernel(const float* __restrict__ in, float* __restrict__ out,
                                        int rows, int cols) {
    // 共享内存，避免访存冲突
    __shared__ float s[TILE][TILE + 1];

    const int local_row = threadIdx.y;
    const int col_group = threadIdx.x;

    const int local_col_start = col_group * VEC_SIZE;

    // 输入矩阵的Tile起始位置
    const int in_row_start = blockIdx.y * TILE;
    const int in_col_start = blockIdx.x * TILE;

    // 输出矩阵的Tile起始位置
    const int out_row_start = blockIdx.x * TILE;
    const int out_col_start = blockIdx.y * TILE;

    // 全局绝对坐标
    const int in_row = in_row_start + local_row;
    const int in_col = in_col_start + local_col_start;

    float4 v = load4(in, cols, in_row, in_col, rows, cols);

    // 写入共享内存
    s[local_row][local_col_start + 0] = v.x;
    s[local_row][local_col_start + 1] = v.y;
    s[local_row][local_col_start + 2] = v.z;
    s[local_row][local_col_start + 3] = v.w;
    __syncthreads();

    // 输出矩阵的全局绝对坐标
    const int out_row = out_row_start + local_row;
    const int out_col = out_col_start + local_col_start;

    // 转置读取共享内存
    v = make_float4(
        s[local_col_start + 0][local_row], 
        s[local_col_start + 1][local_row], 
        s[local_col_start + 2][local_row], 
        s[local_col_start + 3][local_row]
    );
    store4(out, rows, out_row, out_col, v, cols, rows);


}

void solve(const float* input, float* output, int rows, int cols) {
    dim3 threads(TILE / VEC_SIZE, TILE);
    dim3 blocks(CEIL(cols, TILE), CEIL(rows, TILE));
    // matrix_transpose_kernel_naive<<<blocks, threads>>>(input, output, rows, cols);
    matrix_transpose_kernel<<<blocks, threads>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
