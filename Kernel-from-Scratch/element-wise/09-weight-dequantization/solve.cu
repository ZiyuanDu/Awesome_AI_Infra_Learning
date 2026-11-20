#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

#define CEIL(x, y) (((x) + (y) - 1) / (y))

constexpr unsigned int MAX_GRID_ROWS = 128;
constexpr int          BLOCK_COLS    = 32;
constexpr int          BLOCK_ROWS    = 8;

__global__ void dequantize_kernel(const float* X, const float* S, float* Y, int M, int N, int TILE, int S_cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;

    const int col_tile   = col / TILE;
    const int row_begin = blockIdx.y * blockDim.y + threadIdx.y;
    const int row_stride = gridDim.y * blockDim.y;

    for (int row = row_begin; row < M; row += row_stride) {
        const int idx = row * N + col;
        const float scale = S[(row / TILE) * S_cols + col_tile];
        Y[idx] = X[idx] * scale;
    }
}

void solve(const float* X, const float* S, float* Y, int M, int N, int TILE) {
    const int S_cols = CEIL(N, TILE);
    const unsigned int gx = CEIL(N, BLOCK_COLS);
    const unsigned int gy = std::min<unsigned int>(CEIL(M, BLOCK_ROWS), MAX_GRID_ROWS);
    dequantize_kernel<<<dim3(gx, gy), dim3(BLOCK_COLS, BLOCK_ROWS)>>>(X, S, Y, M, N, TILE, S_cols);
}

