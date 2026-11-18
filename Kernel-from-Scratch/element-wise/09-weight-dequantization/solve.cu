#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

#define CEIL(x, y) (((x) + (y) - 1) / (y))

constexpr unsigned int MAX_GRID_ROWS = 128;

__device__ __forceinline__ float4 dequantize4(float4 x, float scale) {
    return make_float4(x.x * scale, x.y * scale, x.z * scale, x.w * scale);
}

__global__ void dequantize_kernel(const float* X, const float* S, float* Y,
                                  int M, int N, int T, int scale_cols, bool vectorized) {
    const int width = vectorized ? 4 : 1;
    const int col = (blockIdx.x * blockDim.x + threadIdx.x) * width;
    if (col >= N) return;

    const int row_begin = blockIdx.y * blockDim.y + threadIdx.y;
    const int row_stride = gridDim.y * blockDim.y;
    const int scale_col = col / T;
    for (int row = row_begin; row < M; row += row_stride) {
        const int i = row * N + col;
        const float scale = S[(row / T) * scale_cols + scale_col];
        if (vectorized) {
            const auto* x4 = reinterpret_cast<const float4*>(X + i);
            auto* y4 = reinterpret_cast<float4*>(Y + i);
            *y4 = dequantize4(*x4, scale);
        } else {
            Y[i] = X[i] * scale;
        }
    }
}

void solve(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE) {
    if (M <= 0 || N <= 0 || TILE_SIZE <= 0) return;

    const auto addresses = reinterpret_cast<std::uintptr_t>(X) | reinterpret_cast<std::uintptr_t>(Y);
    const bool vectorized = N % 4 == 0 && TILE_SIZE % 4 == 0 && addresses % alignof(float4) == 0;
    const int cols = vectorized ? N / 4 : N;
    const int scale_cols = CEIL(N, TILE_SIZE);

    const dim3 threads(32, 8);
    const dim3 blocks(CEIL(cols, threads.x), std::min(CEIL(M, threads.y), MAX_GRID_ROWS));
    dequantize_kernel<<<blocks, threads>>>(X, S, Y, M, N, TILE_SIZE, scale_cols, vectorized);
}
