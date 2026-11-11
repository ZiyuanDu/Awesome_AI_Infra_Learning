#include "common.cuh"

// 数组原地逆序：in[i] <-> in[N-1-i]，只需遍历前 N/2 个元素。

constexpr int THREADS = 256;
constexpr int MAX_BLOCKS = 1024;

__global__ void reverse_kernel(float* __restrict__ in, int N) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    // 只处理前一半，另一半通过对称位置交换完成。
    for (int i = tid; i < N / 2; i += stride) {
        float t = in[i];
        in[i] = in[N - 1 - i];
        in[N - 1 - i] = t;
    }
}

void solve(float* input, int N) {
    int blocks = CEIL(N / 2, THREADS);
    if (blocks < 1)
        blocks = 1;
    if (blocks > MAX_BLOCKS)
        blocks = MAX_BLOCKS;
    reverse_kernel<<<blocks, THREADS>>>(input, N);
    cudaDeviceSynchronize();
}
