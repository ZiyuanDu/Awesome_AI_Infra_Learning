#include "common.cuh"

__global__ void reverse_kernel(float* __restrict__ in, int N) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N / 2; i += stride) {
        float t = in[i];
        in[i] = in[N - 1 - i];
        in[N - 1 - i] = t;
    }
}

void solve(float* input, int N) {
    const int threads = 256;
    int blocks = CEIL(N / 2, threads);
    if (blocks < 1)
        blocks = 1;
    if (blocks > 1024)
        blocks = 1024;
    reverse_kernel<<<blocks, threads>>>(input, N);
    cudaDeviceSynchronize();
}
