#include "common.cuh"

/*
 * 1D valid：out[i] = Σ_k in[i+k] * ker[k]
 *
 * 一个 block 算 BLOCK 个输出；输入 halo 进 smem，核进 constant（广播读）。
 */
constexpr int BLOCK = 256;
constexpr int KMAX = 2047;

__constant__ float c_ker[KMAX];

__global__ void conv1d_kernel(const float* __restrict__ in, float* __restrict__ out, int N, int K) {
    extern __shared__ float sIn[];
    const int i0 = blockIdx.x * BLOCK;
    const int tx = threadIdx.x;
    const int inTile = BLOCK + K - 1;

    for (int t = tx; t < inTile; t += BLOCK)
        sIn[t] = (i0 + t < N) ? in[i0 + t] : 0.f;
    __syncthreads();

    const int i = i0 + tx;
    if (i >= N - K + 1)
        return;

    float acc = 0.f;
    int k = 0;
    for (; k + 3 < K; k += 4) {
        acc += sIn[tx + k] * c_ker[k] + sIn[tx + k + 1] * c_ker[k + 1] +
               sIn[tx + k + 2] * c_ker[k + 2] + sIn[tx + k + 3] * c_ker[k + 3];
    }
    for (; k < K; ++k)
        acc += sIn[tx + k] * c_ker[k];
    out[i] = acc;
}

void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    const int nOut = input_size - kernel_size + 1;
    if (nOut <= 0)
        return;
    cudaMemcpyToSymbol(c_ker, kernel, kernel_size * sizeof(float));
    const int smem = (BLOCK + kernel_size - 1) * (int)sizeof(float);
    conv1d_kernel<<<CEIL(nOut, BLOCK), BLOCK, smem>>>(input, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}
