#include "common.cuh"

// Sigmoid：1 / (1 + exp(-x))。

constexpr int THREADS = 256;
constexpr int MAX_BLOCKS = 1024;

__device__ __forceinline__ float sigmoid(float x) { return 1.f / (1.f + __expf(-x)); }

__device__ __forceinline__ float4 sigmoid4(float4 v) {
    return make_float4(sigmoid(v.x), sigmoid(v.y), sigmoid(v.z), sigmoid(v.w));
}

__global__ void sigmoid_kernel(const float* __restrict__ in, float* __restrict__ out, int N) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    const size_t n4 = (size_t)N / 4;

    const float4* in4 = reinterpret_cast<const float4*>(in);
    float4* out4 = reinterpret_cast<float4*>(out);

    // 向量化主循环。
    for (size_t i = tid; i < n4; i += stride) {
        out4[i] = sigmoid4(in4[i]);
    }

    // 标量尾循环。
    for (size_t i = n4 * 4 + tid; i < N; i += stride) {
        out[i] = sigmoid(in[i]);
    }
}

void solve(const float* input, float* output, int N) {
    int blocks = CEIL(N / 4, THREADS);
    if (blocks < 1)
        blocks = 1;
    if (blocks > MAX_BLOCKS)
        blocks = MAX_BLOCKS;
    sigmoid_kernel<<<blocks, THREADS>>>(input, output, N);
    cudaDeviceSynchronize();
}
