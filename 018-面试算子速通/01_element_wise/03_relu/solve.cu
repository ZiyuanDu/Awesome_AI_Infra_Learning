#include "common.cuh"

// ReLU：out[i] = max(in[i], 0)。
// float4 主循环 + 标量尾循环，纯访存/比较型。

constexpr int THREADS = 256;
constexpr int MAX_BLOCKS = 1024;

__device__ __forceinline__ float4 relu4(float4 v) {
    return make_float4(fmaxf(v.x, 0.f), fmaxf(v.y, 0.f), fmaxf(v.z, 0.f), fmaxf(v.w, 0.f));
}


__global__ void relu_kernel(
    const float* __restrict__ in, 
    float* __restrict__ out, 
    int N
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int n4 = N / 4;

    const float4* in4 = reinterpret_cast<const float4*>(in);
    float4* out4 = reinterpret_cast<float4*>(out);
    // 向量化主循环。
    for (int i = tid; i < n4; i += stride)
        out4[i] = relu4(in4[i]);

    // 标量尾循环。
    for (int i = n4 * 4 + tid; i < N; i += stride)
        out[i] = fmaxf(in[i], 0.f);
}

void solve(const float* input, float* output, int N) {
    int blocks = CEIL(N / 4, THREADS);
    if (blocks < 1)
        blocks = 1;
    if (blocks > MAX_BLOCKS)
        blocks = MAX_BLOCKS;
    relu_kernel<<<blocks, THREADS>>>(input, output, N);
    cudaDeviceSynchronize();
}
