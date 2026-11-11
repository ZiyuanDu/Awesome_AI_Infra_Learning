#include <cuda_runtime.h>

#define CEIL(x, y) (((x) + (y) - 1) / (y))

constexpr float ALPHA = 0.01f;
constexpr int THREADS = 256;

__device__ __forceinline__ float leaky(float x) {
    return fmaxf(x, ALPHA * x);
}

__device__ __forceinline__ float4 leaky4(float4 v) {
    return make_float4(leaky(v.x), leaky(v.y), leaky(v.z), leaky(v.w));
}


__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    size_t N4 = (size_t) N >> 2;

    const float4* input4= reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);

    for(size_t i = tid; i < N4; i += stride) {
        output4[i] = leaky4(input4[i]);
    }

    // 尾处理
    for(size_t i = N4 * 4 + tid; i < N; i += stride) {
        output[i] = leaky(input[i]);
    }
}


void solve(const float* input, float* output, int N) {
    int N4 = N >> 2;
    int blocks = CEIL(N4, THREADS);
    leaky_relu_kernel<<<blocks, THREADS>>>(input, output, N);
}
