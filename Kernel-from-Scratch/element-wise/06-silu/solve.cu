#include <cuda_runtime.h>

#define CEIL(x, y) (((x) + (y) - 1) / (y))

constexpr int THREADS = 256;

__device__ __forceinline__ float silu(float x) {
    return (1 / (1 + __expf(-x))) * x;
}

__device__ __forceinline__ float4 silu4(float4 v) {
    return make_float4(silu(v.x), silu(v.y), silu(v.z), silu(v.w));
}

__global__ void silu_kernel(const float* input, float* output, int N) {
    // 下标
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    // 向量化
    size_t N4 = (size_t) N >> 2;
    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);

    // 循环
    for(size_t i = tid; i < N4; i += stride) {
        // sigmoid
        output4[i] = silu4(input4[i]);
    }

    // 尾处理
    for(size_t i = N4 * 4 + tid; i < N; i += stride) {
        output[i] = silu(input[i]);
    }
}



void solve(const float* input, float* output, int N) {
    int N4 = N >> 2;
    int blocks = CEIL(N4, THREADS);

    silu_kernel<<<blocks, THREADS>>>(input, output, N);
}
