#include <cuda_runtime.h>

#define CEIL(x, y) (((x) + (y) - 1) / (y))

constexpr int THREADS = 256;

__device__ __forceinline__ float silu(float x) {
    return (1 / (1 + __expf(-x))) * x;
}

__device__ __forceinline__ float4 swiglu4(float4 v, float4 u) {
    return make_float4(silu(v.x) * u.x, silu(v.y) * u.y, silu(v.z) * u.z, silu(v.w) * u.w);
}


__global__ void swiglu_kernel(const float* input, float* output, int N) {
    // 下标
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    // 向量化
    size_t halfN = (size_t) N >> 1;
    size_t N4 = (halfN & 3) == 0 ? halfN >> 2 : 0;

    // 要处理两次
    const float4* input4 = reinterpret_cast<const float4*>(input);
    const float4* value4 = reinterpret_cast<const float4*>(input + halfN);
    float4* output4 = reinterpret_cast<float4*>(output);

    // 循环
    for(size_t i = tid; i < N4; i += stride) {
        output4[i] = swiglu4(input4[i], value4[i]);
    }

    // 尾处理
    for(size_t i = N4 * 4 + tid; i < halfN; i += stride) {
        output[i] = silu(input[i]) * input[i + halfN];
    }
}



void solve(const float* input, float* output, int N) {
    int halfN = N >> 1;
    if (halfN <= 0) return;

    int count = (halfN & 3) == 0 ? halfN >> 2 : halfN;
    int blocks = CEIL(count, THREADS);

    swiglu_kernel<<<blocks, THREADS>>>(input, output, N);

}
