#include <cuda_runtime.h>
#define CEIL(x, y) (((x) + (y) - 1) / (y))
const int THREADS = 256;


__device__ __forceinline__ float4 relu4(float4 v) {
    return make_float4(fmaxf(v.x, 0.f), fmaxf(v.y, 0.f), fmaxf(v.z, 0.f), fmaxf(v.w, 0.f));
}

__global__ void relu_kernel(const float* input, float* output, int N) {
    // 坐标
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    // 向量化
    size_t n4 = (size_t) N >> 2;
    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);

    // 循环
    for(size_t i = tid; i < n4; i += stride) {
        output4[i] = relu4(input4[i]);
    }
    // 尾处理
    for(size_t i = n4 * 4 + tid; i < N; i += stride) {
        output[i] = fmaxf(input[i], 0.f);
    }
}


void solve(const float* input, float* output, int N) {
    int n4 = N >> 2;
    int blocks = CEIL(n4, THREADS);
    relu_kernel<<<blocks, THREADS>>>(input, output, N);
}
