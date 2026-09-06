#include "common.cuh"


constexpr float ALPHA = 0.01f;

__device__ __forceinline__ float leaky(float x) { return fmaxf(x, ALPHA * x); }
__device__ __forceinline__ float4 leaky4(float4 v) { return make_float4(leaky(v.x), leaky(v.y), leaky(v.z), leaky(v.w)); }

__global__ void leaky_relu_kernel(
    const float* __restrict__ in, 
    float* __restrict__ out, 
    int N
) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    const size_t n4 = N / 4;

    const float4* in4 = reinterpret_cast<const float4*>(in);
    float4* out4 = reinterpret_cast<float4*>(out);
    for (size_t i = tid; i < n4; i += stride)
        out4[i] = leaky4(in4[i]);

    for (size_t i = n4 * 4 + tid; i < N; i += stride)
        out[i] = leaky(in[i]);
}

void solve(const float* input, float* output, int N) {
    const int threads = 256;
    int blocks = CEIL(N / 4, threads);
    if (blocks < 1)
        blocks = 1;
    if (blocks > 1024)
        blocks = 1024;
    leaky_relu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
