#include "common.cuh"


__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

__global__ void swiglu_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int n4 = (n & 3) ? 0 : n / 4;

    const float4* in4 = reinterpret_cast<const float4*>(in);
    float4* out4 = reinterpret_cast<float4*>(out);
    for (int i = tid; i < n4; i += stride) {
        float4 a = in4[i];
        float4 b = in4[i + n4];
        out4[i] = make_float4(silu(a.x) * b.x, silu(a.y) * b.y, silu(a.z) * b.z, silu(a.w) * b.w);
    }
    for (int i = n4 * 4 + tid; i < n; i += stride)
        out[i] = silu(in[i]) * in[i + n];
}

void solve(const float* input, float* output, int N) {
    const int n = N >> 1;
    const int threads = 256;
    int blocks = CEIL(n / 4, threads);
    if (blocks < 1)
        blocks = 1;
    swiglu_kernel<<<blocks, threads>>>(input, output, n);
    cudaDeviceSynchronize();
}
