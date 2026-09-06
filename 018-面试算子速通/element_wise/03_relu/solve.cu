#include "common.cuh"

__device__ __forceinline__ float4 relu4(float4 v) {
    return make_float4(fmaxf(v.x, 0.f), fmaxf(v.y, 0.f), fmaxf(v.z, 0.f), fmaxf(v.w, 0.f));
}


__global__ void relu_kernel(const float* __restrict__ in, float* __restrict__ out, int N) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int n4 = N / 4;

    const float4* in4 = reinterpret_cast<const float4*>(in);
    float4* out4 = reinterpret_cast<float4*>(out);
    for (int i = tid; i < n4; i += stride)
        out4[i] = relu4(in4[i]);

    for (int i = n4 * 4 + tid; i < N; i += stride)
        out[i] = fmaxf(in[i], 0.f);
}

void solve(const float* input, float* output, int N) {
    const int threads = 256;
    int blocks = CEIL(N / 4, threads);
    if (blocks < 1)
        blocks = 1;
    if (blocks > 1024)
        blocks = 1024;
    relu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
