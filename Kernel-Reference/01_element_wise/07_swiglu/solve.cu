#include "common.cuh"

// SwiGLU 门控：out[i] = silu(x1[i]) * x2[i]。
// 输入前半段是 x1，后半段是 x2；输入长度 N = 2n。

constexpr int THREADS = 256;
constexpr int MAX_BLOCKS = 2048;

__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

__device__ __forceinline__ float4 swiglu4(float4 a, float4 b) {
    return make_float4(silu(a.x) * b.x, silu(a.y) * b.y, silu(a.z) * b.z, silu(a.w) * b.w);
}

__global__ void swiglu_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;

    if ((n & 3) == 0) {
        const size_t n4 = n / 4;
        const float4* x1 = reinterpret_cast<const float4*>(in);
        const float4* x2 = reinterpret_cast<const float4*>(in + n);
        float4* o4 = reinterpret_cast<float4*>(out);

        // 4 对齐时走 float4。
        for (size_t i = tid; i < n4; i += stride) {
            o4[i] = swiglu4(__ldg(&x1[i]), __ldg(&x2[i]));
        }
    } else {
        // 非 4 对齐时走标量，逻辑等价。
        for (size_t i = tid; i < n; i += stride) {
            out[i] = silu(__ldg(in + i)) * __ldg(in + i + n);
        }
    }
}

void solve(const float* input, float* output, int N) {
    const int n = N >> 1;
    // enough for both float4 (n/4) and scalar (n) via grid-stride
    int blocks = CEIL(n, THREADS);
    if (blocks < 1)
        blocks = 1;
    if (blocks > MAX_BLOCKS)
        blocks = MAX_BLOCKS;
    swiglu_kernel<<<blocks, THREADS>>>(input, output, n);
    cudaDeviceSynchronize();
}
