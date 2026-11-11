#include <cuda_runtime.h>
#define CEIL(x, y) (((x) + (y) - 1) / (y));
const int THREADS = 256;

__device__ __forceinline__ float4 add4(float4 a, float4 b) {
    return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}


__global__ void vector_add_kernel(const float* A, const float* B, float* C, int N) {
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    size_t n4 = (size_t) N >> 2;

    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);

    for (size_t i = tid; i < n4; i += stride) {
        C4[i] = add4(A4[i], B4[i]);
    }


    for (size_t i = n4 * 4 + tid; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }

}


void solve(const float* A, const float* B, float* C, int N) {
    if (N <= 0) {
        return;
    }
    int n4 = (size_t) N >> 2;
    int block = CEIL(n4, THREADS);

    vector_add_kernel<<<block, THREADS>>>(A, B, C, N);
}
