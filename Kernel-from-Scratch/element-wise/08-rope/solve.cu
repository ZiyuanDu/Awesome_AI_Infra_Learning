#include <cuda_runtime.h>
#define CEIL(x, y) (((x) + (y) - 1) / (y))

constexpr int THREADS = 256;

__device__ __forceinline__ float2 rope(float a, float b, float c, float d) {
    return make_float2(a * c - b * d, b * c + a * d);
}

__device__ __forceinline__ void rope4(float4 a, float4 b, float4 c, float4 d, float4& first, float4& second) {
    float2 x = rope(a.x, b.x, c.x, d.x);
    float2 y = rope(a.y, b.y, c.y, d.y);
    float2 z = rope(a.z, b.z, c.z, d.z);
    float2 w = rope(a.w, b.w, c.w, d.w);

    first = make_float4(x.x, y.x, z.x, w.x);
    second = make_float4(x.y, y.y, z.y, w.y);
}

__global__ void rope_kernel(const float* Q, const float* cos, const float* sin, float* output, int M, int halfD) {
    // x 处理列， y 处理行
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row_begin = blockDim.y * blockIdx.y + threadIdx.y;
    const int row_stride = gridDim.y * blockDim.y;

    // 向量化
    const int N4 = (halfD & 3) == 0 ? halfD >> 2 : 0;
    if (col >= (N4 ? N4 : halfD)) return;

    const float4* Q4 = reinterpret_cast<const float4*>(Q);
    const float4* cos4 = reinterpret_cast<const float4*>(cos);
    const float4* sin4 = reinterpret_cast<const float4*>(sin);
    float4* output4 = reinterpret_cast<float4*>(output);

    // 循环处理
    for(int row = row_begin; col < N4 && row < M; row += row_stride) {
        const int first = row * (2 * N4) + col;   
        const int second = first + N4;
        rope4(Q4[first], Q4[second], cos4[first], sin4[first], output4[first], output4[second]);
    }

    // 尾处理
    const int tail = N4 * 4 + col;
    for (int row = row_begin; tail < halfD && row < M; row += row_stride) {
        const int first = row * (2 * halfD) + tail;
        const int second = first + halfD;
        float2 v = rope(Q[first], Q[second], cos[first], sin[first]);
        output[first] = v.x;
        output[second] = v.y;
    }


}



void solve(float* Q, float* cos, float* sin, float* output, int M, int D) {

    const int haldD = D >> 1;
    if (M <= 0 || haldD <= 0) return;

    int thread_x = 1;
    while (thread_x < haldD && thread_x < 128) thread_x <<= 1;
    const dim3 threads(thread_x, THREADS / thread_x);

    int blocks_y = CEIL(M, threads.y);
    if (blocks_y > 65535) blocks_y = 65535;
    const dim3 blocks(CEIL(haldD, threads.x), blocks_y);

    rope_kernel<<<blocks, threads>>>(Q, cos, sin, output, M, haldD);


}
