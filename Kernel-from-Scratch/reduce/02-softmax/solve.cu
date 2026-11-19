#include <cuda_runtime.h>
#include <cstdint>
#include "../reduce.cuh"
#define CEIL(x, y) (((x) + (y) - 1) / (y))
constexpr int THREADS = 128, ITEMS = 32, TILE = THREADS * ITEMS;
constexpr int SINGLE_LIMIT = 512 * ITEMS;

__device__ __forceinline__ float4 load4(const float* x, int i, int N) {
    bool full = i + 3 < N;
    if (full && reinterpret_cast<uintptr_t>(x + i) % alignof(float4) == 0)
        return *reinterpret_cast<const float4*>(x + i);
    return make_float4(i < N ? x[i] : -CUDART_INF_F, i+1 < N ? x[i+1] : -CUDART_INF_F,
                      i+2 < N ? x[i+2] : -CUDART_INF_F, i+3 < N ? x[i+3] : -CUDART_INF_F);
}
__device__ __forceinline__ void store4(float* y, int i, int N, float4 v) {
    bool full = i + 3 < N;
    if (full && reinterpret_cast<uintptr_t>(y + i) % alignof(float4) == 0)
        *reinterpret_cast<float4*>(y + i) = v;
    else {
        if (i < N) y[i] = v.x;
        if (i+1 < N) y[i+1] = v.y;
        if (i+2 < N) y[i+2] = v.z;
        if (i+3 < N) y[i+3] = v.w;
    }
}
__device__ __forceinline__ float4 exp4(float4 v, float m, float scale) {
    return make_float4(__expf(v.x-m)*scale, __expf(v.y-m)*scale,
                      __expf(v.z-m)*scale, __expf(v.w-m)*scale);
}

__global__ void tile_stats(const float* input, float* output, float2* partial, int N) {
    float4 values[ITEMS/4], m = make_float4(-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F);
    float4 sum = make_float4(0, 0, 0, 0);
    int start = blockIdx.x * blockDim.x * ITEMS + 4 * threadIdx.x;
    #pragma unroll
    for (int k = 0; k < ITEMS/4; ++k) {
        float4 v = values[k] = load4(input, start + 4*k*blockDim.x, N);
        m.x = fmaxf(m.x, v.x); m.y = fmaxf(m.y, v.y);
        m.z = fmaxf(m.z, v.z); m.w = fmaxf(m.w, v.w);
    }
    float maximum = block_max(fmaxf(fmaxf(m.x, m.y), fmaxf(m.z, m.w)));
    #pragma unroll
    for (int k = 0; k < ITEMS/4; ++k) {
        float4 v = values[k] = exp4(values[k], maximum, 1.0f);
        sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
    }
    float total = block_sum((sum.x + sum.y) + (sum.z + sum.w));
    if (gridDim.x == 1) {
        float inverse = 1.0f / total;
        #pragma unroll
        for (int k = 0; k < ITEMS/4; ++k) {
            float4 v = values[k];
            store4(output, start + 4*k*blockDim.x, N,
                   make_float4(v.x*inverse, v.y*inverse, v.z*inverse, v.w*inverse));
        }
    } else if (threadIdx.x == 0) partial[blockIdx.x] = make_float2(maximum, total);
}
__global__ void normalize(const float* input, float* output, const float2* partial, int N, int tiles) {
    int tid = threadIdx.x;
    float2 s = tid < tiles ? partial[tid] : make_float2(-CUDART_INF_F, 0.0f);
    float maximum = block_max(s.x);
    float inverse = 1.0f / block_sum(s.y * __expf(s.x - maximum));
    #pragma unroll
    for (int k = 0; k < ITEMS/4; ++k) {
        int i = blockIdx.x*TILE + 4*tid + 4*k*THREADS;
        store4(output, i, N, exp4(load4(input, i, N), maximum, inverse));
    }
}
// The exercise guarantees 1 <= N <= 500000; launches use the default stream.
void solve(const float* input, float* output, int N) {
    int threads = THREADS, tiles = CEIL(N, TILE);
    if (N <= SINGLE_LIMIT) {
        threads = 32;
        while (threads * ITEMS < N) threads *= 2;
        tiles = 1;
    }
    float2* partial = nullptr;
    if (tiles > 1) {
        struct Scratch {
            float2* data = nullptr;
            Scratch() { cudaMalloc(&data, CEIL(500000, TILE) * sizeof(float2)); }
            ~Scratch() { cudaFree(data); }
        };
        static thread_local Scratch scratch;
        partial = scratch.data;
    }
    tile_stats<<<tiles, threads>>>(input, output, partial, N);
    if (tiles > 1) normalize<<<tiles, THREADS>>>(input, output, partial, N, tiles);
}
