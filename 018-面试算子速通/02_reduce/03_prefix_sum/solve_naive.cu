#include "common.cuh"

constexpr int BLOCK = 256; 

__global__ void partial_scan(const float* __restrict__ in, float* __restrict__ out, float* __restrict__ chunk_sums, int N) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    float v = (tid < N) ? in[tid] : 0.f;
    v = blockScanInclusive<BLOCK>(v);

    if (tid < N)
        out[tid] = v;
    if (threadIdx.x == blockDim.x - 1) {
        chunk_sums[blockIdx.x] = v;
    }
}

__global__ void seed_scan(float* __restrict__ chunk_sums, int n) {
    float v = (threadIdx.x < n) ? chunk_sums[threadIdx.x] : 0.f;
    v = blockScanInclusive<BLOCK>(v);
    if (threadIdx.x < n) {
        chunk_sums[threadIdx.x] = v;
    }
}

__global__ void add_seed(float* __restrict__ out, const float* __restrict__ chunk_sums, int N) {
    const float seed = blockIdx.x ? chunk_sums[blockIdx.x - 1] : 0.f;
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        out[tid] += seed;
    }
}

void solve(const float* input, float* output, int N) {
    if (N <= 0)
        return;

    const int nBlocks = CEIL(N, BLOCK);

    float* chunk_sums = nullptr;
    cudaMalloc(&chunk_sums, (size_t)nBlocks * sizeof(float));

    partial_scan<<<nBlocks, BLOCK>>>(input, output, chunk_sums, N);
    seed_scan<<<1, BLOCK>>>(chunk_sums, nBlocks);
    add_seed<<<nBlocks, BLOCK>>>(output, chunk_sums, N);

    cudaDeviceSynchronize();
    cudaFree(chunk_sums);
}
