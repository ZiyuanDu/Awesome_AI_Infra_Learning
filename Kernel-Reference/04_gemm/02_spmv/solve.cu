#include "common.cuh"

// SpMV：y = A * x。
// 输入是 M×N row-major 稠密存储，`nnz` 由测试传入但当前实现不跳零。
// 策略：x 能放进 smem 时复用 x；否则按行用 float4 或标量路径。

constexpr int BLOCK = 256;

__global__ void gemv_f4_smem(const float* __restrict__ A, const float* __restrict__ x,
                             float* __restrict__ y, int M, int N) {
    extern __shared__ float sx[];
    const int tid = threadIdx.x;
    const int n4 = N >> 2;
    float4* sx4 = reinterpret_cast<float4*>(sx);

    for (int j = tid; j < n4; j += BLOCK)
        sx4[j] = __ldg(reinterpret_cast<const float4*>(x) + j);
    __syncthreads();

    for (int row = blockIdx.x; row < M; row += gridDim.x) {
        const float4* a4 = reinterpret_cast<const float4*>(A + (size_t)row * N);
        float partial = 0.f;
#pragma unroll 4
        for (int j = tid; j < n4; j += BLOCK) {
            float4 a = __ldg(a4 + j);
            float4 b = sx4[j];
            partial += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
        partial = blockReduceSum<BLOCK>(partial);
        if (tid == 0)
            y[row] = partial;
        __syncthreads();
    }
}

__global__ void gemv_scalar_smem(const float* __restrict__ A, const float* __restrict__ x,
                                 float* __restrict__ y, int M, int N) {
    extern __shared__ float sx[];
    const int tid = threadIdx.x;
    for (int j = tid; j < N; j += BLOCK)
        sx[j] = __ldg(x + j);
    __syncthreads();

    for (int row = blockIdx.x; row < M; row += gridDim.x) {
        const float* rowA = A + (size_t)row * N;
        float partial = 0.f;
        for (int j = tid; j < N; j += BLOCK)
            partial += __ldg(rowA + j) * sx[j];
        partial = blockReduceSum<BLOCK>(partial);
        if (tid == 0)
            y[row] = partial;
        __syncthreads();
    }
}

__global__ void gemv_f4_row(const float* __restrict__ A, const float* __restrict__ x,
                            float* __restrict__ y, int N) {
    const float4* a4 = reinterpret_cast<const float4*>(A + (size_t)blockIdx.x * N);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const int n4 = N >> 2;
    const int tid = threadIdx.x;
    float partial = 0.f;
    for (int j = tid; j < n4; j += BLOCK) {
        float4 a = __ldg(a4 + j);
        float4 b = __ldg(x4 + j);
        partial += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    partial = blockReduceSum<BLOCK>(partial);
    if (tid == 0)
        y[blockIdx.x] = partial;
}

__global__ void gemv_scalar_row(const float* __restrict__ A, const float* __restrict__ x,
                                float* __restrict__ y, int N) {
    const float* row = A + (size_t)blockIdx.x * N;
    const int tid = threadIdx.x;
    float partial = 0.f;
    for (int j = tid; j < N; j += BLOCK)
        partial += __ldg(row + j) * __ldg(x + j);
    partial = blockReduceSum<BLOCK>(partial);
    if (tid == 0)
        y[blockIdx.x] = partial;
}

void solve(const float* A, const float* x, float* y, int M, int N, int /*nnz*/) {
    if (M <= 0)
        return;

    int sm = 0, maxShmem = 0;
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
    cudaDeviceGetAttribute(&maxShmem, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    if (sm <= 0)
        sm = 64;

    const size_t shmem = (size_t)N * sizeof(float);
    const bool fit = (int)shmem <= maxShmem && N > 0;

    // smem 放得下 x：让每个 block 扫多行，摊销 x 的加载成本。
    if (fit) {
        int perSm = maxShmem / (int)shmem;
        if (perSm < 1)
            perSm = 1;
        if (perSm > 4)
            perSm = 4;
        int blocks = sm * perSm;
        if (blocks > M)
            blocks = M;
        if ((N & 3) == 0)
            gemv_f4_smem<<<blocks, BLOCK, shmem>>>(A, x, y, M, N);
        else
            gemv_scalar_smem<<<blocks, BLOCK, shmem>>>(A, x, y, M, N);
    } else if ((N & 3) == 0) {
        gemv_f4_row<<<M, BLOCK>>>(A, x, y, N);
    } else {
        gemv_scalar_row<<<M, BLOCK>>>(A, x, y, N);
    }
    cudaDeviceSynchronize();
}
