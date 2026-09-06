#include "common.cuh"


constexpr int BM = 128, BN = 128, BK = 16;          // 块级处理128x128的矩阵，每个块处理16x16的矩阵
constexpr int TM = 8, TN = 4;                       // 每个线程负责计算8x4的矩阵
constexpr int BLD = BN + 4;                         // 为了解决bank conflict，需要多加4列
constexpr int T = (BM / TM) * (BN / TN);            // 每个 Block 包含的总线程数
constexpr int nA = BM * BK / 4 / T;                 // 每个线程需要加载的 float4 数量 总元素 / 4 / 每个 Block 包含的总线程数
constexpr int nB = BK * BN / 4 / T;                 // 每个线程需要加载的 float4 数量 总元素 / 4 / 每个 Block 包含的总线程数



__device__ __forceinline__ void load_ab(const float* A, const float* B, int M, int N, int K, int n0,
                                       int tid, int by, int bx, float4 ra[nA], float4 rb[nB]) {
    // 加载 A 矩阵 
#pragma unroll
    for (int t = 0; t < nA; ++t) {
        int idx = t * T + tid;
        int m = idx / (BK / 4);
        int n = (idx % (BK / 4)) * 4;
        // 加载 A 矩阵的 float4 元素
        ra[t] = load4(A, N, by * BM + m, n0 + n, M, N);
    }
    // 加载 B 矩阵 
#pragma unroll
    for (int t = 0; t < nB; ++t) {
        int idx = t * T + tid;
        int n = idx / (BN / 4);
        int k = (idx % (BN / 4)) * 4;
        // 加载 B 矩阵的 float4 元素
        rb[t] = load4(B, K, n0 + n, bx * BN + k, N, K);
    }
}

__device__ __forceinline__ void store_ab(float As[][BM], float Bs[][BLD], int tid, const float4 ra[nA],
                                         const float4 rb[nB]) {
    // 面试亮点：A 矩阵转置
#pragma unroll
    for (int t = 0; t < nA; ++t) {
        int idx = t * T + tid;
        int m = idx / (BK / 4);
        int n = (idx % (BK / 4)) * 4;
        As[n + 0][m] = ra[t].x;
        As[n + 1][m] = ra[t].y;
        As[n + 2][m] = ra[t].z;
        As[n + 3][m] = ra[t].w;
    }
#pragma unroll
    for (int t = 0; t < nB; ++t) {
        int idx = t * T + tid;
        int n = idx / (BN / 4);
        int k = (idx % (BN / 4)) * 4;
        *reinterpret_cast<float4*>(&Bs[n][k]) = rb[t];
    }
}

/* 寄存器外积 */
__device__ __forceinline__ void mma(const float As[][BM], const float Bs[][BLD], int ty, int tx,
                                   float acc[TM][TN]) {
#pragma unroll
    // 遍历内层维度 BK
    for (int n = 0; n < BK; ++n) {
        float av[TM], bv[TN];
        // 读取 A 的 8个元素
        *reinterpret_cast<float4*>(av) = *reinterpret_cast<const float4*>(&As[n][ty * TM]);
        *reinterpret_cast<float4*>(av + 4) = *reinterpret_cast<const float4*>(&As[n][ty * TM + 4]);
        // 读取 B 的 4个元素
        *reinterpret_cast<float4*>(bv) = *reinterpret_cast<const float4*>(&Bs[n][tx * TN]);

        // 寄存器外积更新，一次访存支持 32 次运算
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int j = 0; j < TN; ++j)
                acc[i][j] = __fmaf_rn(av[i], bv[j], acc[i][j]);
    }
}

__global__ void matrix_multiplication_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                             float* __restrict__ C, int M, int N, int K) {
    // 双缓冲，算写分离，避免访存冲突
    __shared__ float As[2][BK][BM];
    __shared__ float Bs[2][BK][BLD];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int by = blockIdx.y, bx = blockIdx.x;
    float acc[TM][TN] = {};
    float4 ra[nA], rb[nB];

    // 加载 A 矩阵和 B 矩阵
    load_ab(A, B, M, N, K, 0, tid, by, bx, ra, rb);
    // 存储 A 矩阵和 B 矩阵
    store_ab(As[0], Bs[0], tid, ra, rb);
    __syncthreads();

    int rb_id = 0;
    const int tiles = CEIL(N, BK);
    for (int t = 1; t < tiles; ++t) {
        int wb_id = rb_id ^ 1;
        load_ab(A, B, M, N, K, t * BK, tid, by, bx, ra, rb);
        mma(As[rb_id], Bs[rb_id], ty, tx, acc);
        store_ab(As[wb_id], Bs[wb_id], tid, ra, rb);
        __syncthreads();
        rb_id = wb_id;
    }
    mma(As[rb_id], Bs[rb_id], ty, tx, acc);


    // 存储结果
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        int row = by * BM + ty * TM + i;
        if (row >= M)
            continue;
#pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int col = bx * BN + tx * TN + j;
            int idx = row * K + col;
            if (col + 3 < K && (idx & 3) == 0)
                *reinterpret_cast<float4*>(&C[idx]) =
                    make_float4(acc[i][j], acc[i][j + 1], acc[i][j + 2], acc[i][j + 3]);
            else {
#pragma unroll
                for (int u = 0; u < 4; ++u)
                    if (col + u < K)
                        C[idx + u] = acc[i][j + u];
            }
        }
    }
}

void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(BN / TN, BM / TM);
    dim3 blocks(CEIL(K, BN), CEIL(M, BM));
    matrix_multiplication_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
