#include "common.cuh"
#include <cstdint>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

/*
 * C = alpha*(A@B)+beta*C，half I/O / FP32 累加。
 *
 * 面试可写版：half smem + WMMA 16×16×16 + cp.async 双缓冲 + 高占用小 CTA。
 *
 * 前沿口播（2025–26）：
 *   sm100（数据中心 Blackwell）= tcgen05 + TMEM（Colfax / CUTLASS / gau-nernst）
 *   sm120（消费级 50 系）无 TMEM，仍走 mma.sync / WMMA；峰值看 cuBLAS≈95 TFLOPS@1024³(TC32F)
 */

constexpr int WM = 16, WN = 16, WK = 16;
// 4 warps × 32×32 寄存器 tile → BM=BN=64；128 thr + 小 smem → 多 CTA/SM
constexpr int WARPS_M = 2, WARPS_N = 2;
constexpr int TILES_M = 2, TILES_N = 2;
constexpr int BM = WARPS_M * TILES_M * WM;  // 64
constexpr int BN = WARPS_N * TILES_N * WN;  // 64
constexpr int BK = 32;
constexpr int BLD = BN + 8;  // pad 减 bank conflict
constexpr int THREADS = 128;

__device__ __forceinline__ void cp_async16(void* smem_ptr, const void* gmem_ptr) {
    unsigned smem = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}

__device__ __forceinline__ void g2s_async(const half* __restrict__ A, const half* __restrict__ B,
                                         half* __restrict__ As, half* __restrict__ Bs, int M, int N,
                                         int K, int k0, int by, int bx, int tid) {
    // A: BM×BK，每线程一次 8×half；越界用标量填 0
#pragma unroll
    for (int i = tid * 8; i < BM * BK; i += THREADS * 8) {
        int r = i / BK, c = i % BK;
        int gr = by * BM + r, gc = k0 + c;
        const half* srca = A + gr * K + gc;
        if (gr < M && gc + 7 < K && (gc & 7) == 0 && ((uintptr_t)srca & 15) == 0) {
            cp_async16(As + i, srca);
        } else {
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                int cc = c + u;
                As[i + u] =
                    (gr < M && cc < BK && k0 + cc < K) ? __ldg(A + gr * K + k0 + cc) : __float2half(0.f);
            }
        }
    }
#pragma unroll
    for (int i = tid * 8; i < BK * BN; i += THREADS * 8) {
        int r = i / BN, c = i % BN;
        int gr = k0 + r, gc = bx * BN + c;
        half* dst = &Bs[r * BLD + c];
        const half* srcb = B + gr * N + gc;
        if (r < BK && c + 7 < BN && gr < K && gc + 7 < N && (gc & 7) == 0 &&
            ((uintptr_t)srcb & 15) == 0 && ((uintptr_t)dst & 15) == 0) {
            cp_async16(dst, srcb);
        } else {
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                int cc = c + u;
                Bs[r * BLD + cc] =
                    (r < BK && cc < BN && gr < K && bx * BN + cc < N)
                        ? __ldg(B + gr * N + bx * BN + cc)
                        : __float2half(0.f);
            }
        }
    }
    cp_commit();
}

__global__ void gemm_wmma(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C,
                          int M, int N, int K, float alpha, float beta) {
    __shared__ alignas(16) half As[2][BM * BK];
    __shared__ alignas(16) half Bs[2][BK * BLD];
    __shared__ float Ctmp[4][WM * WN];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = warp / WARPS_N;
    const int warp_n = warp % WARPS_N;
    const int by = blockIdx.y, bx = blockIdx.x;

    wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag[TILES_M];
    wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> b_frag[TILES_N];
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag[TILES_M][TILES_N];
#pragma unroll
    for (int im = 0; im < TILES_M; ++im)
#pragma unroll
        for (int in = 0; in < TILES_N; ++in)
            wmma::fill_fragment(c_frag[im][in], 0.f);

    int stage = 0;
    g2s_async(A, B, As[0], Bs[0], M, N, K, 0, by, bx, tid);
    cp_wait();
    __syncthreads();

    const int ntiles = CEIL(K, BK);
    for (int t = 0; t < ntiles; ++t) {
        const int nxt = stage ^ 1;
        if (t + 1 < ntiles)
            g2s_async(A, B, As[nxt], Bs[nxt], M, N, K, (t + 1) * BK, by, bx, tid);

        half* As_ = As[stage];
        half* Bs_ = Bs[stage];
#pragma unroll
        for (int kk = 0; kk < BK; kk += WK) {
#pragma unroll
            for (int im = 0; im < TILES_M; ++im) {
                const int row = (warp_m * TILES_M + im) * WM;
                wmma::load_matrix_sync(a_frag[im], As_ + row * BK + kk, BK);
            }
#pragma unroll
            for (int in = 0; in < TILES_N; ++in) {
                const int col = (warp_n * TILES_N + in) * WN;
                wmma::load_matrix_sync(b_frag[in], Bs_ + kk * BLD + col, BLD);
            }
#pragma unroll
            for (int im = 0; im < TILES_M; ++im)
#pragma unroll
                for (int in = 0; in < TILES_N; ++in)
                    wmma::mma_sync(c_frag[im][in], a_frag[im], b_frag[in], c_frag[im][in]);
        }

        if (t + 1 < ntiles) {
            cp_wait();
            __syncthreads();
            stage = nxt;
        }
    }

#pragma unroll
    for (int im = 0; im < TILES_M; ++im) {
#pragma unroll
        for (int in = 0; in < TILES_N; ++in) {
            wmma::store_matrix_sync(Ctmp[warp], c_frag[im][in], WN, wmma::mem_row_major);
            __syncwarp();
            const int gr0 = by * BM + (warp_m * TILES_M + im) * WM;
            const int gc0 = bx * BN + (warp_n * TILES_N + in) * WN;
#pragma unroll
            for (int i = lane; i < WM * WN; i += 32) {
                int r = i / WN, c = i % WN;
                int gr = gr0 + r, gc = gc0 + c;
                if (gr < M && gc < N) {
                    float v = alpha * Ctmp[warp][i];
                    if (beta != 0.f)
                        v = __fmaf_rn(beta, __half2float(C[gr * N + gc]), v);
                    C[gr * N + gc] = __float2half(v);
                }
            }
            __syncwarp();
        }
    }
}

constexpr int CBM = 64, CBN = 64, CBK = 16, CTM = 4, CTN = 4;
constexpr int CBLD = CBN + 4;
constexpr int CT = (CBM / CTM) * (CBN / CTN);

__global__ void gemm_cuda(const half* __restrict__ A, const half* __restrict__ B, half* __restrict__ C,
                          int M, int N, int K, float alpha, float beta) {
    __shared__ half As[CBK][CBM];
    __shared__ half Bs[CBK][CBLD];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int by = blockIdx.y, bx = blockIdx.x;
    float acc[CTM][CTN] = {};

    for (int k0 = 0; k0 < K; k0 += CBK) {
        for (int i = tid; i < CBM * CBK; i += CT) {
            int r = i / CBK, c = i % CBK;
            int gr = by * CBM + r, gc = k0 + c;
            As[c][r] = (gr < M && gc < K) ? A[gr * K + gc] : __float2half(0.f);
        }
        for (int i = tid; i < CBK * CBN; i += CT) {
            int r = i / CBN, c = i % CBN;
            int gr = k0 + r, gc = bx * CBN + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : __float2half(0.f);
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < CBK; ++kk) {
            float av[CTM], bv[CTN];
#pragma unroll
            for (int i = 0; i < CTM; ++i)
                av[i] = __half2float(As[kk][ty * CTM + i]);
#pragma unroll
            for (int j = 0; j < CTN; ++j)
                bv[j] = __half2float(Bs[kk][tx * CTN + j]);
#pragma unroll
            for (int i = 0; i < CTM; ++i)
#pragma unroll
                for (int j = 0; j < CTN; ++j)
                    acc[i][j] = __fmaf_rn(av[i], bv[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < CTM; ++i) {
        int row = by * CBM + ty * CTM + i;
        if (row >= M)
            continue;
#pragma unroll
        for (int j = 0; j < CTN; ++j) {
            int col = bx * CBN + tx * CTN + j;
            if (col >= N)
                continue;
            float v = alpha * acc[i][j];
            if (beta != 0.f)
                v = __fmaf_rn(beta, __half2float(C[row * N + col]), v);
            C[row * N + col] = __float2half(v);
        }
    }
}

void solve(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    if ((M % 16) == 0 && (N % 16) == 0 && (K % 16) == 0) {
        gemm_wmma<<<dim3(CEIL(N, BN), CEIL(M, BM)), THREADS>>>(A, B, C, M, N, K, alpha, beta);
    } else {
        dim3 block(CBN / CTN, CBM / CTM);
        gemm_cuda<<<dim3(CEIL(N, CBN), CEIL(M, CBM)), block>>>(A, B, C, M, N, K, alpha, beta);
    }
    cudaDeviceSynchronize();
}
