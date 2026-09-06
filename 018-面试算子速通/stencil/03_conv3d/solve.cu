#include "common.cuh"

/*
 * 3D valid：out[d,r,c] = Σ in[d+kd,r+kr,c+kc] * ker[kd,kr,kc]
 * 布局：depth 最外，idx = ((d * rows + r) * cols + c)
 *
 * 同 2D，多一维：block=(8,8,8) 一次装输入立方 halo；核 → constant。
 */
constexpr int TZ = 8;
constexpr int TY = 8;
constexpr int TX = 8;
constexpr int KMAX = 5 * 5 * 5;

__constant__ float c_ker[KMAX];

__device__ __forceinline__ int idx3(int d, int r, int c, int rows, int cols) {
    return (d * rows + r) * cols + c;
}

__global__ void conv3d_kernel(const float* __restrict__ in, float* __restrict__ out, int inD,
                              int inR, int inC, int kD, int kR, int kC, int outD, int outR,
                              int outC) {
    extern __shared__ float sIn[];
    const int tileD = TZ + kD - 1;
    const int tileR = TY + kR - 1;
    const int tileC = TX + kC - 1;
    const int ld = tileC + 1;  // 最内维 pad，同 2D
    const int plane = tileR * ld;
    const int tileVol = tileD * tileR * tileC;

    const int d0 = blockIdx.z * TZ;
    const int r0 = blockIdx.y * TY;
    const int c0 = blockIdx.x * TX;
    const int tz = threadIdx.z;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int tid = (tz * TY + ty) * TX + tx;
    const int nThreads = TZ * TY * TX;

    for (int t = tid; t < tileVol; t += nThreads) {
        const int ldpth = t / (tileR * tileC);
        const int rem = t - ldpth * tileR * tileC;
        const int lr = rem / tileC;
        const int lc = rem - lr * tileC;
        const int gd = d0 + ldpth;
        const int gr = r0 + lr;
        const int gc = c0 + lc;
        sIn[ldpth * plane + lr * ld + lc] =
            (gd < inD && gr < inR && gc < inC) ? in[idx3(gd, gr, gc, inR, inC)] : 0.f;
    }
    __syncthreads();

    const int od = d0 + tz;
    const int orow = r0 + ty;
    const int ocol = c0 + tx;
    if (od >= outD || orow >= outR || ocol >= outC)
        return;

    float acc = 0.f;
    for (int kd = 0; kd < kD; ++kd) {
        for (int kr = 0; kr < kR; ++kr) {
            const float* inBase = sIn + (tz + kd) * plane + (ty + kr) * ld + tx;
            const float* kBase = c_ker + (kd * kR + kr) * kC;
            for (int kc = 0; kc < kC; ++kc)
                acc += inBase[kc] * kBase[kc];
        }
    }
    out[idx3(od, orow, ocol, outR, outC)] = acc;
}

void solve(const float* input, const float* kernel, float* output, int input_depth, int input_rows,
           int input_cols, int kernel_depth, int kernel_rows, int kernel_cols) {
    const int outD = input_depth - kernel_depth + 1;
    const int outR = input_rows - kernel_rows + 1;
    const int outC = input_cols - kernel_cols + 1;
    if (outD <= 0 || outR <= 0 || outC <= 0)
        return;

    cudaMemcpyToSymbol(c_ker, kernel, kernel_depth * kernel_rows * kernel_cols * sizeof(float));

    const int tileD = TZ + kernel_depth - 1;
    const int tileR = TY + kernel_rows - 1;
    const int ld = TX + kernel_cols - 1 + 1;
    const int smem = tileD * tileR * ld * (int)sizeof(float);

    dim3 block(TX, TY, TZ);
    dim3 grid(CEIL(outC, TX), CEIL(outR, TY), CEIL(outD, TZ));
    conv3d_kernel<<<grid, block, smem>>>(input, output, input_depth, input_rows, input_cols,
                                         kernel_depth, kernel_rows, kernel_cols, outD, outR, outC);
    cudaDeviceSynchronize();
}
