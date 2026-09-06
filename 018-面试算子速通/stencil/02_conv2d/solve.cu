#include "common.cuh"

/*
 * 2D valid：out[i,j] = Σ_{m,n} in[i+m,j+n] * ker[m,n]
 *
 * 同 1D：输出 tile + 输入 halo → smem；核 → constant；行距 +1 减 bank 冲突。
 */
constexpr int TY = 16;
constexpr int TX = 16;
constexpr int KMAX = 31 * 31;

__constant__ float c_ker[KMAX];

__global__ void conv2d_kernel(const float* __restrict__ in, float* __restrict__ out, int inR,
                              int inC, int kR, int kC, int outR, int outC) {
    extern __shared__ float sIn[];
    const int tileR = TY + kR - 1;
    const int tileC = TX + kC - 1;
    const int ld = tileC + 1;

    const int r0 = blockIdx.y * TY;
    const int c0 = blockIdx.x * TX;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int tid = ty * TX + tx;

    for (int t = tid; t < tileR * tileC; t += TY * TX) {
        const int lr = t / tileC;
        const int lc = t - lr * tileC;
        const int gr = r0 + lr;
        const int gc = c0 + lc;
        sIn[lr * ld + lc] = (gr < inR && gc < inC) ? in[gr * inC + gc] : 0.f;
    }
    __syncthreads();

    const int orow = r0 + ty;
    const int ocol = c0 + tx;
    if (orow >= outR || ocol >= outC)
        return;

    float acc = 0.f;
    for (int m = 0; m < kR; ++m) {
        const float* row = sIn + (ty + m) * ld + tx;
        const float* krow = c_ker + m * kC;
        int n = 0;
        for (; n + 3 < kC; n += 4) {
            acc += row[n] * krow[n] + row[n + 1] * krow[n + 1] + row[n + 2] * krow[n + 2] +
                   row[n + 3] * krow[n + 3];
        }
        for (; n < kC; ++n)
            acc += row[n] * krow[n];
    }
    out[orow * outC + ocol] = acc;
}

void solve(const float* input, const float* kernel, float* output, int input_rows, int input_cols,
           int kernel_rows, int kernel_cols) {
    const int outR = input_rows - kernel_rows + 1;
    const int outC = input_cols - kernel_cols + 1;
    if (outR <= 0 || outC <= 0)
        return;

    cudaMemcpyToSymbol(c_ker, kernel, kernel_rows * kernel_cols * sizeof(float));
    const int ld = TX + kernel_cols - 1 + 1;
    const int smem = (TY + kernel_rows - 1) * ld * (int)sizeof(float);

    dim3 block(TX, TY);
    dim3 grid(CEIL(outC, TX), CEIL(outR, TY));
    conv2d_kernel<<<grid, block, smem>>>(input, output, input_rows, input_cols, kernel_rows,
                                         kernel_cols, outR, outC);
    cudaDeviceSynchronize();
}
