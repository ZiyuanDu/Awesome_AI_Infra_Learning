#include "bench.cuh"

void solve(const float* input, const float* kernel, float* output, int input_depth, int input_rows,
           int input_cols, int kernel_depth, int kernel_rows, int kernel_cols);

#include <vector>

static int idx3(int d, int r, int c, int rows, int cols) { return (d * rows + r) * cols + c; }

static void conv3dCpu(const float* in, const float* ker, float* out, int inD, int inR, int inC,
                      int kD, int kR, int kC) {
    const int outD = inD - kD + 1;
    const int outR = inR - kR + 1;
    const int outC = inC - kC + 1;
    for (int i = 0; i < outD; ++i)
        for (int j = 0; j < outR; ++j)
            for (int k = 0; k < outC; ++k) {
                float s = 0.f;
                for (int d = 0; d < kD; ++d)
                    for (int r = 0; r < kR; ++r)
                        for (int c = 0; c < kC; ++c)
                            s += in[idx3(i + d, j + r, k + c, inR, inC)] *
                                 ker[idx3(d, r, c, kR, kC)];
                out[idx3(i, j, k, outR, outC)] = s;
            }
}

static void check(const std::vector<float>& in, const std::vector<float>& ker, int inD, int inR,
                  int inC, int kD, int kR, int kC, const char* name) {
    const int outD = inD - kD + 1;
    const int outR = inR - kR + 1;
    const int outC = inC - kC + 1;
    const size_t nOut = (size_t)outD * outR * outC;
    std::vector<float> ref(nOut), got(nOut);
    conv3dCpu(in.data(), ker.data(), ref.data(), inD, inR, inC, kD, kR, kC);

    float *dIn = nullptr, *dKer = nullptr, *dOut = nullptr;
    cudaMalloc(&dIn, in.size() * sizeof(float));
    cudaMalloc(&dKer, ker.size() * sizeof(float));
    cudaMalloc(&dOut, nOut * sizeof(float));
    cudaMemcpy(dIn, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dKer, ker.data(), ker.size() * sizeof(float), cudaMemcpyHostToDevice);
    solve(dIn, dKer, dOut, inD, inR, inC, kD, kR, kC);
    cudaMemcpy(got.data(), dOut, nOut * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < nOut; ++i) {
        if (std::fabs(got[i] - ref[i]) > 1e-3f) {
            printf("FAIL %s  [%zu]  %g vs %g\n", name, i, got[i], ref[i]);
            std::exit(1);
        }
    }
    printf("OK  %s\n", name);
    cudaFree(dIn);
    cudaFree(dKer);
    cudaFree(dOut);
}

int main() {
    // ex1: 3x3x3 vol, 2x3x3 ker → out 2x1x1 = [82, 163]
    {
        std::vector<float> in(27);
        for (int i = 0; i < 27; ++i)
            in[i] = (float)(i + 1);
        std::vector<float> ker = {
            1, 0, 0, 1, 1, 1, 0, 0, 0,  // d=0
            1, 1, 0, 1, 1, 0, 0, 0, 1   // d=1
        };
        check(in, ker, 3, 3, 3, 2, 3, 3, "ex1");
    }
    // ex2: all-ones 2^3 → 36
    {
        std::vector<float> in = {1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<float> ker(8, 1.f);
        check(in, ker, 2, 2, 2, 2, 2, 2, "ex2");
    }

    {
        std::vector<float> in(5 * 7 * 9), ker(2 * 3 * 2);
        for (size_t i = 0; i < in.size(); ++i)
            in[i] = (float)((i % 11) - 5);
        for (size_t i = 0; i < ker.size(); ++i)
            ker[i] = (float)((i % 3) - 1);
        check(in, ker, 5, 7, 9, 2, 3, 2, "odd");
    }

    // 站点计时给了 rows/cols=128、ker 5×5；depth 取同量级 128、kd=5
    const int D = 128, R = 128, C = 128, kD = 5, kR = 5, kC = 5;
    std::vector<float> in((size_t)D * R * C), ker((size_t)kD * kR * kC);
    for (size_t i = 0; i < in.size(); ++i)
        in[i] = (float)((i % 13) - 6);
    for (size_t i = 0; i < ker.size(); ++i)
        ker[i] = (float)((i % 5) - 2);
    check(in, ker, D, R, C, kD, kR, kC, "leetgpu");

    const int outD = D - kD + 1, outR = R - kR + 1, outC = C - kC + 1;
    float *dIn = nullptr, *dKer = nullptr, *dOut = nullptr;
    cudaMalloc(&dIn, in.size() * sizeof(float));
    cudaMalloc(&dKer, ker.size() * sizeof(float));
    cudaMalloc(&dOut, (size_t)outD * outR * outC * sizeof(float));
    cudaMemcpy(dIn, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dKer, ker.data(), ker.size() * sizeof(float), cudaMemcpyHostToDevice);
    double bytes = (double)(in.size() + ker.size() + (size_t)outD * outR * outC) * sizeof(float);
    double flops = 2.0 * outD * outR * outC * kD * kR * kC;
    float ms = timeMs([&] { solve(dIn, dKer, dOut, D, R, C, kD, kR, kC); }, 5, 10);
    printf("    %.3f ms  %.0f GB/s  %.2f GFLOPS\n", ms, bytes / (ms * 1e6), flops / (ms * 1e6));
    cudaFree(dIn);
    cudaFree(dKer);
    cudaFree(dOut);
}
