#include "bench.cuh"

void solve(const float* input, const float* kernel, float* output, int input_rows, int input_cols,
           int kernel_rows, int kernel_cols);

#include <vector>

static void conv2dCpu(const float* in, const float* ker, float* out, int inR, int inC, int kR,
                      int kC) {
    const int outR = inR - kR + 1;
    const int outC = inC - kC + 1;
    for (int i = 0; i < outR; ++i) {
        for (int j = 0; j < outC; ++j) {
            float s = 0.f;
            for (int m = 0; m < kR; ++m)
                for (int n = 0; n < kC; ++n)
                    s += in[(i + m) * inC + (j + n)] * ker[m * kC + n];
            out[i * outC + j] = s;
        }
    }
}

static void test(int inR, int inC, int kR, int kC, const char* name, double bytes = 0,
                 double flops = 0) {
    std::vector<float> in((size_t)inR * inC), ker((size_t)kR * kC);
    for (size_t i = 0; i < in.size(); ++i)
        in[i] = (float)((i % 13) - 6);
    for (size_t i = 0; i < ker.size(); ++i)
        ker[i] = (float)((i % 5) - 2);

    const int outR = inR - kR + 1;
    const int outC = inC - kC + 1;
    const size_t nOut = (size_t)outR * outC;
    bench(name, {&in, &ker}, nOut,
          [=](const float* const* a, float* out) { conv2dCpu(a[0], a[1], out, inR, inC, kR, kC); },
          [=](const float* const* a, float* out) {
              solve(a[0], a[1], out, inR, inC, kR, kC);
          },
          bytes, flops);
}

int main() {
    // ex1
    {
        std::vector<float> in = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        std::vector<float> ker = {0, 1, 1, 0};
        bench("ex1", {&in, &ker}, 4,
              [&](const float* const* a, float* out) { conv2dCpu(a[0], a[1], out, 3, 3, 2, 2); },
              [&](const float* const* a, float* out) { solve(a[0], a[1], out, 3, 3, 2, 2); });
    }
    // ex2
    {
        std::vector<float> in = {1, 1, 1, 1, 1, 2, 3, 1, 1, 4, 5, 1, 1, 1, 1, 1};
        std::vector<float> ker = {1, 0, 1};
        bench("ex2", {&in, &ker}, 8,
              [&](const float* const* a, float* out) { conv2dCpu(a[0], a[1], out, 4, 4, 1, 3); },
              [&](const float* const* a, float* out) { solve(a[0], a[1], out, 4, 4, 1, 3); });
    }

    test(17, 19, 5, 3, "tail");
    test(64, 64, 7, 7, "mid");

    const int R = 3072, C = 3072, kR = 15, kC = 15;
    const int outR = R - kR + 1, outC = C - kC + 1;
    const double nOut = (double)outR * outC;
    test(R, C, kR, kC, "leetgpu",
         (double)(R * C + kR * kC + outR * outC) * sizeof(float), 2.0 * nOut * kR * kC);
}
