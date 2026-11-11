#include "bench.cuh"

void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size);

#include <vector>

static void convCpu(const float* in, const float* ker, float* out, int N, int K) {
    const int nOut = N - K + 1;
    for (int i = 0; i < nOut; ++i) {
        float s = 0.f;
        for (int k = 0; k < K; ++k)
            s += in[i + k] * ker[k];
        out[i] = s;
    }
}

static void test(int N, int K, const char* name, double bytes = 0, double flops = 0,
                 bool check = true) {
    std::vector<float> in(N), ker(K);
    for (int i = 0; i < N; ++i)
        in[i] = (float)((i % 13) - 6);
    for (int i = 0; i < K; ++i)
        ker[i] = (float)((i % 5) - 2);

    const int nOut = N - K + 1;
    bench(name, {&in, &ker}, nOut,
          [=](const float* const* a, float* out) { convCpu(a[0], a[1], out, N, K); },
          [=](const float* const* a, float* out) { solve(a[0], a[1], out, N, K); }, bytes, flops,
          check);
}

int main() {
    test(5, 3, "tiny");
    test(8, 1, "k1");
    test(17, 5, "tail");
    test(10007, 63, "mid");

    const int N = 1 << 20, K = 31;
    const int nOut = N - K + 1;
    test(N, K, "k31", (double)(N + K + nOut) * sizeof(float), 2.0 * nOut * K);

    const int N2 = 1500000, K2 = 2047;
    const int nOut2 = N2 - K2 + 1;
    test(N2, K2, "leetgpu", (double)(N2 + K2 + nOut2) * sizeof(float), 2.0 * nOut2 * K2, false);
}
