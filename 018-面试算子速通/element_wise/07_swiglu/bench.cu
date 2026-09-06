#include "bench.cuh"

void solve(const float* input, float* output, int N);

#include <cmath>
#include <vector>

static void swigluCpu(const float* in, float* out, int N) {
    const int n = N / 2;
    for (int i = 0; i < n; ++i) {
        float x = in[i];
        out[i] = (x / (1.f + expf(-x))) * in[i + n];
    }
}

static void test(int N, const char* name, double bytes = 0) {
    std::vector<float> in(N);
    for (int i = 0; i < N; ++i)
        in[i] = (float)((i % 13) - 6);
    bench(name, {&in}, (size_t)N / 2,
          [=](const float* const* a, float* out) { swigluCpu(a[0], out, N); },
          [=](const float* const* a, float* out) { solve(a[0], out, N); }, bytes);
}

int main() {
    std::vector<float> ex = {-1.f, 2.f, 0.5f, -0.5f};
    bench("ex", {&ex}, 2,
          [&](const float* const* a, float* out) { swigluCpu(a[0], out, 4); },
          [&](const float* const* a, float* out) { solve(a[0], out, 4); });

    test(2, "n2");
    test(6, "tail");
    test(10006, "odd-half");

    const int N = 100000;
    test(N, "leetgpu", 1.5 * N * sizeof(float));
}
