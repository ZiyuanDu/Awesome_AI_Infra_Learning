#include "bench.cuh"

void solve(const float* input, float* output, int N);

#include <cmath>
#include <vector>

static void sigmoidCpu(const float* in, float* out, int N) {
    for (int i = 0; i < N; ++i)
        out[i] = 1.f / (1.f + expf(-in[i]));
}

static void test(int N, const char* name, double bytes = 0) {
    std::vector<float> in(N);
    for (int i = 0; i < N; ++i)
        in[i] = (float)((i % 13) - 6);
    bench(name, {&in}, N,
          [=](const float* const* a, float* out) { sigmoidCpu(a[0], out, N); },
          [=](const float* const* a, float* out) { solve(a[0], out, N); }, bytes);
}

int main() {
    std::vector<float> ex1 = {0.f, 1.f, -1.f, 2.f};
    bench("ex1", {&ex1}, ex1.size(),
          [&](const float* const* a, float* out) { sigmoidCpu(a[0], out, (int)ex1.size()); },
          [&](const float* const* a, float* out) { solve(a[0], out, (int)ex1.size()); });

    std::vector<float> ex2 = {0.5f, -0.5f, 3.f, -3.f};
    bench("ex2", {&ex2}, ex2.size(),
          [&](const float* const* a, float* out) { sigmoidCpu(a[0], out, (int)ex2.size()); },
          [&](const float* const* a, float* out) { solve(a[0], out, (int)ex2.size()); });

    test(1, "n1");
    test(3, "tail");
    test(10007, "odd");

    const int N = 50000000;
    test(N, "leetgpu", 2.0 * N * sizeof(float));
}
