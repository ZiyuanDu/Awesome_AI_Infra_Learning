#include "bench.cuh"

void solve(const float* input, float* output, int N);

#include <vector>

static void leakyCpu(const float* in, float* out, int N) {
    for (int i = 0; i < N; ++i)
        out[i] = in[i] > 0.f ? in[i] : 0.01f * in[i];
}

static void test(int N, const char* name, double bytes = 0) {
    std::vector<float> in(N);
    for (int i = 0; i < N; ++i)
        in[i] = (float)((i % 13) - 6);
    bench(name, {&in}, N,
          [=](const float* const* a, float* out) { leakyCpu(a[0], out, N); },
          [=](const float* const* a, float* out) { solve(a[0], out, N); }, bytes);
}

int main() {
    std::vector<float> ex = {-2.f, -1.f, 0.f, 1.f, 2.f};
    bench("ex", {&ex}, ex.size(),
          [&](const float* const* a, float* out) { leakyCpu(a[0], out, (int)ex.size()); },
          [&](const float* const* a, float* out) { solve(a[0], out, (int)ex.size()); });

    test(1, "n1");
    test(3, "tail");
    test(10007, "odd");

    const int N = 50000000;
    test(N, "leetgpu", 2.0 * N * sizeof(float));
}
