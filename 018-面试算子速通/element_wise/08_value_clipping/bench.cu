#include "bench.cuh"

void solve(const float* input, float* output, int N, float lo, float hi);

#include <algorithm>
#include <vector>

static void clipCpu(const float* in, float* out, int N, float lo, float hi) {
    for (int i = 0; i < N; ++i)
        out[i] = std::min(std::max(in[i], lo), hi);
}

static void test(int N, float lo, float hi, const char* name, double bytes = 0, bool check = true) {
    std::vector<float> in(N);
    for (int i = 0; i < N; ++i)
        in[i] = (float)((i % 17) - 8);
    bench(name, {&in}, N,
          [=](const float* const* a, float* out) { clipCpu(a[0], out, N, lo, hi); },
          [=](const float* const* a, float* out) { solve(a[0], out, N, lo, hi); }, bytes, 0,
          check);
}

int main() {
    std::vector<float> ex1 = {1.5f, -2.f, 3.f, 4.5f};
    bench("ex1", {&ex1}, ex1.size(),
          [&](const float* const* a, float* out) { clipCpu(a[0], out, (int)ex1.size(), 0.f, 3.5f); },
          [&](const float* const* a, float* out) { solve(a[0], out, (int)ex1.size(), 0.f, 3.5f); });

    std::vector<float> ex2 = {-1.f, 2.f, 5.f};
    bench("ex2", {&ex2}, ex2.size(),
          [&](const float* const* a, float* out) {
              clipCpu(a[0], out, (int)ex2.size(), -0.5f, 2.5f);
          },
          [&](const float* const* a, float* out) {
              solve(a[0], out, (int)ex2.size(), -0.5f, 2.5f);
          });

    test(1, 0.f, 1.f, "n1");
    test(3, -0.5f, 2.5f, "tail");
    test(10007, -2.f, 3.f, "odd");

    // LeetGPU timing size — too small for meaningful BW
    test(100000, -1.f, 1.f, "leetgpu");

    // Local BW: saturate HBM; median via bench.cuh
    const int Nbw = 50000000;
    test(Nbw, -1.f, 1.f, "bw", 2.0 * Nbw * sizeof(float), /*check=*/false);
}
