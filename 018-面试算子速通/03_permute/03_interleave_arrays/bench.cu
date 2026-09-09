#include "bench.cuh"

void solve(const float* A, const float* B, float* output, int N);

#include <vector>

static void interleaveCpu(const float* A, const float* B, float* out, int N) {
    for (int i = 0; i < N; ++i) {
        out[2 * i] = A[i];
        out[2 * i + 1] = B[i];
    }
}

static void test(int N, const char* name, double bytes = 0) {
    std::vector<float> A(N), B(N);
    for (int i = 0; i < N; ++i) {
        A[i] = (float)i;
        B[i] = (float)(1000 + i);
    }
    bench(name, {&A, &B}, (size_t)N * 2,
          [=](const float* const* in, float* out) { interleaveCpu(in[0], in[1], out, N); },
          [=](const float* const* in, float* out) { solve(in[0], in[1], out, N); }, bytes);
}

int main() {
    std::vector<float> A1 = {1.f, 2.f, 3.f}, B1 = {4.f, 5.f, 6.f};
    bench("ex1", {&A1, &B1}, 6,
          [&](const float* const* in, float* out) { interleaveCpu(in[0], in[1], out, 3); },
          [&](const float* const* in, float* out) { solve(in[0], in[1], out, 3); });

    std::vector<float> A2 = {10.f, 20.f}, B2 = {30.f, 40.f};
    bench("ex2", {&A2, &B2}, 4,
          [&](const float* const* in, float* out) { interleaveCpu(in[0], in[1], out, 2); },
          [&](const float* const* in, float* out) { solve(in[0], in[1], out, 2); });

    test(1, "n1");
    test(3, "odd");
    test(10007, "odd-big");

    const int N = 25000000;
    test(N, "leetgpu", 4.0 * N * sizeof(float));
}
