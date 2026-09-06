#include "bench.cuh"

void solve(const float* A, float* B, int N);

#include <vector>

static void copyCpu(const float* A, float* B, int N) {
    const int n = N * N;
    for (int i = 0; i < n; ++i)
        B[i] = A[i];
}

static void test(int N, const char* name, double bytes = 0) {
    std::vector<float> A((size_t)N * N);
    for (size_t i = 0; i < A.size(); ++i)
        A[i] = (float)(i % 17);
    bench(name, {&A}, (size_t)N * N,
          [=](const float* const* a, float* out) { copyCpu(a[0], out, N); },
          [=](const float* const* a, float* out) { solve(a[0], out, N); }, bytes);
}

int main() {
    test(1, "1x1");
    test(2, "2x2");
    test(3, "3x3");
    test(17, "tail");

    const int N = 4096;
    test(N, "leetgpu", 2.0 * N * N * sizeof(float));
}
