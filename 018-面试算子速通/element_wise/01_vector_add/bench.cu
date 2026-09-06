#include "bench.cuh"

void solve(const float* A, const float* B, float* C, int N);

#include <vector>

int main() {
    const int N = 10007;
    std::vector<float> A(N), B(N);
    for (int i = 0; i < N; ++i) {
        A[i] = (float)i;
        B[i] = 1.f;
    }

    bench("add", {&A, &B}, N,
          [&](const float* const* in, float* out) {
              for (int i = 0; i < N; ++i)
                  out[i] = in[0][i] + in[1][i];
          },
          [&](const float* const* in, float* out) { solve(in[0], in[1], out, N); });
}
