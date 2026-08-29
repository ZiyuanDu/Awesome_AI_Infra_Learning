#include "solve.h"
#include "bench.cuh"

#include <vector>

static void transposeCpu(const float* in, float* out, int rows, int cols) {
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            out[c * rows + r] = in[r * cols + c];
}

static void test(int rows, int cols, const char* name, double bytes = 0) {
    std::vector<float> in((size_t)rows * cols);
    for (size_t i = 0; i < in.size(); ++i)
        in[i] = (float)(i % 17);

    bench(name, {&in}, (size_t)rows * cols,
          [=](const float* const* a, float* out) { transposeCpu(a[0], out, rows, cols); },
          [=](const float* const* a, float* out) { solve(a[0], out, rows, cols); }, bytes);
}

int main() {
    test(2, 3, "2x3");
    test(3, 1, "3x1");
    test(1, 1, "1x1");
    test(17, 19, "tail");

    const int rows = 7000, cols = 6000;
    test(rows, cols, "leetgpu", 2.0 * rows * cols * sizeof(float));
}
