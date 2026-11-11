#include <cuda_runtime.h>

#define CEIL(x, y) (((x) + (y) - 1) / (y))

void solve(
    const float* input,
    float* output,
    int N,
    int C,
    int H,
    int W,
    int kernel_size,
    int stride,
    int padding
) {
    // TODO
}
