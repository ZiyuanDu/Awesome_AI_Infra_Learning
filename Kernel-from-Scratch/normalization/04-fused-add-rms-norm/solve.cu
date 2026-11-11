#include <cuda_runtime.h>

#define CEIL(x, y) (((x) + (y) - 1) / (y))

void solve(
    const float* x,
    const float* residual,
    const float* weight,
    float* out,
    int N,
    int C,
    float eps
) {
    // TODO
}
