#include <cuda_runtime.h>

#define CEIL(x, y) (((x) + (y) - 1) / (y))

void solve(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N,
    int C,
    float eps
) {
    // TODO
}
