#include <cuda_runtime.h>

#define CEIL(x, y) (((x) + (y) - 1) / (y))

void solve(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int N,
    int C,
    float eps
) {
    // TODO
}
