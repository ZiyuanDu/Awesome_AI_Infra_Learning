#include <cuda_runtime.h>

#define CEIL(x, y) (((x) + (y) - 1) / (y))

void solve(
    const float* X,
    const float* gamma,
    const float* beta,
    float* Y,
    int N,
    int C,
    int H,
    int W,
    int G,
    float eps
) {
    // TODO
}
