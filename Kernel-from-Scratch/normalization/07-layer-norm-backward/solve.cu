#include <cuda_runtime.h>

#define CEIL(x, y) (((x) + (y) - 1) / (y))

void solve(
    const float* x,
    const float* gamma,
    const float* mean,
    const float* rstd,
    const float* dy,
    float* dx,
    float* dgamma,
    float* dbeta,
    int N,
    int C
) {
    // TODO
}
