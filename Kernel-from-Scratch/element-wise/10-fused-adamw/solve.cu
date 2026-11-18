#include <cuda_runtime.h>

#define CEIL(x, y) (((x) + (y) - 1) / (y))

constexpr int THREADS = 256;

void solve(
    float* params,
    const float* grad,
    float* m,
    float* v,
    int N,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int t
) {
    // TODO
}
