#include <cuda_runtime.h>

#define CEIL(x, y) (((x) + (y) - 1) / (y))

void solve(
    const float* Q,
    const float* K_cache,
    const float* V_cache,
    const int* context_lens,
    float* output,
    int B,
    int H,
    int T,
    int D
) {
    // TODO
}
