#include <cuda_runtime.h>

#define CEIL(x, y) (((x) + (y) - 1) / (y))

void solve(
    const float* partial_out,
    const float* partial_lse,
    float* output,
    int num_splits,
    int num_heads,
    int head_dim
) {
    // TODO
}
