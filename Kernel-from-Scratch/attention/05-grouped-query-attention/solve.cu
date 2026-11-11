#include <cuda_runtime.h>

#define CEIL(x, y) (((x) + (y) - 1) / (y))

void solve(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim
) {
    // TODO
}
