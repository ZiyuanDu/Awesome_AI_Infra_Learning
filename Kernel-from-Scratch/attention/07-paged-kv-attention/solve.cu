#include <cuda_runtime.h>

#define CEIL(x, y) (((x) + (y) - 1) / (y))

void solve(
    const float* Q,
    const float* K_cache,
    const float* V_cache,
    const int* block_table,
    const int* context_lens,
    float* output,
    int batch_size,
    int num_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq
) {
    // TODO
}
