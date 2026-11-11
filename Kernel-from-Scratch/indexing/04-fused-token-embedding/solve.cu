#include <cuda_runtime.h>

#define CEIL(x, y) (((x) + (y) - 1) / (y))

void solve(
    const int* token_ids,
    const int* position_ids,
    const float* token_embeddings,
    const float* position_embeddings,
    const float* gamma,
    const float* beta,
    float* output,
    int B,
    int T,
    int V,
    int P,
    int D,
    float eps
) {
    // TODO
}
