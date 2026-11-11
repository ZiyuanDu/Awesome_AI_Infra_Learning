#include <cuda_runtime.h>

#define CEIL(x, y) (((x) + (y) - 1) / (y))

void solve(const float* logits, float* topk_weights, int* topk_indices, int M, int E, int k) {
    // TODO
}
