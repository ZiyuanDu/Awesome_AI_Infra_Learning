#include <cuda_runtime.h>

#define CEIL(x, y) (((x) + (y) - 1) / (y))

void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {
    // TODO
}
