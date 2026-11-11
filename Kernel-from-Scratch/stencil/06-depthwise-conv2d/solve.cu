#include <cuda_runtime.h>

#define CEIL(x, y) (((x) + (y) - 1) / (y))

void solve(
    const float* input,
    const float* kernel,
    float* output,
    int N,
    int C,
    int H,
    int W,
    int KH,
    int KW
) {
    // TODO
}
