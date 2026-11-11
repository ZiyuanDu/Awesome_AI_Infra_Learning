#include <cuda_fp16.h>
#include <stdint.h>

#define CEIL(x, y) (((x) + (y) - 1) / (y))

void solve(
    const __half* x,
    const uint8_t* w_q,
    const __half* scales,
    __half* y,
    int M,
    int N,
    int K,
    int group_size
) {
    // TODO
}
