#include <cuda_runtime.h>
#include <stdint.h>

#define CEIL(x, y) (((x) + (y) - 1) / (y))

void solve(
    const int8_t* A,
    const int8_t* B,
    int8_t* C,
    int M,
    int N,
    int K,
    float scale_A,
    float scale_B,
    float scale_C,
    int zero_point_A,
    int zero_point_B,
    int zero_point_C
) {
    // TODO
}
