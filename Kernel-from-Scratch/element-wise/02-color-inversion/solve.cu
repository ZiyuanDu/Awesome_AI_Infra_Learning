#include <cuda_runtime.h>

#define CEIL(x, y) (((x) + (y) - 1) / (y))
const int THREADS = 256;

__device__ __forceinline__ uchar4 invertColor(uchar4 v) {
    return make_uchar4(255-v.x, 255-v.y, 255-v.z, v.w);
}

__global__ void color_version_kernel(unsigned char* image, int nPix) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    uchar4* c4 = reinterpret_cast<uchar4*>(image);

    // 向量化处理 + Grid Loop
    for (size_t i = tid; i < (size_t)nPix; i += stride) {
        c4[i] = invertColor(c4[i]);
    }
    // 尾处理 None
}

void solve(unsigned char* image, int width, int height) {

    const int nPix = width * height;
    int blocks = CEIL(nPix, THREADS);

    color_version_kernel<<<blocks, THREADS>>>(image, nPix);
}
