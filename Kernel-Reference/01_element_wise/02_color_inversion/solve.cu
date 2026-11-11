#include "common.cuh"

// 颜色反转：RGB 取反，alpha 保持不变。
// image 是 uchar4 布局，按 4 字节一组向量化处理。

constexpr int THREADS = 256;
constexpr int MAX_BLOCKS = 1024;

__device__ __forceinline__ uchar4 invertColor(uchar4 v) {
    return make_uchar4(255 - v.x, 255 - v.y, 255 - v.z, v.w);  // keep A
}

// nPix = width*height pixels
__global__ void color_invert_kernel(unsigned char* __restrict__ image, int nPix) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    uchar4* p4 = reinterpret_cast<uchar4*>(image);

    for (size_t i = tid; i < (size_t)nPix; i += stride)
        p4[i] = invertColor(p4[i]);
}

void solve(unsigned char* image, int width, int height) {
    const int nPix = width * height;
    int blocks = CEIL(nPix, THREADS);
    if (blocks < 1)
        blocks = 1;
    if (blocks > MAX_BLOCKS)
        blocks = MAX_BLOCKS;
    color_invert_kernel<<<blocks, THREADS>>>(image, nPix);
    cudaDeviceSynchronize();
}
