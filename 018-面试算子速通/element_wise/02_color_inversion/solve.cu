#include "common.cuh"

/*
 * 颜色反转：RGBA 打包，原地。
 * R/G/B = 255 - x，A 不变。逐像素独立，带宽墙。
 * 一个线程一个像素，按 uchar4 读写（4 字节对齐，cudaMalloc 保证）。
 */
__global__ void color_invert_kernel(unsigned char* __restrict__ image, int nPix) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    uchar4* p = reinterpret_cast<uchar4*>(image);
    for (int i = tid; i < nPix; i += stride) {
        uchar4 v = p[i];
        v.x = 255 - v.x;
        v.y = 255 - v.y;
        v.z = 255 - v.z;
        p[i] = v;
    }
}

void solve(unsigned char* image, int width, int height) {
    const int nPix = width * height;
    const int threads = 256;
    int blocks = CEIL(nPix, threads);
    if (blocks < 1)
        blocks = 1;
    if (blocks > 1024)
        blocks = 1024;
    color_invert_kernel<<<blocks, threads>>>(image, nPix);
    cudaDeviceSynchronize();
}
