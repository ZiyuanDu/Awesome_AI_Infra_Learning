#include "bench.cuh"

void solve(unsigned char* image, int width, int height);

#include <cstdio>
#include <vector>

static void invertCpu(unsigned char* img, int nPix) {
    for (int i = 0; i < nPix; ++i) {
        img[i * 4 + 0] = 255 - img[i * 4 + 0];
        img[i * 4 + 1] = 255 - img[i * 4 + 1];
        img[i * 4 + 2] = 255 - img[i * 4 + 2];
    }
}

static void test(int w, int h) {
    const int n = w * h * 4;
    std::vector<unsigned char> in(n);
    for (int i = 0; i < n; ++i)
        in[i] = (unsigned char)(i * 17);

    std::vector<unsigned char> ref = in;
    invertCpu(ref.data(), w * h);

    unsigned char* d = nullptr;
    cudaMalloc(&d, n);
    cudaMemcpy(d, in.data(), n, cudaMemcpyHostToDevice);
    solve(d, w, h);
    std::vector<unsigned char> got(n);
    cudaMemcpy(got.data(), d, n, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        if (got[i] != ref[i]) {
            printf("FAIL color %dx%d  [%d]  %d vs %d\n", w, h, i, got[i], ref[i]);
            std::exit(1);
        }
    }
    printf("OK  color %dx%d\n", w, h);
    cudaFree(d);
}

int main() {
    test(1, 1);
    test(2, 3);
    test(17, 19);

    const int w = 2048, h = 2048;
    const int n = w * h * 4;
    unsigned char* d = nullptr;
    cudaMalloc(&d, n);
    cudaMemset(d, 1, n);
    float ms = timeMs([&] { solve(d, w, h); }, 20, 40);
    printf("color %dx%d   %.3f ms  %.0f GB/s\n", w, h, ms, 2.0 * n / (ms * 1e6));
    cudaFree(d);
}
