#include "bench.cuh"

void solve(const float* input, float* output, int width, int height);

#include <vector>

static void grayCpu(const float* in, float* out, int nPix) {
    for (int i = 0; i < nPix; ++i) {
        const float* p = in + i * 3;
        out[i] = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
    }
}

static void test(int w, int h, const char* name, double bytes = 0, bool check = true) {
    const int nPix = w * h;
    std::vector<float> in((size_t)nPix * 3);
    for (size_t i = 0; i < in.size(); ++i)
        in[i] = (float)(i % 256);
    bench(name, {&in}, (size_t)nPix,
          [=](const float* const* a, float* out) { grayCpu(a[0], out, nPix); },
          [=](const float* const* a, float* out) { solve(a[0], out, w, h); }, bytes, 0, check);
}

int main() {
    std::vector<float> ex1 = {255.f, 0.f, 0.f, 0.f, 255.f, 0.f, 0.f, 0.f, 255.f, 128.f, 128.f, 128.f};
    bench("ex1", {&ex1}, 4,
          [&](const float* const* a, float* out) { grayCpu(a[0], out, 4); },
          [&](const float* const* a, float* out) { solve(a[0], out, 2, 2); });

    std::vector<float> ex2 = {100.f, 150.f, 200.f};
    bench("ex2", {&ex2}, 1,
          [&](const float* const* a, float* out) { grayCpu(a[0], out, 1); },
          [&](const float* const* a, float* out) { solve(a[0], out, 1, 1); });

    test(1, 1, "1x1");
    test(3, 5, "odd");
    test(17, 19, "tail");

    // LeetGPU size (also problem max: w*h <= 2^22). Fits in L2 → don't report GB/s.
    test(2048, 2048, "leetgpu");

    // Stream many disjoint 2048^2 tiles so footprint >> L2 (~96MB on 5090)
    {
        const int w = 2048, h = 2048;
        const int nPix = w * h;
        const size_t inElems = (size_t)nPix * 3;
        const double bytes1 = 4.0 * nPix * sizeof(float);  // read RGB + write gray
        const int tiles = 16;  // ~768MB input → force HBM traffic
        float *dIn = nullptr, *dOut = nullptr;
        cudaMalloc(&dIn, tiles * inElems * sizeof(float));
        cudaMalloc(&dOut, tiles * (size_t)nPix * sizeof(float));
        cudaMemset(dIn, 1, tiles * inElems * sizeof(float));

        float ms = timeMsMedian([&] {
            for (int t = 0; t < tiles; ++t)
                solve(dIn + (size_t)t * inElems, dOut + (size_t)t * nPix, w, h);
        });
        printf("RUN bw (%dx tiles)\n", tiles);
        printf("    %.3f ms (median)  %.0f GB/s\n", ms, (tiles * bytes1) / (ms * 1e6));
        cudaFree(dIn);
        cudaFree(dOut);
    }
}
