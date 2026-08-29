#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <vector>

#include <cuda_runtime.h>


template <class F>
float timeMs(F f, int warmup = 3, int runs = 10) {
    for (int i = 0; i < warmup; ++i)
        f();
    cudaDeviceSynchronize();
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < runs; ++i)
        f();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms / runs;
}

template <class Cpu, class Gpu>
void bench(const char* name, std::initializer_list<const std::vector<float>*> ins, size_t nOut,
           Cpu cpu, Gpu gpu, double bytes = 0, double flops = 0, bool check = true) {
    std::vector<const float*> hIn, dIn;
    std::vector<float*> dOwned;
    for (auto* v : ins) {
        hIn.push_back(v->data());
        float* p = nullptr;
        cudaMalloc(&p, v->size() * sizeof(float));
        cudaMemcpy(p, v->data(), v->size() * sizeof(float), cudaMemcpyHostToDevice);
        dOwned.push_back(p);
        dIn.push_back(p);
    }

    std::vector<float> ref(nOut), got(nOut);
    float* dOut = nullptr;
    cudaMalloc(&dOut, nOut * sizeof(float));

    if (check) {
        cpu(hIn.data(), ref.data());
        gpu(dIn.data(), dOut);
        cudaMemcpy(got.data(), dOut, nOut * sizeof(float), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < nOut; ++i) {
            if (std::fabs(got[i] - ref[i]) > 1e-4f) {
                printf("FAIL %s  [%zu]  %g vs %g\n", name, i, got[i], ref[i]);
                std::exit(1);
            }
        }
        printf("OK  %s\n", name);
    } else {
        gpu(dIn.data(), dOut);
        printf("RUN %s\n", name);
    }

    if (bytes > 0 || flops > 0) {
        float ms = timeMs([&] { gpu(dIn.data(), dOut); });
        printf("    %.3f ms", ms);
        if (flops > 0)
            printf("  %.2f TFLOPS", flops / (ms * 1e9));
        if (bytes > 0)
            printf("  %.0f GB/s", bytes / (ms * 1e6));
        printf("\n");
    }

    for (float* p : dOwned)
        cudaFree(p);
    cudaFree(dOut);
}
