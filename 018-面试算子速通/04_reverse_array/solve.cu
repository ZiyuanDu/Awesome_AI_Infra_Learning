 #include "solve.h"


// 最朴素：一个线程反转一对元素。读、写仍是合并的，
// 但每线程只处理 4 字节，指令更多、延迟掩盖差，带宽低于 float4 版。
__global__ void reverse_kernel_naive(float* __restrict__ in, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N / 2) {
        float t = in[i];
        in[i] = in[N - 1 - i];
        in[N - 1 - i] = t;
    }
}

__global__ void reverse_kernel(float* __restrict__ in, int N) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    if ((N & 3) == 0) {
        const int n4 = N >> 2;    // float4 组数
        const int half = n4 >> 1; // 成对组数
        for (int g = tid; g < half; g += stride) {
            const int lo = g << 2;
            const int hi = (n4 - 1 - g) << 2;
            float4 a = *reinterpret_cast<const float4*>(in + lo);
            float4 b = *reinterpret_cast<const float4*>(in + hi);
            // 镜像组内元素也需反转：in[lo+k] = b[3-k]
            *reinterpret_cast<float4*>(in + lo) = make_float4(b.w, b.z, b.y, b.x);
            *reinterpret_cast<float4*>(in + hi) = make_float4(a.w, a.z, a.y, a.x);
        }
        if (n4 & 1) { // 中间剩一组，组内元素也需反转
            const int c = half << 2;
            if (tid == 0) {
                float t = in[c];
                in[c] = in[c + 3];
                in[c + 3] = t;
                t = in[c + 1];
                in[c + 1] = in[c + 2];
                in[c + 2] = t;
            }
        }
    } else {
        for (int i = tid; i < N / 2; i += stride) {
            float t = in[i];
            in[i] = in[N - 1 - i];
            in[N - 1 - i] = t;
        }
    }
}

void solve(float* input, int N) {
    const int threads = 256;
    int blocks = CEIL(N / 2, threads);
    if (blocks < 1)
        blocks = 1;
    if (blocks > 1024)
        blocks = 1024; // 大数组由 grid-stride 兜底
    reverse_kernel<<<blocks, threads>>>(input, N);
    cudaDeviceSynchronize();
}
