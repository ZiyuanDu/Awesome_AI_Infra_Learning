#include "common.cuh"

/*
 * Inclusive prefix sum（面试版）
 *
 * 复用 common：load4 / store4 / blockScanInclusive / blockReduceSum
 * 三步：tile 总和 → 扫 totals（inclusive）→ 带 seed 写一遍 output
 *
 * TILE=2048（512×float4）；N≤1e8 时 totals 两级够。
 */
constexpr int BLOCK = 512;
constexpr int TILE = BLOCK * 4;  // 2048

__global__ void reduce_tile(const float* __restrict__ in, float* __restrict__ tot, int N) {
    const int base = blockIdx.x * TILE + threadIdx.x * 4;
    float4 a = load4(in, base, N);
    float s = blockReduceSum<BLOCK>(a.x + a.y + a.z + a.w);
    if (threadIdx.x == 0)
        tot[blockIdx.x] = s;
}

/* seeds=nullptr 或 seeds 为 tile 总和的 inclusive scan：seed = totals[bid-1] */
__global__ void scan_tile(const float* __restrict__ in, float* __restrict__ out,
                          const float* __restrict__ seeds, int N) {
    const int base = blockIdx.x * TILE + threadIdx.x * 4;
    const float seed = (seeds && blockIdx.x) ? seeds[blockIdx.x - 1] : 0.f;

    float4 a = load4(in, base, N);
    float x = a.x, y = x + a.y, z = y + a.z, w = z + a.w;
    float pref = seed + blockScanExclusive<BLOCK>(w);
    store4(out, base, N, make_float4(pref + x, pref + y, pref + z, pref + w));
}

/* ---- totals 上的 1-item/thread scan（BLOCK 线程盖一层）---- */
__global__ void scan_tile1(const float* __restrict__ in, float* __restrict__ out,
                           float* __restrict__ tot, int n) {
    const int i = blockIdx.x * BLOCK + threadIdx.x;
    float v = (i < n) ? in[i] : 0.f;
    v = blockScanInclusive<BLOCK>(v);
    if (i < n)
        out[i] = v;
    const int nv = min(BLOCK, n - blockIdx.x * BLOCK);
    if (threadIdx.x == nv - 1)
        tot[blockIdx.x] = v;
}

__global__ void scan_inplace(float* __restrict__ data, int n) {
    float v = (threadIdx.x < n) ? data[threadIdx.x] : 0.f;
    v = blockScanInclusive<BLOCK>(v);
    if (threadIdx.x < n)
        data[threadIdx.x] = v;
}

__global__ void add_prev(float* __restrict__ data, const float* __restrict__ inc, int n) {
    const float add = blockIdx.x ? inc[blockIdx.x - 1] : 0.f;
    const int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n)
        data[i] += add;
}

static float *g_t1 = nullptr, *g_t1o = nullptr, *g_t2 = nullptr;
static int g_c1 = 0, g_c2 = 0;

static void ensure(int n1, int n2) {
    if (n1 > g_c1) {
        cudaFree(g_t1);
        cudaFree(g_t1o);
        cudaMalloc(&g_t1, n1 * sizeof(float));
        cudaMalloc(&g_t1o, n1 * sizeof(float));
        g_c1 = n1;
    }
    if (n2 > g_c2) {
        cudaFree(g_t2);
        cudaMalloc(&g_t2, n2 * sizeof(float));
        g_c2 = n2;
    }
}

/* 把 data[0..n) 变成 inclusive scan（就地）；调用方已 ensure(n1,n2) */
static void inclusive_scan_small(float* data, int n) {
    if (n <= BLOCK) {
        scan_inplace<<<1, BLOCK>>>(data, n);
        return;
    }
    const int n2 = CEIL(n, BLOCK);
    scan_tile1<<<n2, BLOCK>>>(data, g_t1o, g_t2, n);
    scan_inplace<<<1, BLOCK>>>(g_t2, n2);
    add_prev<<<n2, BLOCK>>>(g_t1o, g_t2, n);
    cudaMemcpy(data, g_t1o, n * sizeof(float), cudaMemcpyDeviceToDevice);
}

void solve(const float* input, float* output, int N) {
    if (N <= 0)
        return;
    if (N <= TILE) {
        scan_tile<<<1, BLOCK>>>(input, output, nullptr, N);
        cudaDeviceSynchronize();
        return;
    }

    const int n1 = CEIL(N, TILE);
    const int n2 = CEIL(n1, BLOCK);
    ensure(n1, n2);

    reduce_tile<<<n1, BLOCK>>>(input, g_t1, N);
    inclusive_scan_small(g_t1, n1);
    scan_tile<<<n1, BLOCK>>>(input, output, g_t1, N);
    cudaDeviceSynchronize();
}
