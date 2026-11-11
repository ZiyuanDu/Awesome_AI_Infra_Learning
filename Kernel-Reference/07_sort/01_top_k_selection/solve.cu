#include "common.cuh"
#include <algorithm>
#include <cstdint>
#include <vector>

/*
 * Top-K Selection: pick the k largest elements (descending) of input[0..N).
 *
 * Fast path (k small, e.g. the measured k=100): MSD radix-select.
 *   floats -> monotone uint32 keys, so "top k by value" == "top k by key".
 *   Loop over slices of the key bits: histogram the current candidate set,
 *   find the bucket the k-th largest falls into, keep only buckets >= it.
 *   Survivors always number >= k and shrink ~512x per level, so after 1-2
 *   full passes they fit in one block and are bitonic-sorted -> top k out.
 *   Cost for N=5e7,k=100: ~2 full reads of N + sorting a few thousand keys.
 *
 * Ties/duplicates: handled by keeping whole buckets + the final sort.
 * General fallback: any k (incl. k ~ N) -> host std::partial_sort. This keeps
 * the interface total even though the GPU path is tuned for k << N.
 *
 * Scratch is cached in function-local statics (no per-call malloc).
 */

constexpr int BLOCK = 512;
constexpr int LEV_BITS = 9;         // 512 buckets per level (2 KB shared)
constexpr int NB = 1 << LEV_BITS;
constexpr int SORT_CAP = 8192;      // final sort in one block (32 KB shared)

// ---- float <-> monotone uint32 key (ascending value == ascending key) ----
__device__ __forceinline__ uint32_t f2key(float f) {
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float key2f(uint32_t k) {
    uint32_t u = (k & 0x80000000u) ? (k & 0x7fffffffu) : ~k;
    return __uint_as_float(u);
}

// =============================================================================
// Histogram src[0..n) into hist[] on buckets (key >> shift) & mask
// =============================================================================
__global__ void hist_level(const float* __restrict__ src, int n, int shift, int mask,
                           int* __restrict__ hist) {
    __shared__ int sh[NB];
    for (int t = threadIdx.x; t < NB; t += BLOCK) sh[t] = 0;
    __syncthreads();

    const int n4 = n / 4;
    const float4* s4 = reinterpret_cast<const float4*>(src);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += blockDim.x * gridDim.x) {
        float4 a = s4[i];
        atomicAdd(&sh[(f2key(a.x) >> shift) & mask], 1);
        atomicAdd(&sh[(f2key(a.y) >> shift) & mask], 1);
        atomicAdd(&sh[(f2key(a.z) >> shift) & mask], 1);
        atomicAdd(&sh[(f2key(a.w) >> shift) & mask], 1);
    }
    for (int i = n4 * 4 + blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x)
        atomicAdd(&sh[(f2key(src[i]) >> shift) & mask], 1);
    __syncthreads();

    for (int t = threadIdx.x; t < NB; t += BLOCK)
        if (sh[t]) atomicAdd(&hist[t], sh[t]);
}

// =============================================================================
// Boundary bucket j: smallest bucket index with suffix[j] >= k. Also kept=survivors
// =============================================================================
__global__ void find_boundary(const int* __restrict__ hist, int k, int* __restrict__ j,
                              int* __restrict__ kept) {
    if (threadIdx.x || blockIdx.x) return;
    int acc = 0;
    for (int b = NB - 1; b >= 0; --b) {
        acc += hist[b];
        if (acc >= k) {
            *j = b;
            *kept = acc;
            return;
        }
    }
    *j = 0;      // unreachable: k <= N guarantees some bucket crosses k
    *kept = k;
}

// =============================================================================
// Filter: copy src elements with bucket >= j into dst (exactly `kept` of them)
// =============================================================================
__global__ void filter_level(const float* __restrict__ src, float* __restrict__ dst, int n,
                             int shift, int mask, int j, int* __restrict__ cnt) {
    const int n4 = n / 4;
    const float4* s4 = reinterpret_cast<const float4*>(src);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += blockDim.x * gridDim.x) {
        float4 a = s4[i];
        if ((int)((f2key(a.x) >> shift) & mask) >= j) dst[atomicAdd(cnt, 1)] = a.x;
        if ((int)((f2key(a.y) >> shift) & mask) >= j) dst[atomicAdd(cnt, 1)] = a.y;
        if ((int)((f2key(a.z) >> shift) & mask) >= j) dst[atomicAdd(cnt, 1)] = a.z;
        if ((int)((f2key(a.w) >> shift) & mask) >= j) dst[atomicAdd(cnt, 1)] = a.w;
    }
    for (int i = n4 * 4 + blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x)
        if ((int)((f2key(src[i]) >> shift) & mask) >= j)
            dst[atomicAdd(cnt, 1)] = src[i];
}

// =============================================================================
// Bitonic-sort up to SORT_CAP keys ascending; write the largest k (descending)
// =============================================================================
__global__ void final_sort(const float* __restrict__ cand, int count,
                           float* __restrict__ out, int k) {
    __shared__ uint32_t keys[SORT_CAP];
    for (int i = threadIdx.x; i < SORT_CAP; i += blockDim.x)
        keys[i] = (i < count) ? f2key(cand[i]) : 0u;  // 0 < every valid key
    __syncthreads();

    for (int len = 2; len <= SORT_CAP; len <<= 1) {
        for (int half = len >> 1; half > 0; half >>= 1) {
            for (int idx = threadIdx.x; idx < SORT_CAP; idx += blockDim.x) {
                const int ixj = idx ^ half;
                if (ixj > idx) {
                    const bool up = (idx & len) == 0;
                    uint32_t a = keys[idx], b = keys[ixj];
                    if ((up && a > b) || (!up && a < b)) {
                        keys[idx] = b;
                        keys[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (int i = threadIdx.x; i < k; i += blockDim.x)
        out[i] = key2f(keys[SORT_CAP - 1 - i]);
}

// =============================================================================
// General fallback (any k): host partial_sort on a copy of the input
// =============================================================================
static void hostTopK(const float* input, float* output, int N, int k) {
    std::vector<float> h((size_t)N);
    cudaMemcpy(h.data(), input, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost);
    std::partial_sort(h.begin(), h.begin() + k, h.end(), std::greater<float>());
    cudaMemcpy(output, h.data(), (size_t)k * sizeof(float), cudaMemcpyHostToDevice);
}

void solve(const float* input, float* output, int N, int k) {
    if (N <= 0 || k <= 0)
        return;
    if (k > N)
        k = N;

    // The single-block final sort caps the fast path.
    if (k > SORT_CAP) {
        hostTopK(input, output, N, k);
        return;
    }

    // SM 数只在第一次调用时查询，避免每个请求重复开销。
    static int sm = [] {
        int s = 0;
        cudaDeviceGetAttribute(&s, cudaDevAttrMultiProcessorCount, 0);
        return s > 0 ? s : 1;
    }();

    // Cached scratch (function-local statics, reused across calls).
    static float* bufA = nullptr;   // ping-pong candidate buffers
    static float* bufB = nullptr;
    static int* hist = nullptr;     // NB counters
    static int* jd = nullptr;
    static int* keptd = nullptr;
    static int* cntd = nullptr;
    static int cap = 0;

    if (N > cap) {
        if (bufA) cudaFree(bufA);
        if (bufB) cudaFree(bufB);
        cudaMalloc(&bufA, (size_t)N * sizeof(float));
        cudaMalloc(&bufB, (size_t)N * sizeof(float));
        cap = N;
        if (!hist) cudaMalloc(&hist, (size_t)NB * sizeof(int));
        if (!jd) cudaMalloc(&jd, sizeof(int));
        if (!keptd) cudaMalloc(&keptd, sizeof(int));
        if (!cntd) cudaMalloc(&cntd, sizeof(int));
    }

    const int grid = std::min(CEIL(N, BLOCK * 4), sm * 16);

    const float* src = input;   // current candidate set
    int n = N;
    float* tgt = bufA;          // where the next filter writes
    int pos = 0;                // key bits consumed so far

    for (;;) {
        const int bits = std::min(LEV_BITS, 32 - pos);
        const int shift = 32 - pos - bits;
        const int mask = (1 << bits) - 1;

        cudaMemset(hist, 0, (size_t)NB * sizeof(int));
        hist_level<<<grid, BLOCK>>>(src, n, shift, mask, hist);
        find_boundary<<<1, 1>>>(hist, k, jd, keptd);

        int j = 0, kept = 0;
        cudaMemcpy(&j, jd, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&kept, keptd, sizeof(int), cudaMemcpyDeviceToHost);

        if (kept <= SORT_CAP) {
            cudaMemset(cntd, 0, sizeof(int));
            filter_level<<<grid, BLOCK>>>(src, tgt, n, shift, mask, j, cntd);
            final_sort<<<1, BLOCK>>>(tgt, kept, output, k);
            cudaDeviceSynchronize();
            return;
        }

        pos += bits;
        if (pos >= 32) {
            // Survivors still exceed the sort cap after all 32 bits are pinned:
            // they only agree on slice buckets, not on full keys. Fall back.
            cudaMemset(cntd, 0, sizeof(int));
            filter_level<<<grid, BLOCK>>>(src, tgt, n, shift, mask, j, cntd);
            hostTopK(tgt, output, kept, k);
            return;
        }

        // Shrink to the survivors (count == kept) and continue on the next slice.
        cudaMemset(cntd, 0, sizeof(int));
        filter_level<<<grid, BLOCK>>>(src, tgt, n, shift, mask, j, cntd);
        n = kept;
        src = tgt;
        tgt = (tgt == bufA) ? bufB : bufA;
    }
}
