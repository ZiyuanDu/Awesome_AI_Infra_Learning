#include "common.cuh"

/*
 * Prefix Sum：inclusive scan，out[i] = in[0] + ... + in[i]。
 *
 * Inclusive prefix sum: out[i] = in[0] + ... + in[i]
 *
 * Three stages (same mental model as solve_naive.cu), vectorized + hierarchical:
 *   (1) sum_tiles   each block sums its TILE (512 thr x 4 = 2048 elems)
 *                   → chunk_sums[b] = block b's total
 *   (2) scan_sums   inclusive-scan chunk_sums[] in place (1 or 2 levels)
 *                   → chunk_sums[k] = total of blocks 0..k  (= the seed)
 *   (3) scan_tile   each block scans its TILE and adds its seed (chunk_sums[b-1])
 *
 * A block covers TILE = BLOCK*4 elements with one float4 per thread; the per-thread
 * 4-element sum feeds blockScanExclusive to get the cross-thread prefix. Scratch is
 * cached in function-local statics (no file-scope globals, no per-call malloc).
 */

constexpr int BLOCK = 512;
constexpr int TILE = BLOCK * 4;

__global__ void sum_tiles(const float* __restrict__ in, float* __restrict__ chunk_sums, int N);
__global__ void scan_tile(const float* __restrict__ in, float* __restrict__ out,
                          const float* __restrict__ seeds, int N);
__global__ void scan_single(float* __restrict__ data, int n);
__global__ void scan_chunks(const float* __restrict__ in, float* __restrict__ out,
                            float* __restrict__ block_sums, int n);
__global__ void add_offsets(float* __restrict__ data, const float* __restrict__ offsets, int n);
static void scan_sums(float* chunk_sums, float* scan_scratch, float* block_sums, int n_tiles);

void solve(const float* input, float* output, int N) {
    if (N <= 0)
        return;

    // One tile: no cross-tile carry needed
    if (N <= TILE) {
        scan_tile<<<1, BLOCK>>>(input, output, /*seeds=*/nullptr, N);
        cudaDeviceSynchronize();
        return;
    }

    const int n_tiles = CEIL(N, TILE);

    // Cached scratch (function-local statics, reused across calls).
    // chunk_sums holds per-tile totals, then becomes the seed array (seeds[b-1]).
    static float* chunk_sums = nullptr;
    static float* scan_scratch = nullptr;  // 2nd-level scan buffer (same size as chunk_sums)
    static float* block_sums = nullptr;    // 2nd-level per-block totals
    static size_t sum_cap = 0;             // covers chunk_sums & scan_scratch
    static size_t blk_cap = 0;             // covers block_sums

    if ((size_t)n_tiles > sum_cap) {
        if (chunk_sums) cudaFree(chunk_sums);
        if (scan_scratch) cudaFree(scan_scratch);
        cudaMalloc(&chunk_sums, (size_t)n_tiles * sizeof(float));
        sum_cap = (size_t)n_tiles;
        if (n_tiles > BLOCK)
            cudaMalloc(&scan_scratch, (size_t)n_tiles * sizeof(float));
    }
    if (n_tiles > BLOCK) {
        const size_t need_blk = (size_t)CEIL(n_tiles, BLOCK);
        if (need_blk > blk_cap) {
            if (block_sums) cudaFree(block_sums);
            cudaMalloc(&block_sums, need_blk * sizeof(float));
            blk_cap = need_blk;
        }
    }

    sum_tiles<<<n_tiles, BLOCK>>>(input, chunk_sums, N);     // (1) sum each tile
    scan_sums(chunk_sums, scan_scratch, block_sums, n_tiles); // (2) scan sums -> seeds
    scan_tile<<<n_tiles, BLOCK>>>(input, output, chunk_sums, N); // (3) local scan + seed

    cudaDeviceSynchronize();
}

// =============================================================================
// (1) Each block sums its TILE (one float4 per thread) into chunk_sums[bid]
// =============================================================================
__global__ void sum_tiles(const float* __restrict__ in, float* __restrict__ chunk_sums, int N) {
    const size_t base = blockIdx.x * TILE + threadIdx.x * 4;
    float4 a = load4(in, base, N);
    float s = blockReduceSum<BLOCK>(a.x + a.y + a.z + a.w);
    if (threadIdx.x == 0)
        chunk_sums[blockIdx.x] = s;
}

// =============================================================================
// (3) Local inclusive scan of 4 elems/thread, plus the seed from previous tiles.
//     seeds[b-1] == sum of all elements before this tile (from stage 2).
// =============================================================================
__global__ void scan_tile(const float* __restrict__ in, float* __restrict__ out,
                          const float* __restrict__ seeds, int N) {
    const size_t base = blockIdx.x * TILE + threadIdx.x * 4;
    const float seed = (seeds && blockIdx.x) ? seeds[blockIdx.x - 1] : 0.f;

    float4 a = load4(in, base, N);
    // 线程内 4 个元素先做串行 inclusive scan。
    float x = a.x;
    float y = x + a.y;
    float z = y + a.z;
    float w = z + a.w;  // 该线程 4 个元素的总和

    // 对每线程总和做 exclusive scan，得到本线程之前的跨线程前缀。
    float pref = seed + blockScanExclusive<BLOCK>(w);
    store4(out, base, N, make_float4(pref + x, pref + y, pref + z, pref + w));
}

// =============================================================================
// (2) Inclusive-scan the short chunk_sums[] in place -> it becomes the seeds.
//     One block if it fits; otherwise one extra level: scan each block, scan the
//     block totals, then add the offsets back.
// =============================================================================

// Single block: inclusive scan of n values in place.
__global__ void scan_single(float* __restrict__ data, int n) {
    float v = (threadIdx.x < n) ? data[threadIdx.x] : 0.f;
    v = blockScanInclusive<BLOCK>(v);
    if (threadIdx.x < n)
        data[threadIdx.x] = v;
}

// Multi-block level: each block scans BLOCK values and reports its own total.
__global__ void scan_chunks(const float* __restrict__ in, float* __restrict__ out,
                            float* __restrict__ block_sums, int n) {
    const size_t i = blockIdx.x * BLOCK + threadIdx.x;
    float v = (i < (size_t)n) ? in[i] : 0.f;
    v = blockScanInclusive<BLOCK>(v);
    if (i < (size_t)n)
        out[i] = v;

    // last live thread of this block writes the block's total
    const int n_live = min(BLOCK, n - (int)blockIdx.x * BLOCK);
    if (threadIdx.x == n_live - 1)
        block_sums[blockIdx.x] = v;
}

// Add each block's exclusive offset (= offsets[b-1]) to its values.
__global__ void add_offsets(float* __restrict__ data, const float* __restrict__ offsets, int n) {
    const float add = blockIdx.x ? offsets[blockIdx.x - 1] : 0.f;
    const size_t i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < (size_t)n)
        data[i] += add;
}

static void scan_sums(float* chunk_sums, float* scan_scratch, float* block_sums, int n_tiles) {
    if (n_tiles <= BLOCK) {
        scan_single<<<1, BLOCK>>>(chunk_sums, n_tiles);
        return;
    }
    // chunk_sums longer than one block
    const int n_blk = CEIL(n_tiles, BLOCK);
    scan_chunks<<<n_blk, BLOCK>>>(chunk_sums, scan_scratch, block_sums, n_tiles);
    scan_single<<<1, BLOCK>>>(block_sums, n_blk);
    add_offsets<<<n_blk, BLOCK>>>(scan_scratch, block_sums, n_tiles);
    cudaMemcpy(chunk_sums, scan_scratch, (size_t)n_tiles * sizeof(float), cudaMemcpyDeviceToDevice);
}
