# Reduction（Reduce）

**LeetGPU：** Medium · [Reduction](https://leetgpu.com/challenges/reduction)  
**目录：** `reduce/01_reduction`

$$
\mathrm{out}=\sum_{i=0}^{N-1}x[i]
$$

例：`[1..8] → 36`，`[-2.5,1.5,-1,2] → 0`。

## 接口

```cuda
void solve(const float* input, float* output, int N);
```

`output` 是 **长度为 1** 的 device 标量。`1 ≤ N ≤ 1e8`，计时 `N=2^{22}`。和必须能放进 float。

## 为什么归 reduce

多读一写，线程要通信。element-wise 不够；permute 不改数值聚合。骨架：

1. **grid-stride** 每个线程攒私有部分和（可读 `float4`）
2. **`blockReduceSum`**（`common.cuh`）：warp `__shfl_xor_sync` → 每 warp 一个数进 smem → 再 warp reduce
3. **每 block 一次** `atomicAdd(out, v)`（不要每线程 atomic）

`*output` 启动前 `cudaMemset` 清 0。grid 取约 `2 × SM`，再大也吃不满 HBM。

## 和 common 的关系

`warpReduceSum` / `blockReduceSum` 已经在 `include/common.cuh`。本题只组核 + `solve`；后面 Softmax / Dot Product / RMSNorm 都复用同一套。
