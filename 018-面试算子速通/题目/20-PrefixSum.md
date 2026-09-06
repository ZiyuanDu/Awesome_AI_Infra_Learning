# Prefix Sum（Reduce / Scan）

**LeetGPU：** Medium · [Prefix Sum](https://leetgpu.com/challenges/prefix-sum)  
**目录：** `reduce/03_prefix_sum`

Inclusive：`out[i]=in[0]+…+in[i]`。例：`[1,2,3,4]→[1,3,6,10]`。

## 接口

```cuda
void solve(const float* input, float* output, int N);
```

`1 ≤ N ≤ 1e8`，计时 `N=2.5e5`。

## 面试默写（三步）

复用 `common.cuh`：`load4` / `store4` / `blockScanInclusive` / `blockScanExclusive` / `blockReduceSum`。

1. **reduce_tile**：每 tile 一个总和（`float4` + `blockReduceSum`）
2. **扫 totals**：`blockScanInclusive`（一层不够再套 tile+add）
3. **scan_tile**：`seed + blockScanExclusive(线程内 4 元和)`，一次写回

口播：`exclusive = inclusive - self`；不要先写局部再 RMW。

## 和 013 的关系

`013-Scan/scan_opt` 更长（递归 exclusive、每线程 8 items）。本题计时点小，面试版够快；大 N 仍是带宽墙。
