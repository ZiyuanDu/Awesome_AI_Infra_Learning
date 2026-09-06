# Dot Product（Reduce）

**LeetGPU：** Medium · [Dot Product](https://leetgpu.com/challenges/dot-product)  
**目录：** `reduce/04_dot_product`

$$
\mathrm{out}=\sum_{i=0}^{N-1} A[i]\cdot B[i]
$$

## 放哪

**`reduce/`，紧挨 `01_reduction`。** 不是 element-wise，也不是 GEMM。

| 类别 | 适不适合 | 原因 |
|---|---|---|
| **reduce** | **是** | 多→1；骨架与 Reduction 相同，只是累加换成 `A[i]*B[i]` |
| element-wise | 否 | 若先写 `C[i]=A[i]*B[i]` 再 reduce，多一整次写带宽 |
| GEMM | 否 | 这是 1D 内积，不是矩阵 tile；口播可说「GEMM 的特例 / epilogue」 |

## 高效点

1. **融合**：grid-stride 里直接 `v += a*b`，不物化乘积
2. **`float4`** 两路同读
3. 复用 `blockReduceSum` + 每 block 一次 `atomicAdd`
4. `*output` 启动前清 0

相对「先 map 再 reduce」：少写 `N` 个 float，带宽从约 `3N` 降到 `2N`。
