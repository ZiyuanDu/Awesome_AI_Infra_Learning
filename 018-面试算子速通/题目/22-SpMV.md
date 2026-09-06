# Sparse Matrix-Vector Multiplication（GEMV）

**LeetGPU：** Medium · [SpMV](https://leetgpu.com/challenges/sparse-matrix-vector-multiplication)  
**目录：** `gemm/02_spmv`

\(y=Ax\)。站点给的 \(A\) 是 **row-major 稠密数组**（约 60–70% 为零），**不是 CSR**。计时 \(M{=}1000,\ N{=}10^4\)。

## 放哪

`gemm/`：按行点积 = GEMV。复用 `blockReduceSum` / `__ldg`。

## 面试写法

一行一 block 也能过；仓库版把 **x 放进 smem**，block grid-stride 扫多行（A 读一遍、x 每 block 一次），计时点约能贴近单次拷贝 A 的带宽。

`N%4==0` 走 `float4`。**不要**跳零。`nnz` 可忽略。

本地（5090）`M=1000,N=10000`：约 **1.6 TB/s**（对照 `memcpy` A ≈ 1.9 TB/s）。
