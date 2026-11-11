# Batched GEMM / FP32 批量矩阵乘

**LeetGPU：** Medium · [官方题目与参考](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/30_batched_matrix_multiplication)

**练习层级：** 基础

对每个 batch 独立计算矩阵乘法，三张张量均为 float32。

$$
C_{bij}=\sum_kA_{bik}B_{bkj}
$$

## 接口

```cuda
void solve(const float* A, const float* B, float* C, int BATCH, int M, int N, int K);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `A` | 输入，形状 (BATCH, M, K)。 |
| `B` | 输入，形状 (BATCH, K, N)。 |
| `C` | 输出，形状 (BATCH, M, N)。 |
| `BATCH` | 独立矩阵乘法的批次数；不做 batch 广播。 |
| `M` | 输出矩阵行数。 |
| `N` | 输出矩阵列数。 |
| `K` | 乘积归约维长度。 |

## 示例

BATCH=1、A=[[1,2]]、B=[[3],[4]]，输出 C=[[[11]]]。

## 约束与对照

1≤BATCH≤128，1≤M,N,K≤1024；矩阵在各自 batch 内按行连续存储。

**PyTorch 对照：** `torch.bmm(A, B)`

**前置练习：** [gemm/01-matrix-mul](../01-matrix-mul/README.md)
