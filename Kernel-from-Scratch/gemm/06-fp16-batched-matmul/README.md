# Batched GEMM / FP16 输入输出

**LeetGPU：** Medium · [官方题目与参考](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/57_fp16_batched_matmul)

**练习层级：** 专项

同样的批量矩阵乘法，输入输出为 FP16，累加使用 FP32，最终转换为 FP16。

$$
C=\mathrm{FP16}(\mathrm{FP32}(A)\,\mathrm{FP32}(B))
$$

## 接口

```cuda
void solve(const half* A, const half* B, half* C, int BATCH, int M, int N, int K);
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

BATCH=1、A=[[[1,2]]]、B=[[[3],[4]]]，FP16 输出 [[[11]]]。

## 约束与对照

1≤BATCH≤128，1≤M,N,K≤1024。输入输出为 half；以 FP32 累加结果转 FP16 作对照。

**PyTorch 对照：** `torch.bmm(A.float(), B.float()).half()`

**前置练习：** [gemm/05-batched-matmul](../05-batched-matmul/README.md)
