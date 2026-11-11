# GEMM + Bias + GELU / 融合后处理

**来源：** 面试扩展练习；形状、接口和边界行为由本题明确约定。

**练习层级：** 融合

在矩阵乘结果上按列广播偏置，再计算 erf 精确形式 GELU。

$$
Y_{ij}=\mathrm{GELU}\left(\sum_k A_{ik}B_{kj}+bias_j\right)
$$

## 接口

```cuda
void solve(const float* A, const float* B, const float* bias, float* Y, int M, int N, int K);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `A` | float32 输入，形状 (M, K)。 |
| `B` | float32 输入，形状 (K, N)。 |
| `bias` | float32 偏置，长度 N。 |
| `Y` | float32 输出，形状 (M, N)。 |
| `M` | 输出行数。 |
| `N` | 输出列数。 |
| `K` | 点积长度。 |

## 示例

A=[[1]]、B=[[0]]、bias=[0]，输出 [[0]]。

## 约束与对照

M,N,K≥1，输入有限；GELU 使用 erf 形式；输入输出不别名。

**PyTorch 对照：** `torch.nn.functional.gelu(A @ B + bias, approximate='none')`

**前置练习：** [gemm/01-matrix-mul](../01-matrix-mul/README.md) → [element-wise/09-gelu](../../element-wise/09-gelu/README.md) → [element-wise/12-bias-add](../../element-wise/12-bias-add/README.md)
