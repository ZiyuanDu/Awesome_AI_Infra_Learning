# GEMV / 矩阵向量乘法

**来源：** 面试扩展练习；形状、接口和边界行为由本题明确约定。

**练习层级：** 基础

每个输出是矩阵一行与同一个向量的点积，练习小 batch 场景。

$$
y_i=\sum_{k=0}^{K-1}A_{ik}x_k
$$

## 接口

```cuda
void solve(const float* A, const float* x, float* y, int M, int K);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `A` | float32 矩阵，形状 (M, K)。 |
| `x` | float32 向量，长度 K。 |
| `y` | float32 输出，长度 M。 |
| `M` | 输出行数。 |
| `K` | 点积长度。 |

## 示例

A=[[1,2],[3,4]]、x=[2,1]，输出 [4,10]。

## 约束与对照

M,K≥1；行优先连续存储；输入输出分开。

**PyTorch 对照：** `A @ x`

**前置练习：** [reduce/04-dot-product](../../reduce/04-dot-product/README.md) → [reduce/05-row-reduction](../../reduce/05-row-reduction/README.md)
