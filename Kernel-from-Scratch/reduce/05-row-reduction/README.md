# Row-wise Reduction / 行求和与最大值

**来源：** 面试扩展练习；形状、接口和边界行为由本题明确约定。

**练习层级：** 基础

对矩阵每行同时输出和与最大值，为按行 Softmax、归一化建立基础。

$$
s_i=\sum_j X_{ij},\qquad m_i=\max_j X_{ij}
$$

## 接口

```cuda
void solve(const float* input, float* row_sum, float* row_max, int M, int N);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `input` | float32 输入，形状 (M, N)。 |
| `row_sum` | float32 输出，长度 M。 |
| `row_max` | float32 输出，长度 M。 |
| `M` | 行数。 |
| `N` | 每行列数。 |

## 示例

input=[[1,2,3],[-1,-2,-3]]，row_sum=[6,-6]，row_max=[3,-1]。

## 约束与对照

M,N≥1；输入有限。重点覆盖 N=1、非 32/256 整数倍，以及长行。

**PyTorch 对照：** `input.sum(dim=1)，input.amax(dim=1)。`

**前置练习：** [reduce/01-reduction](../01-reduction/README.md)
