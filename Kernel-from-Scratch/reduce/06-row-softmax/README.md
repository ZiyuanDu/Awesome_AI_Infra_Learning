# Row-wise Softmax / 按行 Softmax

**来源：** 面试扩展练习；形状、接口和边界行为由本题明确约定。

**练习层级：** 基础

每行独立归一化，不同于已有的整条一维向量 Softmax。

$$
m_i=\max_jX_{ij},\qquad Y_{ij}=\frac{\exp(X_{ij}-m_i)}{\sum_k\exp(X_{ik}-m_i)}
$$

## 接口

```cuda
void solve(const float* input, float* output, int M, int N);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `input` | float32 输入，形状 (M, N)。 |
| `output` | float32 输出，形状 (M, N)。 |
| `M` | 行数。 |
| `N` | 每行归一化的元素个数。 |

## 示例

input=[[0,0],[1,1]]，output=[[0.5,0.5],[0.5,0.5]]。

## 约束与对照

M,N≥1，输入有限。每行独立使用减最大值的稳定计算。

**PyTorch 对照：** `torch.softmax(input, dim=1)`

**前置练习：** [reduce/05-row-reduction](../05-row-reduction/README.md) → [reduce/02-softmax](../02-softmax/README.md)
