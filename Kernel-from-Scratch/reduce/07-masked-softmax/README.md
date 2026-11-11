# Masked Softmax / 带掩码的按行归一化

**来源：** 面试扩展练习；形状、接口和边界行为由本题明确约定。

**练习层级：** 基础

只对每行 mask=1 的位置做 Softmax；屏蔽位置为 0，并明确定义全屏蔽行。

有有效元素时，m_i 是该行所有 mask=1 位置的最大值；全屏蔽行直接输出全 0。

$$
Y_{ij}=\begin{cases}\exp(X_{ij}-m_i)/\sum_{k:\,mask_{ik}=1}\exp(X_{ik}-m_i)&mask_{ij}=1\\0&mask_{ij}=0\end{cases}
$$

## 接口

```cuda
void solve(const float* input, const unsigned char* mask, float* output, int M, int N);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `input` | float32 输入，形状 (M, N)。 |
| `mask` | uint8 输入，形状 (M, N)；1 表示保留，0 表示屏蔽。 |
| `output` | float32 输出，形状 (M, N)；全屏蔽行输出全 0。 |
| `M` | 行数。 |
| `N` | 每行列数。 |

## 示例

input=[[0,0],[1,2]]，mask=[[1,0],[0,0]]，输出 [[1,0],[0,0]]。

## 约束与对照

M,N≥1；输入有限；mask 只取 0 或 1。全屏蔽行必须单独处理，不能产生 NaN。

**PyTorch 对照：** `有效位置按行归一化；对全屏蔽行显式置零。`

**前置练习：** [reduce/06-row-softmax](../06-row-softmax/README.md)
