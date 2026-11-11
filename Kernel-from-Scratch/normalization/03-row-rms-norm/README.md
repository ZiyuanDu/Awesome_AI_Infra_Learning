# Row-wise RMSNorm / 按行均方根归一化

**来源：** 面试扩展练习；形状、接口和边界行为由本题明确约定。

**练习层级：** 基础

练习 LLM 中常见的逐行、逐通道权重版本；不减均值，也没有偏置。

$$
r_i=(\frac1C\sum_jx_{ij}^2+\epsilon)^{-1/2},\qquad y_{ij}=x_{ij}r_iw_j
$$

## 接口

```cuda
void solve(const float* input, const float* weight, float* output, int N, int C, float eps);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `input` | float32 输入，形状 (N, C)。 |
| `weight` | float32 只读权重，长度 C。 |
| `output` | float32 输出，形状 (N, C)。 |
| `N` | 行数。 |
| `C` | 每行归约的通道数。 |
| `eps` | 正的稳定项。 |

## 示例

input=[[3,4]]，weight=[1,1]、eps≈0，输出约 [[0.848528,1.131371]]。

## 约束与对照

N,C≥1，eps>0，输入有限。使用 FP32 统计均方值。

**PyTorch 对照：** `input * torch.rsqrt((input*input).mean(dim=1, keepdim=True)+eps) * weight`

本题采用按行、逐通道权重的扩展接口；官方一维标量 gamma/beta 版本见前置练习。

**前置练习：** [normalization/02-vector-rms-norm](../02-vector-rms-norm/README.md)
