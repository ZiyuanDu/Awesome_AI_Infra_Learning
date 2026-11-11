# RMSNorm Backward / 按行反向

**LeetGPU：** Medium · [官方题目与参考](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/142_rms_norm_backward)

**练习层级：** 专项

对应按行、逐通道权重版本的 RMSNorm，输出输入梯度和权重梯度。

$$
r=\mathrm{rsqrt}(\mathrm{mean}_C(x^2)+\epsilon),\quad\hat x=xr,\quad g=dy\,w
dx=r(g-\hat x\,\mathrm{mean}_C(g\hat x)),\qquad dw=\sum_Ndy\,\hat x
$$

## 接口

```cuda
void solve(
    const float* x,
    const float* weight,
    const float* dy,
    float* dx,
    float* dweight,
    int N,
    int C,
    float eps
);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `x` | float32 输入，形状 (N, C)。 |
| `weight` | float32 通道权重，长度 C。 |
| `dy` | float32 上游梯度，形状 (N, C)。 |
| `dx` | float32 输出，形状 (N, C)。 |
| `dweight` | float32 输出，长度 C；沿 N 求和。 |
| `N` | 行数。 |
| `C` | 通道数。 |
| `eps` | 与前向相同的正稳定项。 |

## 示例

dy 全 0 时，dx 和 dweight 均为 0。

## 约束与对照

1≤N≤65536，1≤C≤16384，1e-6≤eps≤1；所有输出从本次输入重新计算。

**前置练习：** [normalization/03-row-rms-norm](../03-row-rms-norm/README.md)
