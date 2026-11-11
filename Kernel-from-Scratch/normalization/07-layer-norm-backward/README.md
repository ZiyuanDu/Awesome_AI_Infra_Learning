# LayerNorm Backward / 带保存统计量的反向

**LeetGPU：** Medium · [官方题目与参考](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/144_layer_norm_backward)

**练习层级：** 专项

使用前向保存的均值和逆标准差，计算输入与仿射参数梯度。

$$
\hat x=(x-\mu)r,\quad g=dy\,\gamma,\quad dx=r(g-\mathrm{mean}_C(g)-\hat x\,\mathrm{mean}_C(g\hat x))
d\gamma=\sum_Ndy\,\hat x,\qquad d\beta=\sum_Ndy
$$

## 接口

```cuda
void solve(
    const float* x,
    const float* gamma,
    const float* mean,
    const float* rstd,
    const float* dy,
    float* dx,
    float* dgamma,
    float* dbeta,
    int N,
    int C
);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `x` | float32 原输入，形状 (N, C)。 |
| `gamma` | float32 前向权重，长度 C。 |
| `mean` | float32 前向行均值，长度 N。 |
| `rstd` | float32 前向逆标准差，长度 N，等于 rsqrt(var+eps)。 |
| `dy` | float32 上游梯度，形状 (N, C)。 |
| `dx` | float32 输出，输入梯度，形状 (N, C)。 |
| `dgamma` | float32 输出，长度 C，沿所有行求和。 |
| `dbeta` | float32 输出，长度 C，沿所有行求和。 |
| `N` | 行数。 |
| `C` | 通道数。 |

## 示例

dy 全 0 时，dx、dgamma、dbeta 都应为 0。

## 约束与对照

1≤N≤65536，1≤C≤16384；调用方提供与 x、前向 eps 一致的 mean/rstd。梯度输出初值未定义，必须完整覆盖。

本题同时包含行归约和跨行的参数梯度归约；不能把 dgamma/dbeta 仅写成某一行的结果。

**前置练习：** [normalization/01-layer-norm](../01-layer-norm/README.md)
