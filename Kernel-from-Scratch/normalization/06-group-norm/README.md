# GroupNorm / 分组归一化

**LeetGPU：** Medium · [官方题目与参考](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/105_group_normalization)

**练习层级：** 专项

NCHW 输入在每个样本内按通道分组，每组共同统计通道和空间维。

$$
S=(C/G)HW,\qquad \mu_{ng}=\frac1S\sum_{(c,h,w)\in g}x_{nchw},\quad y_{nchw}=\gamma_c(x_{nchw}-\mu_{ng})/\sqrt{v_{ng}+\epsilon}+\beta_c
$$

## 接口

```cuda
void solve(
    const float* X,
    const float* gamma,
    const float* beta,
    float* Y,
    int N,
    int C,
    int H,
    int W,
    int G,
    float eps
);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `X` | float32 输入，形状 (N, C, H, W)。 |
| `gamma` | float32 逐通道缩放，长度 C。 |
| `beta` | float32 逐通道偏置，长度 C。 |
| `Y` | float32 输出，形状 (N, C, H, W)。 |
| `N` | batch 大小。 |
| `C` | 通道数，必须能被 G 整除。 |
| `H` | 空间高度。 |
| `W` | 空间宽度。 |
| `G` | 每个样本的通道分组数。 |
| `eps` | 正稳定项。 |

## 示例

N=1、C=2、H=W=1、G=1，X=[1,3]，gamma=1、beta=0，输出约 [-1,1]。

## 约束与对照

1≤N≤32，1≤C≤1024，1≤H,W≤128，1≤G≤C，C%G=0；eps=1e-5。方差为总体方差。

**PyTorch 对照：** `torch.nn.functional.group_norm(X, G, gamma, beta, eps)`

**前置练习：** [normalization/01-layer-norm](../01-layer-norm/README.md)
