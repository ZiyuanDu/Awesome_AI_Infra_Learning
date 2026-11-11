# LayerNorm / 按行层归一化

**LeetGPU：** Medium · [官方题目与参考](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/113_layer_normalization)

**练习层级：** 基础

对每个样本的通道维统计均值与总体方差，再做逐通道仿射变换。

$$
\mu_i=\frac1C\sum_jx_{ij},\quad v_i=\frac1C\sum_j(x_{ij}-\mu_i)^2,\quad y_{ij}=\frac{x_{ij}-\mu_i}{\sqrt{v_i+\epsilon}}w_j+b_j
$$

## 接口

```cuda
void solve(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N,
    int C,
    float eps
);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `input` | float32 输入，形状 (N, C)。 |
| `weight` | float32 仿射权重，长度 C。 |
| `bias` | float32 仿射偏置，长度 C。 |
| `output` | float32 输出，形状 (N, C)。 |
| `N` | 样本行数。 |
| `C` | 每行通道数，也是归约长度。 |
| `eps` | 正的稳定项，加入方差后再开平方。 |

## 示例

input=[[1,3]]、weight=[1,1]、bias=[0,0]，输出约 [-1,1]（eps 很小时）。

## 约束与对照

1≤N≤65536，1≤C≤4096，eps=1e-5。方差分母为 C，不做无偏修正。官方计时 N=65536、C=512。

**PyTorch 对照：** `torch.nn.functional.layer_norm(input, (C,), weight, bias, eps)`

**前置练习：** [reduce/05-row-reduction](../../reduce/05-row-reduction/README.md)
