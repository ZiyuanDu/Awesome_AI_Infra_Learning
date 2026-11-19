# BatchNorm / 二维训练前向

**LeetGPU：** Medium · [官方题目与参考](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/40_batch_normalization)

**练习层级：** 专项

沿 batch 维为每个通道计算统计量，区别于 LayerNorm 的按行统计。

$$
\mu_j=\frac1N\sum_ix_{ij},\quad v_j=\frac1N\sum_i(x_{ij}-\mu_j)^2,\quad y_{ij}=\gamma_j(x_{ij}-\mu_j)/\sqrt{v_j+\epsilon}+\beta_j
$$

## 接口

```cuda
void solve(
    const float* input,
    const float* gamma,
    const float* beta,
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
| `gamma` | float32 缩放向量，长度 C。 |
| `beta` | float32 偏置向量，长度 C。 |
| `output` | float32 输出，形状 (N, C)。 |
| `N` | batch 大小，也是统计量归约长度。 |
| `C` | 通道数。 |
| `eps` | 正稳定项。 |

## 示例

input=[[1,2],[3,4]]，gamma=[1,1]、beta=[0,0]，输出约 [[-1,-1],[1,1]]。

## 约束与对照

1≤N≤10000，1≤C≤1024，eps=1e-5；使用总体方差。此接口不维护 running mean/variance。

**前置练习：** [normalization/01-layer-norm](../01-layer-norm/README.md)
