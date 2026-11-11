# RMSNorm / 官方一维标量仿射版本

**LeetGPU：** Medium · [官方题目与参考](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/50_rms_normalization)

**练习层级：** 基础

对整个向量求均方根，再用标量 gamma 和 beta 变换。与下一题按行、逐通道权重的 RMSNorm 分开。

$$
r=\sqrt{\frac1N\sum_ix_i^2+\epsilon},\qquad y_i=\gamma x_i/r+\beta
$$

## 接口

```cuda
void solve(const float* input, float gamma, float beta, float* output, int N, float eps);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `input` | float32 输入，长度 N。 |
| `gamma` | float 标量缩放系数；不是指针。 |
| `beta` | float 标量偏置；不是指针。 |
| `output` | float32 输出，长度 N。 |
| `N` | 向量长度。 |
| `eps` | 加到均方值上的正稳定项。 |

## 示例

input=[3,4]、gamma=1、beta=0、eps≈0，输出约 [0.848528,1.131371]。

## 约束与对照

1≤N≤100000，eps=1e-5；输入在 [-100,100]。官方计时 N=100000。

**PyTorch 对照：** `gamma * input / sqrt(mean(input**2)+eps) + beta`

**前置练习：** [reduce/01-reduction](../../reduce/01-reduction/README.md)
