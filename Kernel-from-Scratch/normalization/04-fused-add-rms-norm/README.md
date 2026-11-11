# Fused Residual Add + RMSNorm

**LeetGPU：** Medium · [官方题目与参考](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/83_fused_residual_add_rms_norm)

**练习层级：** 融合

先加残差，再按行做 RMSNorm，一次写出最终结果。

$$
z=x+residual,\qquad out_{ij}=z_{ij}w_j/\sqrt{\frac1C\sum_kz_{ik}^2+\epsilon}
$$

## 接口

```cuda
void solve(
    const float* x,
    const float* residual,
    const float* weight,
    float* out,
    int N,
    int C,
    float eps
);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `x` | float32 输入，形状 (N, C)，只读。 |
| `residual` | float32 残差，形状 (N, C)，只读。 |
| `weight` | float32 通道权重，长度 C。 |
| `out` | float32 输出，形状 (N, C)。 |
| `N` | 行数。 |
| `C` | 通道数。 |
| `eps` | 正稳定项。 |

## 示例

x=[[1,1]]，residual=[[2,3]]、weight=[1,1]，归一化对象为 [[3,4]]。

## 约束与对照

1≤N,C≤65536，eps=1e-5。官方计时 N=C=4096。只写 out，不更新 residual。

**前置练习：** [normalization/03-row-rms-norm](../03-row-rms-norm/README.md)
