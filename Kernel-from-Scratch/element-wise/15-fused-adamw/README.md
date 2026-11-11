# Fused AdamW / 融合优化器更新

**LeetGPU：** Medium · [官方题目与参考](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/127_fused_adamw)

**练习层级：** 专项

执行一个优化器步骤，原地更新参数和一、二阶矩；先应用解耦的权重衰减。

$$
\theta\leftarrow\theta(1-\eta\lambda),\quad m\leftarrow\beta_1m+(1-\beta_1)g,\quad v\leftarrow\beta_2v+(1-\beta_2)g^2
\hat m=m/(1-\beta_1^t),\quad\hat v=v/(1-\beta_2^t),\quad\theta\leftarrow\theta-\eta\hat m/(\sqrt{\hat v}+\epsilon)
$$

## 接口

```cuda
void solve(
    float* params,
    const float* grad,
    float* m,
    float* v,
    int N,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int t
);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `params` | float32 参数，长度 N，原地更新。 |
| `grad` | float32 只读梯度，长度 N。 |
| `m` | float32 一阶矩，长度 N，原地更新。 |
| `v` | float32 非负二阶矩，长度 N，原地更新。 |
| `N` | 元素个数。 |
| `lr` | 学习率 η。 |
| `beta1` | 一阶矩衰减系数，0≤beta1<1。 |
| `beta2` | 二阶矩衰减系数，0≤beta2<1。 |
| `eps` | 正数，位于平方根外的分母稳定项。 |
| `weight_decay` | 解耦权重衰减系数 λ。 |
| `t` | 从 1 开始的训练步数。 |

## 示例

单个参数 θ=1、g=1、m=v=0，t=1、lr=0.1、weight_decay=0，则更新后 θ≈0.9。

## 约束与对照

1≤N≤16777216，t≥1，v≥0，eps>0。官方计时 N=16777216、t=1000。

没有单独输出缓冲区；参数、m、v 都必须按本轮更新后的状态写回。
