# Fused AdamW / 融合优化器更新

**LeetGPU Medium** · **专项练习** · [官方题目与参考](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/127_fused_adamw)

给定长度为 `N` 的参数 `params`、梯度 `grad`、一阶矩 `m` 和二阶矩 `v`，在一个 GPU kernel 中完成第 `t` 步 AdamW 更新。将结果写回 `params`、`m` 和 `v`，`grad` 只读。

对于每个下标 `i`，设 $\theta$、$g$、$m$、$v$ 为四个数组在该位置的输入值。$\eta$、$\lambda$、$\beta_1$、$\beta_2$、$\epsilon$ 分别对应 `lr`、`weight_decay`、`beta1`、`beta2`、`eps`。

$$
\begin{aligned}
\theta_\mathrm{decay} &= \theta(1-\eta\lambda) \\
m' &= \beta_1m + (1-\beta_1)g \\
v' &= \beta_2v + (1-\beta_2)g^2 \\
\hat m &= \frac{m'}{1-\beta_1^t}, \qquad
\hat v = \frac{v'}{1-\beta_2^t} \\
\theta' &= \theta_\mathrm{decay} - \eta\frac{\hat m}{\sqrt{\hat v}+\epsilon}
\end{aligned}
$$

权重衰减直接作用于原参数。写回的三个值是 $\theta'$、$m'$、$v'$。偏差校正后的 $\hat m$、$\hat v$ 用于计算本轮步长。

## 示例

**输入**

```text
N = 1, t = 1
params = [2.0], grad = [2.0], m = [0.0], v = [0.0]
lr = 0.25, beta1 = 0.5, beta2 = 0.5, eps = 1e-8, weight_decay = 0.2
```

**输出（原地更新）**

```text
params ≈ [1.65], m = [1.0], v = [2.0]
```

![单个参数的 AdamW 更新。参数 2 先乘 0.95 得到 1.9。梯度 2 更新两阶矩，再算出约 0.25 的步长。最终写回参数约 1.65、一阶矩 1、二阶矩 2。](./fused-adamw-step.svg)

权重衰减得到 $2 \times 0.95 = 1.9$。此时两个偏差校正分母都是 $0.5$，所以 $\hat m = 2$、$\hat v = 4$，参数再减去约 $0.25$。图中的浅蓝色框表示三个写回值。

## 约束

- `1 <= N <= 16,777,216`，`1 <= t <= 100,000`。
- `-10 <= params[i], grad[i] <= 10`，`v[i] >= 0`。
- `0 < lr <= 1`，`0 <= beta1, beta2 < 1`，`0 <= weight_decay <= 1`，`1e-12 <= eps <= 1e-2`。
- 四个数组均为 `float32`，由调用方分配在 GPU 上，连续存储且互不重叠。
- 官方计时规模为 `N = 16,777,216`，`t = 1,000`。

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

## Benchmark

在官方尺寸上对比自写 CUDA、`torch.optim.AdamW(fused=True)` 和手写 PyTorch 张量运算。三种实现从相同的状态执行第 `1000` 步，先核对 `params`、`m`、`v`，再用 CUDA event 测量单步耗时并取中位数。每次计时前恢复初始状态。

```bash
python Kernel-from-Scratch/element-wise/10-fused-adamw/bench.py
```
