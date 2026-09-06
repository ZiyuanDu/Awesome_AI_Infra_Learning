# Sigmoid Activation（Element-wise）

**LeetGPU：** Easy · [Sigmoid Activation](https://leetgpu.com/challenges/sigmoid-activation)  
**目录：** `element_wise/10_sigmoid`

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

逐点把实数压到 `(0,1)`。例：`[0,1,-1,2] → [0.5, 0.7311, 0.2689, 0.8808]`。

和 [SiLU](https://leetgpu.com/challenges/sigmoid-linear-unit)（`06_silu`）的关系：`SiLU(x)=x·σ(x)`。本题只有 `σ`，少乘一次。

## 接口

```cuda
void solve(const float* input, float* output, int N);
```

`1 ≤ N ≤ 1e8`，计时 `N=5e7`。

## 为什么归 element-wise

每个输出只看同位置 `x`，无邻域、无通信。和 ReLU / SiLU 同一模板：grid-stride + `float4`，公式换成 `1 / (1 + __expf(-x))`。

## 写法

`__expf` 硬件近似，这题精度够。`x` 很负 → 结果 → 0；很正 → 1。N 到 5e7，带宽墙；smem 没用。
