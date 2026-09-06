# SiLU / Sigmoid Linear Unit（Element-wise）

**LeetGPU：** Easy · [Sigmoid Linear Unit](https://leetgpu.com/challenges/sigmoid-linear-unit)  
**目录：** `element_wise/06_silu`

$$
\sigma(x)=\frac{1}{1+e^{-x}},\qquad \mathrm{SiLU}(x)=x\cdot\sigma(x)=\frac{x}{1+e^{-x}}
$$

也叫 **Swish**。只有一个输入。  
[SWiGLU](https://leetgpu.com/challenges/swish-gated-linear-unit) 是另一题：切两半，`SiLU(x1)*x2`，见 `07_swiglu`。

## 接口

```cuda
void solve(const float* input, float* output, int N);
```

`1 ≤ N ≤ 10000`（计时却是 `N=50000`），输入 ∈ [-100, 100]。

## 为什么在 element_wise

每个输出只看同位置 `x`，没有邻域、没有通信。和 ReLU 同一套：grid-stride + `float4`，公式换成一行 `x / (1 + __expf(-x))`。

不要放 gemm（那是乘加），也不要一上来写 SwiGLU。

## 面试写法

`__expf` 走硬件近似，比 `expf` 快，这题精度够。`x` 很负时 `e^{-x}` 涨成 inf，除法得到 0，和真值一致。N 很小，已经是带宽/指令混着吃；再加 smem 没用。
