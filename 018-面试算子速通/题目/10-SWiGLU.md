# SWiGLU（Element-wise，门控）

**LeetGPU：** Easy · [Swish-Gated Linear Unit](https://leetgpu.com/challenges/swish-gated-linear-unit)  
**目录：** `element_wise/07_swiglu`

**不是 SiLU。** SiLU 是一元 `x·σ(x)`（`06_silu`）。本题把输入切两半再门控：

$$
n=N/2,\quad x_1=\mathrm{in}[0:n),\quad x_2=\mathrm{in}[n:2n)
$$

$$
\mathrm{out}[i]=\mathrm{SiLU}(x_1[i])\cdot x_2[i]=\frac{x_1[i]}{1+e^{-x_1[i]}}\,x_2[i]
$$

输出长度 **N/2**。`N` 保证偶数。切法是 **前半 / 后半**，不是交错。

站点例子：`[1, 2, 3, 4]` → `[SiLU(1)·3, SiLU(2)·4]` = `[2.1931758, 7.0463767]`。

## 接口

```cuda
void solve(const float* input, float* output, int N);
```

`1 ≤ N ≤ 1e5`（偶数），输入 ∈ [-100, 100]，计时 `N=1e5`。

## 和另外两题的关系

| | SiLU | 本题 Easy SWiGLU | Medium SwiGLU MLP |
|---|---|---|---|
| 公式 | `silu(x)` | `silu(x1)*x2` | `(silu(xWg)⊙(xWu)) Wd` |
| 输入 | 1 向量 | 1 向量切半 | 激活 + 三个权重 |
| 写法 | 一元 map | 二元 map | GEMM + 本题 |
| 目录 | `06_silu` | **这里** | 019 / 以后 |

面试先写 SiLU 一行，再加 `* x2` 和切半下标。完整 LLaMA FFN 是 Medium，不要塞进这一题。
