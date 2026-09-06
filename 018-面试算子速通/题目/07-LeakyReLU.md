# Leaky ReLU（Element-wise）

**LeetGPU：** Easy · [Leaky ReLU](https://leetgpu.com/challenges/leaky-relu)  
**目录：** `element_wise/04_leaky_relu`

$$
f(x)=\begin{cases}x & x>0\\ \alpha x & x\le 0\end{cases},\quad \alpha=0.01
$$

和 ReLU 同一模板，负半轴留一条小斜率，避免 dying ReLU。例：`[-2,-1,0,1,2] → [-0.02,-0.01,0,1,2]`。

## 接口

```cuda
void solve(const float* input, float* output, int N);
```

`α` **写死 0.01**，不进参数。`1 ≤ N ≤ 1e8`，计时 `N=5e7`。

## 写法

`fmaxf(x, αx)` 与分段等价（`α<1`），无 `if`。其余照抄 ReLU：`float4` + grid-stride + 标量尾巴。
