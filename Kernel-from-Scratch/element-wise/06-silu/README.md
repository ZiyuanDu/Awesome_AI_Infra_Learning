# SiLU

**LeetGPU：** Easy · [Sigmoid Linear Unit](https://leetgpu.com/challenges/sigmoid-linear-unit)  

$$
\sigma(x)=\frac{1}{1+e^{-x}},\qquad \mathrm{SiLU}(x)=x\cdot\sigma(x)=\frac{x}{1+e^{-x}}
$$

也叫 **Swish**，对输入数组逐元素计算。

## 接口

```cuda
void solve(const float* input, float* output, int N);
```

所有指针参数均指向 **GPU 设备内存**；数组由调用方分配，尺寸参数和其他标量按值传入。

| 参数 | 含义 |
|---|---|
| `input` | 长度为 N 的只读 float 输入数组。 |
| `output` | 长度为 N 的 float 输出数组，写入 x / (1 + exp(-x))。 |
| `N` | 输入和输出的元素个数。 |

`1 ≤ N ≤ 10000`（计时却是 `N=50000`），输入 ∈ [-100, 100]。
