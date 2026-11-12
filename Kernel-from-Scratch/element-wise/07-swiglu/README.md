# SWiGLU

**LeetGPU：** Easy · [Swish-Gated Linear Unit](https://leetgpu.com/challenges/swish-gated-linear-unit)  

将输入数组分为前半段和后半段，再计算门控结果：

$$
n=N/2,\quad x_1=\mathrm{in}[0:n),\quad x_2=\mathrm{in}[n:2n)
$$

$$
\mathrm{out}[i]=\mathrm{SiLU}(x_1[i])\cdot x_2[i]=\frac{x_1[i]}{1+e^{-x_1[i]}}\,x_2[i]
$$

输出长度 **N/2**。`N` 保证偶数。切法是 **前半 / 后半**。

站点例子：`[1, 2, 3, 4]` → `[SiLU(1)·3, SiLU(2)·4]` = `[2.1931758, 7.0463767]`。

## 接口

```cuda
void solve(const float* input, float* output, int N);
```

所有指针参数均指向 **GPU 设备内存**；数组由调用方分配，尺寸参数和其他标量按值传入。

| 参数 | 含义 |
|---|---|
| `input` | 长度为 N 的只读 float 输入数组，前半段为门控输入，后半段为被乘输入。 |
| `output` | 长度为 N / 2 的 float 输出数组，写入 SiLU(input[i]) × input[N / 2 + i]。 |
| `N` | 完整输入的元素个数，必须为偶数。 |

`1 ≤ N ≤ 1e5`（偶数），输入 ∈ [-100, 100]，计时 `N=1e5`。
