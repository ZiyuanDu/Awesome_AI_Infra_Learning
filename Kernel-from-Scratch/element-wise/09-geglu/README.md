# GEGLU

**LeetGPU：** Easy · [Gaussian Error Gated Linear Unit](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/easy/65_geglu)

将输入向量分为前半段和后半段，对后半段应用 GELU，再与前半段逐元素相乘：

$$
n=N/2,\quad x_1=\mathrm{input}[0:n),\quad x_2=\mathrm{input}[n:N)
$$

$$
\mathrm{output}[i]=x_1[i]\cdot\mathrm{GELU}(x_2[i])
=x_1[i]\cdot\frac{x_2[i]}{2}\left(1+\mathrm{erf}\!\left(\frac{x_2[i]}{\sqrt{2}}\right)\right)
$$

输出长度 **N/2**，`N` 保证偶数。输入按 **前半 / 后半** 切分，GELU 作用在 **后半段**；使用 erf 精确形式。

## 接口

```cuda
void solve(const float* input, float* output, int N);
```

所有指针参数均指向 **GPU 设备内存**；数组由调用方分配，尺寸参数和其他标量按值传入。

| 参数 | 含义 |
|---|---|
| `input` | 长度为 N 的只读 float 向量，前半段为乘数，后半段为 GELU 的输入。 |
| `output` | 长度为 N / 2 的 float 输出向量，写入 input[i] × GELU(input[N / 2 + i])。 |
| `N` | 完整输入的元素个数，必须为偶数。 |

## 示例

`[2, -1, 1, 0.5] → ≈[1.6826895, -0.3457312]`。

## 约束

- `2 ≤ N ≤ 1e6`，`N` 为偶数。
- 输入范围为 `[-100, 100]`，输入和输出均为 float32。
- 计时规模：`N=1e6`。
