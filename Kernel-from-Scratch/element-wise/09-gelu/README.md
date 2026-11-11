# GELU

**LeetGPU：** Easy · [GELU Activation](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/easy/124_gelu)

对输入向量逐元素计算 Gaussian Error Linear Unit，结果写入输出向量：

$$
\mathrm{GELU}(x)=x\Phi(x)=\frac{x}{2}\left(1+\mathrm{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right)
$$

本题使用 **erf 精确形式**。PyTorch 对照为 `torch.nn.functional.gelu(X, approximate="none")`，不使用 tanh 近似。

## 接口

```cuda
void solve(const float* X, float* Y, int N);
```

所有指针参数均指向 **GPU 设备内存**；数组由调用方分配，尺寸参数和其他标量按值传入。

| 参数 | 含义 |
|---|---|
| `X` | 长度为 N 的只读 float 输入向量。 |
| `Y` | 长度为 N 的 float 输出向量，写入 GELU(X[i])。 |
| `N` | 输入和输出的元素个数。 |

## 示例

`[-1, 0, 1, 2] → ≈[-0.1586553, 0, 0.8413447, 1.9544997]`。

## 约束

- `1 ≤ N ≤ 1e8`。
- 输入为有限的 32 位浮点数。
- 计时规模：`N=5e7`。
