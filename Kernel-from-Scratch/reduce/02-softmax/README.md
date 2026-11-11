# Softmax

**LeetGPU：** Medium · [Softmax](https://leetgpu.com/challenges/softmax)  

一维向量（不是按行矩阵）：

$$
\sigma(x)_i=\frac{e^{x_i-m}}{\ell},\quad m=\max_j x_j,\quad \ell=\sum_j e^{x_j-m}
$$

例：`[1,2,3] → ≈[0.090, 0.244, 0.665]`。站点要求用 **max trick** 防溢出。

## 接口

```cuda
void solve(const float* input, float* output, int N);
```

所有指针参数均指向 **GPU 设备内存**；数组由调用方分配，尺寸参数和其他标量按值传入。

| 参数 | 含义 |
|---|---|
| `input` | 长度为 N 的只读 float 输入向量；对整个向量做 softmax。 |
| `output` | 长度为 N 的 float 输出数组，写入数值稳定的归一化结果。 |
| `N` | 输入和输出的元素个数。 |

`1 ≤ N ≤ 5e5`，计时 `N=5e5`。
