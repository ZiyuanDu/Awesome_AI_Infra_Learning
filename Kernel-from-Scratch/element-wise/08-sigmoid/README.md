# Sigmoid

**LeetGPU：** Easy · [Sigmoid Activation](https://leetgpu.com/challenges/sigmoid-activation)  

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

逐点把实数压到 `(0,1)`。例：`[0,1,-1,2] → [0.5, 0.7311, 0.2689, 0.8808]`。

## 接口

```cuda
void solve(const float* input, float* output, int N);
```

所有指针参数均指向 **GPU 设备内存**；数组由调用方分配，尺寸参数和其他标量按值传入。

| 参数 | 含义 |
|---|---|
| `input` | 长度为 N 的只读 float 输入数组。 |
| `output` | 长度为 N 的 float 输出数组，写入 1 / (1 + exp(-x))。 |
| `N` | 输入和输出的元素个数。 |

`1 ≤ N ≤ 1e8`，计时 `N=5e7`。
