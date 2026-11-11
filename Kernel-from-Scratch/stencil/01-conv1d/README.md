# 1D 卷积

**LeetGPU：** Easy · [1D Convolution](https://leetgpu.com/challenges/1d-convolution)  

给定 `input[N]`、`kernel[K]`，**valid** 边界：

$$
\mathrm{out}[i] = \sum_{k=0}^{K-1} \mathrm{input}[i+k]\cdot\mathrm{kernel}[k],\quad 0 \le i < N-K+1
$$

## 接口

```cuda
void solve(const float* input, const float* kernel, float* output,
           int input_size, int kernel_size);
```

所有指针参数均指向 **GPU 设备内存**；数组由调用方分配，尺寸参数和其他标量按值传入。

| 参数 | 含义 |
|---|---|
| `input` | 长度为 input_size 的只读 float 输入数组。 |
| `kernel` | 长度为 kernel_size 的只读 float 卷积核，按题面顺序使用，不翻转。 |
| `output` | 长度为 input_size - kernel_size + 1 的 float 输出数组。 |
| `input_size` | 输入长度。 |
| `kernel_size` | 卷积核长度，要求 1 ≤ kernel_size ≤ input_size；valid 边界，步长为 1。 |

device 指针。约束：`N ≤ 1.5e6`，`K ≤ 2047`。计时规模 `N=1.5e6, K=2047`。不许用 cuDNN / cuFFT / CUTLASS。
