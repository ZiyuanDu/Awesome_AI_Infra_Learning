# 2D 卷积

**LeetGPU：** Medium · [2D Convolution](https://leetgpu.com/challenges/2d-convolution)  

valid 边界，单通道：

$$
\mathrm{out}[i,j]=\sum_{m,n}\mathrm{in}[i+m,j+n]\cdot\mathrm{ker}[m,n]
$$

`out_rows = in_rows - k_rows + 1`（cols 同理）。行优先一维数组。

## 接口

```cuda
void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols);
```

所有指针参数均指向 **GPU 设备内存**；数组由调用方分配，尺寸参数和其他标量按值传入。

| 参数 | 含义 |
|---|---|
| `input` | 行优先的 input_rows × input_cols 只读 float 输入矩阵。 |
| `kernel` | 行优先的 kernel_rows × kernel_cols 只读 float 卷积核，按题面顺序使用，不翻转。 |
| `output` | 行优先的 (input_rows - kernel_rows + 1) × (input_cols - kernel_cols + 1) float 输出矩阵。 |
| `input_rows` | 输入行数。 |
| `input_cols` | 输入列数。 |
| `kernel_rows` | 核的行数，不超过 input_rows。 |
| `kernel_cols` | 核的列数，不超过 input_cols；valid 边界，各维步长为 1。 |

`1 ≤ rows,cols ≤ 3072`，`1 ≤ k ≤ 31`。计时 `3072×3072`、`ker 15×15`。
