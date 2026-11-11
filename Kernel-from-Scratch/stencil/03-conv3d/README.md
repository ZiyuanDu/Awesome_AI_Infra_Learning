# 3D 卷积

**LeetGPU：** Medium · [3D Convolution](https://leetgpu.com/challenges/3d-convolution)  

$$
\mathrm{out}[i,j,k]=\sum_{d,r,c}\mathrm{in}[i+d,j+r,k+c]\cdot\mathrm{ker}[d,r,c]
$$

布局：**depth 最外**：`idx = ((d * rows + r) * cols + c)`。

## 接口

```cuda
void solve(const float* input, const float* kernel, float* output,
           int input_depth, int input_rows, int input_cols,
           int kernel_depth, int kernel_rows, int kernel_cols);
```

所有指针参数均指向 **GPU 设备内存**；数组由调用方分配，尺寸参数和其他标量按值传入。

| 参数 | 含义 |
|---|---|
| `input` | 形状为 input_depth × input_rows × input_cols 的只读 float 数组，列维连续。 |
| `kernel` | 形状为 kernel_depth × kernel_rows × kernel_cols 的只读 float 数组，按题面顺序使用，不翻转。 |
| `output` | float 输出数组；各维长度分别为 input_* - kernel_* + 1，布局与输入相同。 |
| `input_depth` | 输入深度。 |
| `input_rows` | 输入行数。 |
| `input_cols` | 输入列数。 |
| `kernel_depth` | 核的深度，不超过 input_depth。 |
| `kernel_rows` | 核的行数，不超过 input_rows。 |
| `kernel_cols` | 核的列数，不超过 input_cols；valid 边界，各维步长为 1。 |

`1 ≤ 各维 ≤ 256`，`1 ≤ k ≤ 5`。计时约 `128³`、`ker 5³`。
