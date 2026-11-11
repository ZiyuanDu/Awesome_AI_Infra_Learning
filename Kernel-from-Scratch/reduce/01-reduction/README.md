# Reduction

**LeetGPU：** Medium · [Reduction](https://leetgpu.com/challenges/reduction)  

$$
\mathrm{out}=\sum_{i=0}^{N-1}x[i]
$$

例：`[1..8] → 36`，`[-2.5,1.5,-1,2] → 0`。

## 接口

```cuda
void solve(const float* input, float* output, int N);
```

所有指针参数均指向 **GPU 设备内存**；数组由调用方分配，尺寸参数和其他标量按值传入。

| 参数 | 含义 |
|---|---|
| `input` | 长度为 N 的只读 float 输入数组。 |
| `output` | 长度为 1 的 float 输出数组，保存所有输入元素的和。 |
| `N` | 输入元素个数。 |

`output` 是 **长度为 1** 的 device 标量。`1 ≤ N ≤ 1e8`，计时 `N=2^{22}`。和必须能放进 float。
