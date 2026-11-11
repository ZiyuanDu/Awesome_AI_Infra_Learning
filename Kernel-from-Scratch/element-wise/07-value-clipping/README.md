# Value Clipping

**LeetGPU：** Easy · [Value Clipping](https://leetgpu.com/challenges/value-clipping)  

$$
\mathrm{clip}(x)=\min(\max(x,\mathrm{lo}),\mathrm{hi})
$$

对 `input[N]` 逐点钳到 `[lo, hi]`。例：`[1.5,-2,3,4.5], lo=0, hi=3.5 → [1.5,0,3,3.5]`。

常用于激活稳定、量化前饱和。

## 接口

```cuda
void solve(const float* input, float* output, int N, float lo, float hi);
```

所有指针参数均指向 **GPU 设备内存**；数组由调用方分配，尺寸参数和其他标量按值传入。

| 参数 | 含义 |
|---|---|
| `input` | 长度为 N 的只读 float 输入数组。 |
| `output` | 长度为 N 的 float 输出数组，将输入限制在 [lo, hi]。 |
| `N` | 输入和输出的元素个数。 |
| `lo` | 裁剪下界，float 标量。 |
| `hi` | 裁剪上界，float 标量，要求 lo ≤ hi。 |

`1 ≤ N ≤ 1e5`，`lo ≤ hi`，计时 `N=1e5`。
