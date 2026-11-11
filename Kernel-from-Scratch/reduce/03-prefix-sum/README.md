# Prefix Sum

**LeetGPU：** Medium · [Prefix Sum](https://leetgpu.com/challenges/prefix-sum)  

Inclusive：`out[i]=in[0]+…+in[i]`。例：`[1,2,3,4]→[1,3,6,10]`。

## 接口

```cuda
void solve(const float* input, float* output, int N);
```

所有指针参数均指向 **GPU 设备内存**；数组由调用方分配，尺寸参数和其他标量按值传入。

| 参数 | 含义 |
|---|---|
| `input` | 长度为 N 的只读 float 输入数组。 |
| `output` | 长度为 N 的 float 输出数组，output[i] 包含 input[0] 到 input[i] 的和。 |
| `N` | 输入和输出的元素个数。 |

`1 ≤ N ≤ 1e8`，计时 `N=2.5e5`。
