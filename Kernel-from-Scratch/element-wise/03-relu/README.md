# ReLU

**LeetGPU：** Easy · [ReLU](https://leetgpu.com/challenges/relu)  

$$
\mathrm{ReLU}(x)=\max(0,x)
$$

对 `input[N]` 逐点写到 `output`。例：`[-2,-1,0,1,2] → [0,0,0,1,2]`。

## 接口

```cuda
void solve(const float* input, float* output, int N);
```

所有指针参数均指向 **GPU 设备内存**；数组由调用方分配，尺寸参数和其他标量按值传入。

| 参数 | 含义 |
|---|---|
| `input` | 长度为 N 的只读 float 输入数组。 |
| `output` | 长度为 N 的 float 输出数组，写入 max(0, input[i])。 |
| `N` | 输入和输出的元素个数。 |

`1 ≤ N ≤ 1e8`，计时 `N=2.5e7`。
