# Leaky ReLU

**LeetGPU：** Easy · [Leaky ReLU](https://leetgpu.com/challenges/leaky-relu)  

$$
f(x)=\begin{cases}x & x>0\\ \alpha x & x\le 0\end{cases},\quad \alpha=0.01
$$

对输入逐元素计算上述函数。例：`[-2,-1,0,1,2] → [-0.02,-0.01,0,1,2]`。

## 接口

```cuda
void solve(const float* input, float* output, int N);
```

所有指针参数均指向 **GPU 设备内存**；数组由调用方分配，尺寸参数和其他标量按值传入。

| 参数 | 含义 |
|---|---|
| `input` | 长度为 N 的只读 float 输入数组。 |
| `output` | 长度为 N 的 float 输出数组；负半轴斜率固定为 0.01。 |
| `N` | 输入和输出的元素个数。 |

`α` **写死 0.01**，不进参数。`1 ≤ N ≤ 1e8`，计时 `N=5e7`。
