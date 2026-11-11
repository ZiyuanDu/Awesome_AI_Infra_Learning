# 交错合并数组

**LeetGPU：** Easy · [Interleave Arrays](https://leetgpu.com/challenges/interleave-arrays)  

给定两个长度为 N 的数组 A 和 B，将其交错合并到长度为 2N 的输出数组：

$$
\mathrm{out}[2i]=A[i],\quad \mathrm{out}[2i+1]=B[i]
$$

例：`A=[1,2,3], B=[4,5,6] → [1,4,2,5,3,6]`。

## 接口

```cuda
void solve(const float* A, const float* B, float* output, int N);
```

所有指针参数均指向 **GPU 设备内存**；数组由调用方分配，尺寸参数和其他标量按值传入。

| 参数 | 含义 |
|---|---|
| `A` | 长度为 N 的只读 float 输入数组。 |
| `B` | 长度为 N 的只读 float 输入数组。 |
| `output` | 长度为 2 × N 的 float 输出数组，依次存储 A[0]、B[0]、A[1]、B[1] 等。 |
| `N` | 每个输入数组的元素个数，输出长度为 2 × N。 |

`1 ≤ N ≤ 5e7`，计时 `N=2.5e7`。输出长度 `2N`。
