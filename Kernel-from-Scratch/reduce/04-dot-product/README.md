# Dot Product

**LeetGPU：** Medium · [Dot Product](https://leetgpu.com/challenges/dot-product)  

$$
\mathrm{out}=\sum_{i=0}^{N-1} A[i]\cdot B[i]
$$

## 接口

```cuda
void solve(const float* A, const float* B, float* output, int N);
```

所有指针参数均指向 **GPU 设备内存**；数组由调用方分配，尺寸参数和其他标量按值传入。

| 参数 | 含义 |
|---|---|
| `A` | 长度为 N 的只读 float 输入向量。 |
| `B` | 长度为 N 的只读 float 输入向量。 |
| `output` | 长度为 1 的 float 输出数组，保存 A 与 B 的点积。 |
| `N` | 两个输入向量的元素个数。 |
