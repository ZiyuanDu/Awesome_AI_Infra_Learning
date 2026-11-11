# SpMV

**LeetGPU：** Medium · [SpMV](https://leetgpu.com/challenges/sparse-matrix-vector-multiplication)  

\(y=Ax\)。站点给的 \(A\) 是 **row-major 稠密数组**（约 60–70% 为零），**不是 CSR**。计时 \(M{=}1000,\ N{=}10^4\)。

## 接口

```cuda
void solve(const float* A, const float* x, float* y, int M, int N, int nnz);
```

所有指针参数均指向 **GPU 设备内存**；数组由调用方分配，尺寸参数和其他标量按值传入。

| 参数 | 含义 |
|---|---|
| `A` | 行优先的 M × N 只读 float 矩阵；虽然含有大量零，仍以稠密数组存储。 |
| `x` | 长度为 N 的只读 float 输入向量。 |
| `y` | 长度为 M 的 float 输出向量，写入 A × x。 |
| `M` | 矩阵行数及输出向量长度。 |
| `N` | 矩阵列数及输入向量长度。 |
| `nnz` | 矩阵中非零元素个数，int 标量；接口保留此参数，不采用 CSR 索引。 |
