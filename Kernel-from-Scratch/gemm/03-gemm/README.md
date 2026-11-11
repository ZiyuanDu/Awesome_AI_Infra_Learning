# GEMM（FP16，α / β）

**LeetGPU：** Medium · [GEMM](https://leetgpu.com/challenges/general-matrix-multiplication-gemm)  

给定 FP16 矩阵 A、B 和 C，以及标量 alpha、beta，使用 FP32 累加计算：

$$
C \leftarrow \alpha\,(A B) + \beta\, C_{\mathrm{init}}
$$

| 矩阵 | 形状 | 类型 | 布局 |
|---|---|---|---|
| \(A\) | \(M\times K\) | half | row-major |
| \(B\) | \(K\times N\) | half | row-major |
| \(C\) | \(M\times N\) | half in/out | row-major |
| \(\alpha,\beta\) | 标量 | float | — |

例：\(A{=}[[1,2,3],[4,5,6]]\)，\(B{=}[[1,2],[3,4],[5,6]]\)，\(C_0{=}1\)，\(\alpha{=}1,\beta{=}0\) → \(C{=}[[22,28],[49,64]]\)。

## 接口（CUDA）

```cuda
void solve(const half* A, const half* B, half* C,
           int M, int N, int K, float alpha, float beta);
```

所有指针参数均指向 **GPU 设备内存**；数组由调用方分配，尺寸参数和其他标量按值传入。

| 参数 | 含义 |
|---|---|
| `A` | 行优先的 M × K 只读 half 矩阵。 |
| `B` | 行优先的 K × N 只读 half 矩阵。 |
| `C` | 行优先的 M × N half 输入输出矩阵；初值参与 beta × C，最终被结果覆盖。 |
| `M` | A 和 C 的行数。 |
| `N` | B 和 C 的列数。 |
| `K` | A 的列数及 B 的行数，即归约维度。 |
| `alpha` | 矩阵乘积 A × B 的 float 缩放系数。 |
| `beta` | C 初值的 float 缩放系数。 |

- 允许 **WMMA**，禁止 cuBLAS
- 乘加累加用 **FP32**，写回 **half**
- \(16\le M,N,K\le 4096\)；计时 **\(M{=}N{=}K{=}1024\)**
