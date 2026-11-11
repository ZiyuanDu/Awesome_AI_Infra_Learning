# INT8 Quantized MatMul / 量化矩阵乘

**LeetGPU：** Medium · [官方题目与参考](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/32_int8_quantized_matmul)

**练习层级：** 专项

对带标量零点的 int8 输入相乘，再按输出量化尺度舍入并饱和到 int8。

$$
S_{ij}=\sum_k(A_{ik}-z_A)(B_{kj}-z_B),\qquad C_{ij}=\mathrm{clip}_{[-128,127]}(\mathrm{round}(S_{ij}s_As_B/s_C)+z_C)
$$

## 接口

```cuda
void solve(
    const int8_t* A,
    const int8_t* B,
    int8_t* C,
    int M,
    int N,
    int K,
    float scale_A,
    float scale_B,
    float scale_C,
    int zero_point_A,
    int zero_point_B,
    int zero_point_C
);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `A` | int8 输入，形状 (M, K)。 |
| `B` | int8 输入，形状 (K, N)。 |
| `C` | int8 输出，形状 (M, N)。 |
| `M` | 输出行数。 |
| `N` | 输出列数。 |
| `K` | 点积长度。 |
| `scale_A` | 输入 A 的正量化尺度。 |
| `scale_B` | 输入 B 的正量化尺度。 |
| `scale_C` | 输出 C 的正量化尺度。 |
| `zero_point_A` | A 的整数零点，位于 [-128,127]。 |
| `zero_point_B` | B 的整数零点，位于 [-128,127]。 |
| `zero_point_C` | C 的整数零点，位于 [-128,127]。 |

## 示例

A=[[2]]、B=[[3]]，尺度均为 1、零点均为 0，输出 [[6]]。

## 约束与对照

题面 M,N,K≤4096；官方计时却使用 M=8192、N=4096、K=2048。舍入按 PyTorch 的 ties-to-even；最终饱和到 int8。

官方 reference 对去零点后的 float32 矩阵乘结果先 round，再缩放、round 和饱和；大点积时需关注该参考的精度与容差。

**前置练习：** [gemm/01-matrix-mul](../01-matrix-mul/README.md)
