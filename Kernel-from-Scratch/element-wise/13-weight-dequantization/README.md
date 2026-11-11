# Weight Dequantization / 分块反量化

**LeetGPU：** Medium · [官方题目与参考](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/64_weight_dequantization)

**练习层级：** 专项

每个矩阵元素乘以所属二维 tile 的缩放系数。此题输入 X 本身也是 float32，不涉及 INT4 解包。

$$
Y_{ij}=X_{ij}\,S_{\lfloor i/T\rfloor,\lfloor j/T\rfloor}
$$

## 接口

```cuda
void solve(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `X` | float32 输入矩阵，形状 (M, N)。 |
| `S` | float32 缩放表，形状 (ceil(M/TILE_SIZE), ceil(N/TILE_SIZE))。 |
| `Y` | float32 输出矩阵，形状 (M, N)。 |
| `M` | 矩阵行数。 |
| `N` | 矩阵列数。 |
| `TILE_SIZE` | 二维方形 tile 的边长。 |

## 示例

X 为 2×2 的全 2 矩阵，TILE_SIZE=16，S=[[0.5]]，输出为全 1 矩阵。

## 约束与对照

1≤M,N≤8192；TILE_SIZE 属于 {16,32,64,128}。官方计时 M=N=8192、TILE_SIZE=128。

**PyTorch 对照：** `将 S 的行列分别 repeat_interleave(TILE_SIZE)，截取到 (M,N)，再乘 X。`
