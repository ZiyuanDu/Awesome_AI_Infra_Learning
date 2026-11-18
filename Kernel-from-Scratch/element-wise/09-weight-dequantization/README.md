# Weight Dequantization / 分块反量化

**LeetGPU：** Medium · [在线题目](https://leetgpu.com/challenges/weight-dequantization) · [官方参考](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/64_weight_dequantization)

**练习层级：** 专项

反量化就是将存储的权重值乘以缩放系数，例如 `12 × 0.1 = 1.2`。本题将输入矩阵 `X` 按 `T × T` 划块，每个块共用一个缩放系数，`T = TILE_SIZE`。

## 示例

取 `T=2`。`X` 中的每个 `2 × 2` 块都乘以 `S` 中同一位置的系数，得到 `Y` 中对应的块。

![分块反量化示意图。X 的四个块分别乘以 S 中的 0.1、0.2、0.3 和 0.4，得到 Y 中的 1、4、9 和 16。](./block-dequantization.svg)

浅蓝色标出右下块的计算。块内每个 `40` 都乘以 `S[1, 1] = 0.4`，得到 `16`。

## 公式

行列下标从 0 开始，`//` 表示整数除法：

```text
块行号 = i // T
块列号 = j // T
Y[i, j] = X[i, j] × S[块行号, 块列号]
```

上例中，`X[2, 3]` 属于块 `(2 // 2, 3 // 2) = (1, 1)`，因此 `Y[2, 3] = 40 × 0.4 = 16`。

## 接口

```cuda
void solve(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE);
```

矩阵均为 `float32`，在 GPU 上按行优先连续存储，由调用方分配。`X`、`S` 只读，`Y` 为独立的输出矩阵。

| 参数 | 形状 | 含义 |
|---|---|---|
| `X` | `(M, N)` | 输入权重，`M` 行、`N` 列。 |
| `S` | `(ceil(M/T), ceil(N/T))` | 每个块的缩放系数。 |
| `Y` | `(M, N)` | 反量化结果。 |

## 测试约束

- `1 <= M, N <= 8192`，`TILE_SIZE ∈ {16, 32, 64, 128}`。
- `M`、`N` 可以不是 `TILE_SIZE` 的倍数，边缘块使用对应的缩放系数。
- 官方计时：`M=N=8192`，`TILE_SIZE=128`。

