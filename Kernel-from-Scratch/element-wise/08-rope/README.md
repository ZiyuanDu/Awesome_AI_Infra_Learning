# RoPE / 旋转位置编码

**LeetGPU：** Medium · [官方题目与参考](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/61_rope_embedding)

**练习层级：** 基础

给定查询矩阵 `Q` 和预先计算好的余弦、正弦矩阵 `cos`、`sin`，计算旋转位置编码，将结果写入 `output`。所有矩阵均为 `float32`，形状为 `(M, D)`。

对于每一行 $r$（$0\le r<M$），输出定义为：

$$
\mathrm{output}_{r,j}=
\begin{cases}
Q_{r,j}\cdot\mathrm{cos}_{r,j}
-Q_{r,\,j+D/2}\cdot\mathrm{sin}_{r,j},
& 0\le j<D/2,\\[6pt]
Q_{r,j}\cdot\mathrm{cos}_{r,j}
+Q_{r,\,j-D/2}\cdot\mathrm{sin}_{r,j},
& D/2\le j<D.
\end{cases}
$$

## 示例

```text
输入：M = 1, D = 4
      Q   = [[1, 2, 3, 4]]
      cos = [[0, 0, 0, 0]]
      sin = [[1, 1, 1, 1]]
输出：output = [[-3, -4, 1, 2]]
```

## 约束

- `M >= 1`，`D >= 2`，且 `D` 为偶数。
- 对于 `cos` 和 `sin`，每行的后 `D/2` 个元素与前 `D/2` 个元素相同。
- 所有矩阵由调用方分配在 GPU 上，按行优先连续存储。
- `Q`、`cos`、`sin` 只读，`output` 与输入使用独立内存。

## 接口

```cuda
void solve(float* Q, float* cos, float* sin, float* output, int M, int D);
```
