# Softmax Attention / Online

**LeetGPU：** Medium · [Softmax Attention](https://leetgpu.com/challenges/softmax-attention)

给定 Query 矩阵 Q、Key 矩阵 K 和 Value 矩阵 V，在 GPU 上计算：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d}}\right)V
$$

Softmax 按行计算，并使用减去行最大值的方式保证数值稳定性。结果写入 output。

## 公式与形状

| 矩阵 | 形状 |
|---|---|
| \(Q\) | \(M\times d\) |
| \(K,V\) | \(N\times d\) |
| 输出 \(O\) | \(M\times d\) |

\(1\le M,N\le 10^5\)，\(1\le d\le 128\)。计时 \(M=512,N=256\)。

对每个 query 行 \(i\)：

\[
S_{ij}=\frac{q_i\cdot k_j}{\sqrt{d}},\quad
P_{i:}=\mathrm{softmax}(S_{i:}),\quad
o_i=\sum_j P_{ij}\,v_j
\]

## 接口

```cuda
void solve(const float* Q, const float* K, const float* V, float* output,
           int M, int N, int d);
```

所有指针参数均指向 **GPU 设备内存**；数组由调用方分配，尺寸参数和其他标量按值传入。

| 参数 | 含义 |
|---|---|
| `Q` | 行优先的 M × d 只读 float Query 矩阵。 |
| `K` | 行优先的 N × d 只读 float Key 矩阵。 |
| `V` | 行优先的 N × d 只读 float Value 矩阵。 |
| `output` | 行优先的 M × d float 输出矩阵，写入 softmax(Q × Kᵀ / sqrt(d)) × V，按行做 softmax。 |
| `M` | Query 序列长度及输出行数。 |
| `N` | Key / Value 序列长度。 |
| `d` | Q、K、V 及输出的特征维度。 |
