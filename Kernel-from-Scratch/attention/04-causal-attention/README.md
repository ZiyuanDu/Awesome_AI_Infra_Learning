# Causal Self-Attention / 因果自注意力

**LeetGPU：** Hard · [官方题目与参考](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/hard/53_casual_attention)

**练习层级：** 专项

只允许第 i 个 query 访问不晚于 i 的 key/value。该题为单头、无 batch 的自注意力。

$$
S=QK^\top/\sqrt d,\quad S_{ij}=-\infty\ (j>i),\quad output=\mathrm{softmax}_{row}(S)V
$$

## 接口

```cuda
void solve(const float* Q, const float* K, const float* V, float* output, int M, int d);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `Q` | float32 Query，形状 (M, d)。 |
| `K` | float32 Key，形状 (M, d)。 |
| `V` | float32 Value，形状 (M, d)。 |
| `output` | float32 输出，形状 (M, d)。 |
| `M` | 序列长度。 |
| `d` | 每个 token 的特征数。 |

## 示例

M=2、d=1，Q=K=0、V=[[2],[4]]，输出 [[2],[3]]。

## 约束与对照

1≤M≤10000，1≤d≤128；包含对角线位置；Softmax 按行稳定计算。

**前置练习：** [attention/02-online](../02-online/README.md) → [reduce/07-masked-softmax](../../reduce/07-masked-softmax/README.md)
