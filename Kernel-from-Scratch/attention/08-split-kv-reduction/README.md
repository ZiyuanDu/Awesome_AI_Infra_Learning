# Split-KV Reduction / 合并注意力分片

**LeetGPU：** Medium · [官方题目与参考](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/128_split_kv_attention_reduction)

**练习层级：** 融合

合并各 KV 分片已经归一化的局部输出，权重来自各分片的 LogSumExp。

$$
m_h=\max_sL_{sh},\quad w_{sh}=\frac{\exp(L_{sh}-m_h)}{\sum_r\exp(L_{rh}-m_h)},\quad O_{hd}=\sum_sw_{sh}P_{shd}
$$

## 接口

```cuda
void solve(
    const float* partial_out,
    const float* partial_lse,
    float* output,
    int num_splits,
    int num_heads,
    int head_dim
);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `partial_out` | float32 各分片局部归一化输出，形状 (num_splits, num_heads, head_dim)。 |
| `partial_lse` | float32 局部分数的 LogSumExp，形状 (num_splits, num_heads)。 |
| `output` | float32 合并输出，形状 (num_heads, head_dim)。 |
| `num_splits` | KV 分片数。 |
| `num_heads` | 头数。 |
| `head_dim` | 头维度。 |

## 示例

一个头有两个分片，partial_out=[2,4]，partial_lse=[0,log(3)]，合并输出 3.5。

## 约束与对照

1≤num_splits≤256，1≤num_heads≤128，1≤head_dim≤256。本练习采用官方测试的有限 LSE 输入；空分片变体需另行约定 -inf 和全空行为。

**前置练习：** [attention/06-decode-attention](../06-decode-attention/README.md) → [reduce/09-logsumexp](../../reduce/09-logsumexp/README.md)
