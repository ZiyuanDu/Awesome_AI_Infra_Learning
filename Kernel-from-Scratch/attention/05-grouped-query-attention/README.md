# GQA / 分组查询注意力

**LeetGPU：** Medium · [官方题目与参考](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/80_grouped_query_attention)

**练习层级：** 专项

多个 Query 头共享一个 KV 头。本题计算完整序列注意力，不带因果掩码，也没有 batch 维。

$$
g=H_q/H_{kv},\quad kv(h)=\lfloor h/g\rfloor,\quad O_h=\mathrm{softmax}(Q_hK_{kv(h)}^\top/\sqrt D)V_{kv(h)}
$$

## 接口

```cuda
void solve(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim
);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `Q` | float32 Query，形状 (num_q_heads, seq_len, head_dim)。 |
| `K` | float32 Key，形状 (num_kv_heads, seq_len, head_dim)。 |
| `V` | float32 Value，形状与 K 相同。 |
| `output` | float32 输出，形状与 Q 相同。 |
| `num_q_heads` | Query 头数。 |
| `num_kv_heads` | KV 头数，必须整除 Query 头数。 |
| `seq_len` | 每个头的序列长度。 |
| `head_dim` | 每个头的特征维度。 |

## 示例

Hq=2、Hkv=1、L=1 时，两个输出头都等于唯一 KV 头的 V。

## 约束与对照

1≤Hkv≤Hq≤64，Hq%Hkv=0；1≤L≤4096；8≤D≤256 且 D 为 8 的倍数。

共享关系是连续 Query 头分组；无需显式复制 K/V 成 Hq 个头。

**前置练习：** [attention/01-naive](../01-naive/README.md)
