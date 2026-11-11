# Decode Attention / 连续 KV Cache 解码

**来源：** 面试扩展练习；形状、接口和边界行为由本题明确约定。

**练习层级：** 专项

每个序列每个头只有一个 query，访问已经缓存的有效历史。先练连续缓存，再练分页。

$$
s_{bth}=Q_{bh}\cdot K_{bth}/\sqrt D,\quad O_{bh}=\sum_{t<L_b}\mathrm{softmax}_{t<L_b}(s_{bth})V_{bth}
$$

## 接口

```cuda
void solve(
    const float* Q,
    const float* K_cache,
    const float* V_cache,
    const int* context_lens,
    float* output,
    int B,
    int H,
    int T,
    int D
);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `Q` | float32 Query，形状 (B, H, D)。 |
| `K_cache` | float32 缓存，形状 (B, T, H, D)。 |
| `V_cache` | float32 缓存，形状同 K_cache。 |
| `context_lens` | int32 有效历史长度，长度 B。 |
| `output` | float32 输出，形状 (B, H, D)。 |
| `B` | 序列数。 |
| `H` | Query 和 KV 共同的头数。 |
| `T` | 每条缓存的最大容量。 |
| `D` | 头维度。 |

## 示例

B=H=D=1、T=2，Q=K=0、V=[2,4]、有效长度为 2，输出 3。

## 约束与对照

B,H,T,D≥1；1≤context_lens[b]≤T；不读取超出有效历史的位置；不更新缓存。

本题 Q 的序列长度为 1，工作划分与训练/prefill 的整块 QK 不同。

**前置练习：** [attention/02-online](../02-online/README.md)
