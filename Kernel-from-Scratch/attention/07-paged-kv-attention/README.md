# Paged KV-Cache Attention / 分页解码

**LeetGPU：** Medium · [官方题目与参考](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/86_paged_attention)

**练习层级：** 专项

通过逻辑页到物理页的映射读取 KV 缓存，再计算每个序列、每个头的一条解码 query。

$$
page=block\_table[b,\lfloor t/S\rfloor],\quad offset=t\bmod S,\quad K_{bth}=K_{cache}[page,offset,h,:]
$$

## 接口

```cuda
void solve(
    const float* Q,
    const float* K_cache,
    const float* V_cache,
    const int* block_table,
    const int* context_lens,
    float* output,
    int batch_size,
    int num_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq
);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `Q` | float32 Query，形状 (batch_size, num_heads, head_dim)。 |
| `K_cache` | float32 物理池，形状 (num_physical_blocks, block_size, num_heads, head_dim)。 |
| `V_cache` | float32 物理池，形状与 K_cache 相同。 |
| `block_table` | int32 页表，形状 (batch_size, max_blocks_per_seq)。 |
| `context_lens` | int32 有效 token 数，长度 batch_size。 |
| `output` | float32 输出，形状与 Q 相同。 |
| `batch_size` | 序列数。 |
| `num_heads` | Query/KV 头数。 |
| `head_dim` | 头维度。 |
| `block_size` | 每页包含的 token 个数。 |
| `max_blocks_per_seq` | 页表每行的容量。 |

## 示例

一个序列有两页，页表 [1,0]、每页一个 token；应先读取物理页 1，再读取物理页 0。

## 约束与对照

1≤batch_size≤32，1≤num_heads≤64；head_dim≤256 且为 4 的倍数；block_size 为不大于 64 的 2 的幂；有效长度在 [1,8192]。

只访问有效 token 覆盖的页；页表尾部是 padding，池里未被引用的页也不得读取。前面的公式只定义寻址，后续按连续 Decode Attention 的公式归一化、加权 V。

**前置练习：** [attention/06-decode-attention](../06-decode-attention/README.md) → [indexing/01-gather](../../indexing/01-gather/README.md)
