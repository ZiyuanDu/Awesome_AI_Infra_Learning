# Token Embedding Layer / 查表与 LayerNorm 融合

**LeetGPU：** Medium · [官方题目与参考](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/106_token_embedding_layer)

**练习层级：** 融合

官方完整题：token embedding 与 position embedding 相加后，对最后一维做 LayerNorm。

$$
z_{btd}=E^{token}_{ids_{bt},d}+E^{pos}_{positions_t,d},\qquad Y=\mathrm{LayerNorm}_D(z;\gamma,\beta,\epsilon)
$$

## 接口

```cuda
void solve(
    const int* token_ids,
    const int* position_ids,
    const float* token_embeddings,
    const float* position_embeddings,
    const float* gamma,
    const float* beta,
    float* output,
    int B,
    int T,
    int V,
    int P,
    int D,
    float eps
);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `token_ids` | int32 输入，形状 (B, T)；值域 [0,V)。 |
| `position_ids` | int32 输入，长度 T；所有 batch 共用；值域 [0,P)。 |
| `token_embeddings` | float32 表，形状 (V, D)。 |
| `position_embeddings` | float32 表，形状 (P, D)。 |
| `gamma` | float32 LayerNorm 权重，长度 D。 |
| `beta` | float32 LayerNorm 偏置，长度 D。 |
| `output` | float32 输出，形状 (B, T, D)。 |
| `B` | batch 大小。 |
| `T` | 每个样本的 token 数。 |
| `V` | token 表的行数。 |
| `P` | position 表的行数。 |
| `D` | embedding 维度，也是归一化长度。 |
| `eps` | LayerNorm 稳定项。 |

## 示例

单个 token 的两张表相加得到 [1,3]，gamma=[1,1]、beta=[0,0]，输出约 [-1,1]。

## 约束与对照

1≤B≤64，1≤T≤1024，1≤V≤50000，1≤P≤4096，1≤D≤1024，eps=1e-5。

主分类按索引访存放在 indexing；同时带有 normalization 和融合标签。

**前置练习：** [indexing/02-embedding-lookup](../02-embedding-lookup/README.md) → [normalization/01-layer-norm](../../normalization/01-layer-norm/README.md)
