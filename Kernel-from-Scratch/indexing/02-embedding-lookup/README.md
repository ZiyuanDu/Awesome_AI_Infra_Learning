# Embedding Lookup / 基础查表

**来源：** 面试扩展练习；形状、接口和边界行为由本题明确约定。

**练习层级：** 基础

每个 token id 读取 embedding 表的一整行；不融合位置编码和归一化。

$$
output_{i,d}=weight_{ids_i,d}
$$

## 接口

```cuda
void solve(const float* weight, const int* ids, float* output, int V, int D, int N);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `weight` | float32 只读表，形状 (V, D)。 |
| `ids` | int32 token id，长度 N。 |
| `output` | float32 输出，形状 (N, D)。 |
| `V` | 词表大小。 |
| `D` | 每个 embedding 向量的长度。 |
| `N` | token 个数。 |

## 示例

weight=[[1,2],[3,4],[5,6]]，ids=[2,0]，输出 [[5,6],[1,2]]。

## 约束与对照

V,D,N≥1，0≤ids[i]<V；允许重复 id。

**PyTorch 对照：** `weight[ids.long()]`

**前置练习：** [indexing/01-gather](../01-gather/README.md)
