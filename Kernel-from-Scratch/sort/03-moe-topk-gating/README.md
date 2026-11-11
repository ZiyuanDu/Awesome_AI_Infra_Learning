# MoE Top-K Gating / 专家选择与权重

**LeetGPU：** Medium · [官方题目与参考](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/67_moe_topk_gating)

**练习层级：** 融合

每个 token 选出分数最高的 k 个专家，再只对这 k 个分数做 Softmax。

$$
(v,I)=\mathrm{TopK}_{desc}(logits_{i:},k),\qquad weights_{i:}=\mathrm{softmax}(v)
$$

## 接口

```cuda
void solve(const float* logits, float* topk_weights, int* topk_indices, int M, int E, int k);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `logits` | float32 输入，形状 (M, E)。 |
| `topk_weights` | float32 输出，形状 (M, k)，每行和为 1。 |
| `topk_indices` | int32 输出，形状 (M, k)；与权重位置一一对应。 |
| `M` | token 数。 |
| `E` | 专家总数。 |
| `k` | 每个 token 选择的专家数。 |

## 示例

logits=[[1,2,3,4]]、k=2，indices=[[3,2]]，weights≈[[0.731059,0.268941]]。

## 约束与对照

1≤M≤10000，1≤E≤256，1≤k≤E；专家按 logit 降序输出。相同值的索引顺序按官方 torch.topk 参考处理，不自行假定 stable sort 语义。

**前置练习：** [sort/01-top-k-selection](../01-top-k-selection/README.md) → [reduce/06-row-softmax](../../reduce/06-row-softmax/README.md)
