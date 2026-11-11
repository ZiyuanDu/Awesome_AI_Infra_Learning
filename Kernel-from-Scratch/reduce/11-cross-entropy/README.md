# Categorical Cross Entropy / 分类交叉熵

**LeetGPU：** Medium · [官方题目与参考](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/25_categorical_cross_entropy_loss)

**练习层级：** 基础

输入为未归一化 logits 和整数类别标签，输出 batch 平均损失。

$$
loss=\frac1N\sum_i\left(\mathrm{LogSumExp}(logits_{i:})-logits_{i,label_i}\right)
$$

## 接口

```cuda
void solve(const float* logits, const int* true_labels, float* loss, int N, int C);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `logits` | float32 输入，形状 (N, C)；未归一化分数。 |
| `true_labels` | int32 标签，长度 N，每个值满足 0≤label<C。 |
| `loss` | float32 输出，长度 1，保存平均损失。 |
| `N` | 样本数。 |
| `C` | 类别数。 |

## 示例

两个样本均为 logits=[0,0]，标签分别为 0 和 1，loss=log(2)。

## 约束与对照

1≤N≤10000，2≤C≤1000，logits∈[-10,10]。标签上界必须严格小于 C。

**PyTorch 对照：** `torch.nn.functional.cross_entropy(logits, true_labels.long(), reduction='mean')`

**前置练习：** [reduce/09-logsumexp](../09-logsumexp/README.md)
