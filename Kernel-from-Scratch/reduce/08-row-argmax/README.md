# Row-wise Argmax / 按行最大值索引

**LeetGPU：** Medium · [官方题目与参考](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/126_row_argmax)

**练习层级：** 基础

返回每行最大元素的列索引；出现相同最大值时，返回最小列索引。

$$
output_b=\min\{j:X_{bj}=\max_kX_{bk}\}
$$

## 接口

```cuda
void solve(const float* logits, int* output, int B, int V);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `logits` | float32 输入，形状 (B, V)。 |
| `output` | int32 输出，长度 B。 |
| `B` | 行数。 |
| `V` | 每行列数，例如词表大小。 |

## 示例

logits=[[1,3,3],[-2,-1,-3]]，输出 [1,1]。

## 约束与对照

1≤B≤1024，1≤V≤262144；输入有限。官方计时 B=256、V=128000。

**PyTorch 对照：** `logits.argmax(dim=1).to(torch.int32)`

**前置练习：** [reduce/05-row-reduction](../05-row-reduction/README.md)
