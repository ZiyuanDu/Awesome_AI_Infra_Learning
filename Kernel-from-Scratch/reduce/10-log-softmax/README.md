# LogSoftmax / 按行对数概率

**来源：** 面试扩展练习；形状、接口和边界行为由本题明确约定。

**练习层级：** 基础

直接计算稳定的对数概率，避免先 Softmax 再 log 的下溢。

$$
Y_{ij}=X_{ij}-\mathrm{LogSumExp}(X_{i:})
$$

## 接口

```cuda
void solve(const float* input, float* output, int M, int N);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `input` | float32 输入，形状 (M, N)。 |
| `output` | float32 输出，形状 (M, N)。 |
| `M` | 行数。 |
| `N` | 每行类别数。 |

## 示例

input=[[0,0]]，输出 [[-log(2),-log(2)]]。

## 约束与对照

M,N≥1；输入有限。

**PyTorch 对照：** `torch.log_softmax(input, dim=1)`

**前置练习：** [reduce/09-logsumexp](../09-logsumexp/README.md)
