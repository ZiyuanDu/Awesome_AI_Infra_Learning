# LogSumExp / 按行稳定对数求和

**来源：** 面试扩展练习；形状、接口和边界行为由本题明确约定。

**练习层级：** 基础

输出每行指数和的对数，避免直接 exp 大数溢出。

$$
m_i=\max_jX_{ij},\qquad L_i=m_i+\log\sum_j\exp(X_{ij}-m_i)
$$

## 接口

```cuda
void solve(const float* input, float* output, int M, int N);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `input` | float32 输入，形状 (M, N)。 |
| `output` | float32 输出，长度 M。 |
| `M` | 行数。 |
| `N` | 每行列数。 |

## 示例

input=[[0,0],[1,1]]，输出 [log(2),1+log(2)]。

## 约束与对照

M,N≥1；输入有限。

**PyTorch 对照：** `torch.logsumexp(input, dim=1)`

**前置练习：** [reduce/05-row-reduction](../05-row-reduction/README.md)
