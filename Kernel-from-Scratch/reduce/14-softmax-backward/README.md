# Softmax Backward / 按行反向

**来源：** 面试扩展练习；形状、接口和边界行为由本题明确约定。

**练习层级：** 专项

给定前向概率和上游梯度，计算输入梯度。

$$
dX_{ij}=Y_{ij}\left(dY_{ij}-\sum_kdY_{ik}Y_{ik}\right)
$$

## 接口

```cuda
void solve(const float* y, const float* dy, float* dx, int M, int N);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `y` | float32 前向 Softmax 输出，形状 (M, N)。 |
| `dy` | float32 上游梯度，形状 (M, N)。 |
| `dx` | float32 输出梯度，形状 (M, N)。 |
| `M` | 行数。 |
| `N` | 每行列数。 |

## 示例

y=[[0.5,0.5]]，dy=[[1,0]]，dx=[[0.25,-0.25]]。

## 约束与对照

M,N≥1；每行 y 非负且和为 1；输入输出不别名。

**PyTorch 对照：** `y * (dy - (dy*y).sum(dim=1, keepdim=True))`

**前置练习：** [reduce/06-row-softmax](../06-row-softmax/README.md)
