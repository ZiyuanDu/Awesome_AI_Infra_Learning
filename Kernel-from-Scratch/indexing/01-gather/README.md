# Gather / 一维索引读取

**来源：** 面试扩展练习；形状、接口和边界行为由本题明确约定。

**练习层级：** 基础

按整数索引读取输入，允许重复索引。

$$
output_i=input_{indices_i}
$$

## 接口

```cuda
void solve(const float* input, const int* indices, float* output, int N, int K);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `input` | float32 只读输入，长度 N。 |
| `indices` | int32 索引，长度 K。 |
| `output` | float32 输出，长度 K。 |
| `N` | 输入长度。 |
| `K` | 需要读取的元素个数。 |

## 示例

input=[10,20,30]，indices=[2,0,2]，输出 [30,10,30]。

## 约束与对照

N,K≥1；0≤indices[i]<N；输入输出不别名。

**PyTorch 对照：** `input[indices.long()]`
