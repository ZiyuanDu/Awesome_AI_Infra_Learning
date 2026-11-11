# Histogram / 直方图

**LeetGPU：** Medium · [官方题目与参考](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/13_histogramming)

**练习层级：** 基础

按输入值分桶计数，属于多目标归约；原子操作是实现手段之一。

$$
histogram_b=\sum_i[input_i=b]
$$

## 接口

```cuda
void solve(const int* input, int* histogram, int N, int num_bins);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `input` | int32 输入，长度 N。 |
| `histogram` | int32 输出，长度 num_bins；初始值未定义，调用后必须是本轮计数。 |
| `N` | 输入元素个数。 |
| `num_bins` | 桶数。 |

## 示例

input=[0,1,2,1,0]，num_bins=3，输出 [2,2,1]。

## 约束与对照

1≤N≤1e8，1≤num_bins≤1024，0≤input[i]<num_bins。官方计时 N=50000000、num_bins=256。

**PyTorch 对照：** `torch.bincount(input.long(), minlength=num_bins).to(torch.int32)`

每次 solve 必须重新建立计数；不能把上次调用的结果继续累加。
