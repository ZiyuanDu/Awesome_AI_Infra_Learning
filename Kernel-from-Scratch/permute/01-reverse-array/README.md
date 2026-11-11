# 反转数组

将长度为 $N$ 的 32 位浮点数组在 GPU 上原地反转：

$$
\mathrm{input}_{\mathrm{after}}[i] = \mathrm{input}_{\mathrm{before}}[N - 1 - i], \qquad 0 \le i < N
$$

## 接口

```cuda
void solve(float* input, int N);
```

所有指针参数均指向 **GPU 设备内存**；数组由调用方分配，尺寸参数和其他标量按值传入。

| 参数 | 含义 |
|---|---|
| `input` | 长度为 N 的 float 数组；原地反转，结果写回同一块设备内存。 |
| `N` | 数组元素个数。 |

## 示例

```text
输入：input = [1, 2, 3, 4, 5], N = 5
输出：input = [5, 4, 3, 2, 1]
```

只有一个元素时结果不变；长度为奇数时中间元素保持原值。
结果必须写回传入的数组；接口不接收单独的输出指针。
