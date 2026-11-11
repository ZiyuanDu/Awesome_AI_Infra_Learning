# Dropout / 给定随机数的前向

**来源：** 面试扩展练习；形状、接口和边界行为由本题明确约定。

**练习层级：** 专项

实现 inverted dropout。由调用方提供随机数，先把掩码和缩放的行为与随机数生成分开。

$$
m_i=[u_i\ge p],\qquad Y_i=m_i\,X_i/(1-p)
$$

## 接口

```cuda
void solve(const float* input, const float* uniform, float* output, unsigned char* mask, int N, float p);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `input` | float32 输入，长度 N。 |
| `uniform` | float32 随机数，长度 N，每个值位于 [0,1)。 |
| `output` | float32 输出，长度 N。 |
| `mask` | uint8 输出掩码，长度 N；保留为 1，丢弃为 0。 |
| `N` | 元素个数。 |
| `p` | 丢弃概率，满足 0≤p<1。 |

## 示例

input=[2,4]，uniform=[0.2,0.8]，p=0.5，输出 [0,8]，mask=[0,1]。

## 约束与对照

N≥1；0≤p<1；全部输出元素必须写入。p=0 时输出等于输入。

**PyTorch 对照：** `mask=(uniform>=p)；output=where(mask, input/(1-p), 0)。`

进阶变体：在 kernel 内使用明确指定的 seed/counter 随机算法；不同随机流不能直接逐元素对照。
