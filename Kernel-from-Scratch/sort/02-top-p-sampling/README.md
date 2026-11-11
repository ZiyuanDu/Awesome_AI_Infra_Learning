# Top-p Sampling / 确定性核采样

**LeetGPU：** Medium · [官方题目与参考](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/60_top_p_sampling)

**练习层级：** 专项

对 logits 做稳定 Softmax，按概率降序取最短的累计概率达到 p 的候选前缀，再使用官方指定随机映射采样。

$$
J=\min\{j:\sum_{r=0}^{j}p_{(r)}\ge p\},\qquad sample=\arg\max_{i\in nucleus}\frac{\tilde p_i}{-\log u_i}
$$

## 接口

```cuda
void solve(const float* logits, const float* p, const int* seed, int* sampled_token, int vocab_size);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `logits` | float32 分数，长度 vocab_size。 |
| `p` | GPU float32 指针，长度 1，保存阈值；不是按值传入的标量。 |
| `seed` | GPU int32 指针，长度 1；按其 32 位比特模式使用。 |
| `sampled_token` | GPU int32 输出，长度 1。 |
| `vocab_size` | 词表大小。 |

## 示例

logits=[10,0,-10]、p=0.5，候选集只有 token 0，因此任何 seed 都返回 0。

## 约束与对照

3≤vocab_size≤50000，logits∈[-100,100]，0<p≤1；稳定排序时等概率按原 token id 排序。

随机算法必须和官方 reference 一致，不能用任意 multinomial 代替。以下整数计算均按 32 位无符号数处理并回绕：
- s=(x*747796405+2891336453) mod 2^32。
- w=(((s >> ((s >> 28)+4)) XOR s)*277803737) mod 2^32。
- hash(x)=(w >> 22) XOR w。
- 对 token id i，bits=hash((hash(seed)+i) mod 2^32)，u_i=(bits+0.5)/2^32。
- 官方采样分数使用 float64 概率和 log 计算；返回分数最大的候选原 token id。分数相同时选择候选顺序中最先者。

**前置练习：** [sort/01-top-k-selection](../01-top-k-selection/README.md) → [reduce/02-softmax](../../reduce/02-softmax/README.md) → [reduce/03-prefix-sum](../../reduce/03-prefix-sum/README.md)
