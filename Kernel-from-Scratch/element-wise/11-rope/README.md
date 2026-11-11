# RoPE / 旋转位置编码

**LeetGPU：** Medium · [官方题目与参考](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/61_rope_embedding)

**练习层级：** 基础

对每行特征的前后半区进行成对旋转。不同元素对独立计算，归入逐元素类；模型语义标签为位置编码。

$$
x_1=Q_{:,0:D/2},\quad x_2=Q_{:,D/2:D},\qquad Y=[x_1c-x_2s,\ x_2c+x_1s]
$$

## 接口

```cuda
void solve(float* Q, float* cos, float* sin, float* output, int M, int D);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `Q` | float32 输入，形状 (M, D)，只读。 |
| `cos` | float32 余弦表，形状 (M, D)；每行的前后半区相同。 |
| `sin` | float32 正弦表，形状 (M, D)；每行的前后半区相同。 |
| `output` | float32 输出，形状 (M, D)。 |
| `M` | 行数，即 token 数量。 |
| `D` | 每行特征数，必须为偶数。 |

## 示例

Q=[1,2,3,4]，cos 全为 0，sin 全为 1，输出 [-3,-4,1,2]。

## 约束与对照

D 必须为偶数。官方计时 M=1048576、D=128；计时 M 超过题面所列的 10000 上限。

**PyTorch 对照：** `先构造 rotate_half(Q)=concat(-Q[:, D/2:], Q[:, :D/2])，再计算 Q*cos + rotate_half(Q)*sin。`

本题使用前后半区配对。相邻元素交错配对是另一个布局变体，不能混用。
