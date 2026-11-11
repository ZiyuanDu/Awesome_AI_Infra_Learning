# 矩阵转置

**难度：** Easy

将 32 位浮点矩阵 $A$（$rows \times cols$）转置为 $A^{\top}$（$cols \times rows$）。全部 **行优先**：

$$
\text{out}[c, r] = \text{in}[r, c]
$$

下标：$\text{in}[r\cdot cols + c]$，$\text{out}[c\cdot rows + r]$。

## 接口

```cuda
void solve(const float* input, float* output, int rows, int cols);
```

所有指针参数均指向 **GPU 设备内存**；数组由调用方分配，尺寸参数和其他标量按值传入。

| 参数 | 含义 |
|---|---|
| `input` | 行优先的 rows × cols 只读 float 输入矩阵。 |
| `output` | 行优先的 cols × rows float 输出矩阵，output[c × rows + r] = input[r × cols + c]。 |
| `rows` | 输入矩阵行数。 |
| `cols` | 输入矩阵列数。 |

- `input`、`output` 均为 **device 指针**
- 不得修改 `solve` 的签名
- 结果必须写在 `output` 中
- 不允许使用外部库

## 示例 1

**输入：** $2 \times 3$

```
[[1, 2, 3],
 [4, 5, 6]]
```

**输出：** $3 \times 2$

```
[[1, 4],
 [2, 5],
 [3, 6]]
```

## 示例 2

**输入：** $3 \times 1$

```
[[1],
 [2],
 [3]]
```

**输出：** $1 \times 3$

```
[[1, 2, 3]]
```

## 约束

- $1 \le rows, cols \le 8192$
- 计时规模：$rows = 7000,\ cols = 6000$
- 评测卡：NVIDIA A100-80GB
