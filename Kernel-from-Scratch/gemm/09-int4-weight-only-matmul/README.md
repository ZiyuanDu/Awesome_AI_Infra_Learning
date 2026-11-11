# W4A16 GEMM / INT4 权重量化矩阵乘

**LeetGPU：** Medium · [官方题目与参考](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/81_int4_matmul)

**练习层级：** 专项

输入激活与缩放系数为 FP16，权重每字节打包两个 4-bit 码，按组反量化后执行矩阵乘。

$$
W_{n,k}=(q_{n,k}-8)\,scales_{n,\lfloor k/group\_size\rfloor},\qquad y=\mathrm{FP16}(x\,W^\top)
$$

## 接口

```cuda
void solve(
    const __half* x,
    const uint8_t* w_q,
    const __half* scales,
    __half* y,
    int M,
    int N,
    int K,
    int group_size
);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `x` | FP16 输入，形状 (M, K)。 |
| `w_q` | uint8 权重码，形状 (N, K/2)；每字节高 4 位对应偶数 k，低 4 位对应奇数 k。 |
| `scales` | FP16 缩放表，形状 (N, K/group_size)。 |
| `y` | FP16 输出，形状 (M, N)。 |
| `M` | token 行数。 |
| `N` | 输出通道数，也是权重行数。 |
| `K` | 输入通道数，必须能被 2 和 group_size 整除。 |
| `group_size` | 沿每个权重行的量化组长度。 |

## 示例

M=N=1、K=2、group_size=2，x=[1,1]、w_q=[0x98]、scale=1；解码权重 [1,0]，输出 [1]。

## 约束与对照

1≤M,N,K≤8192；group_size∈{2,4,8,16,32,64,128}。这是减 8 的偏移编码，不是把 nibble 直接按二补码符号扩展。

**PyTorch 对照：** `解码到 FP32、乘对应 scale，计算 x.float() @ W.T，最后转 half。`

**前置练习：** [gemm/06-fp16-batched-matmul](../06-fp16-batched-matmul/README.md) → [element-wise/13-weight-dequantization](../../element-wise/13-weight-dequantization/README.md)
