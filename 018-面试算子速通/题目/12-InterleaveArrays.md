# Interleave Arrays（Permute）

**LeetGPU：** Easy · [Interleave Arrays](https://leetgpu.com/challenges/interleave-arrays)  
**目录：** `permute/03_interleave_arrays`

两路 SoA 收成一路 AoS，数值不变，只换下标：

$$
\mathrm{out}[2i]=A[i],\quad \mathrm{out}[2i+1]=B[i]
$$

例：`A=[1,2,3], B=[4,5,6] → [1,4,2,5,3,6]`。

## 接口

```cuda
void solve(const float* A, const float* B, float* output, int N);
```

`1 ≤ N ≤ 5e7`，计时 `N=2.5e7`。输出长度 `2N`。

## 为什么是 permute，不是 reduce / element-wise

| 类别 | 这题像不像 | 原因 |
|---|---|---|
| **permute** | **是** | 值原样搬，写地址 `i → 2i / 2i+1` |
| element-wise | 接近但不是 | 逐点有计算，且 `out[i]` 对应同下标；这里写侧步长变了 |
| reduce | **否** | 没有多→少，没有 shuffle / smem 树 / atomic |

面试别往 reduce 靠：加一层归约会多同步、多写，带宽只会更差。

## 写法

读侧 `A`、`B` 都是顺序扫 → 合并。朴素两次标量写 `out[2i]`、`out[2i+1]` 也合并（相邻线程写相邻 `float2` 槽），但指令多。

默写档：每个线程吃一对 `float2`，拼一个 `float4` 写出去：

```text
(A[2i], A[2i+1]) + (B[2i], B[2i+1]) → (A[2i], B[2i], A[2i+1], B[2i+1])
```

读写都 8B/16B 向量化，grid-stride；`N` 奇数补最后一对 `float2`。带宽墙（搬运 `4N` 个 float，零算术）。
