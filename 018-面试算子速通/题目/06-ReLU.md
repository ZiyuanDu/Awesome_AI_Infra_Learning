# ReLU（Element-wise）

**LeetGPU：** Easy · [ReLU](https://leetgpu.com/challenges/relu)  
**目录：** `element_wise/03_relu`

$$
\mathrm{ReLU}(x)=\max(0,x)
$$

对 `input[N]` 逐点写到 `output`。例：`[-2,-1,0,1,2] → [0,0,0,1,2]`。

## 接口

```cuda
void solve(const float* input, float* output, int N);
```

`1 ≤ N ≤ 1e8`，计时 `N=2.5e7`。

## 为什么归 element-wise

每个输出只看同位置输入，没有邻域、没有通信。和 vector add 同一核：grid-stride + `float4`，公式换成 `fmaxf(x, 0)`。`fmaxf` 是谓词，正负混在一个 warp 里也不会分化。带宽墙（1 次比较 / 8 字节）。
