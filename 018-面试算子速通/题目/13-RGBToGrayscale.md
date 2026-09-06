# RGB to Grayscale（Element-wise）

**LeetGPU：** Easy · [RGB to Grayscale](https://leetgpu.com/challenges/rgb-to-grayscale)  
**目录：** `element_wise/09_rgb_to_grayscale`

输入是 `height × width × 3` 的 float，按像素交错 `R,G,B,R,G,B,…`。输出 `height × width` 灰度：

$$
\mathrm{gray}=0.299\,R+0.587\,G+0.114\,B
$$

系数必须用这三个数。例：`[255,0,0] → 76.245`，`[100,150,200] → 140.75`。

## 接口

```cuda
void solve(const float* input, float* output, int width, int height);
```

`1 ≤ width,height ≤ 4096`，`width×height ≤ 2^{22}`，RGB ∈ [0, 255]。计时 `2048×2048`。

## 为什么归 element-wise

每个输出只吃自己那三个通道，无邻域、无通信。和 color inversion 同一核：grid-stride + 向量化。差别：

| | color invert | 本题 |
|---|---|---|
| 类型 | `uchar` RGBA | `float` RGB |
| 向量 | `uchar4` | `float3` |
| 读写比 | 4→4 原地 | **3→1** |
| 公式 | `255-x` | 加权和 |

不是 permute（没有「只换下标」），也不是 reduce（没有多像素归约）。

## 写法

线程 `i` 对应像素 `i`，读 `float3`，写一个 `float`。相邻线程读相邻像素 → 整段 `3×32` 个 float 连续，读合并；写本来就是合并。算术强度低，带宽墙。
