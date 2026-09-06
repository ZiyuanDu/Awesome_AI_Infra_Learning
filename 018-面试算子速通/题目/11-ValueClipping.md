# Value Clipping（Element-wise）

**LeetGPU：** Easy · [Value Clipping](https://leetgpu.com/challenges/value-clipping)  
**目录：** `element_wise/08_value_clipping`

$$
\mathrm{clip}(x)=\min(\max(x,\mathrm{lo}),\mathrm{hi})
$$

对 `input[N]` 逐点钳到 `[lo, hi]`。例：`[1.5,-2,3,4.5], lo=0, hi=3.5 → [1.5,0,3,3.5]`。

常用于激活稳定、量化前饱和。

## 接口

```cuda
void solve(const float* input, float* output, int N, float lo, float hi);
```

`1 ≤ N ≤ 1e5`，`lo ≤ hi`，计时 `N=1e5`。

## 写法

和 ReLU 同一核：grid-stride + `float4`，公式换成 `fminf(fmaxf(x, lo), hi)`。谓词无分支，正负混在一个 warp 也不分化。带宽墙（两次比较 / 8 字节）。
