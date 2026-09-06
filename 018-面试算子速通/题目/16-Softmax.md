# Softmax（Reduce / Online）

**LeetGPU：** Medium · [Softmax](https://leetgpu.com/challenges/softmax)  
**目录：** `reduce/02_softmax`  
**论文：** Milakov & Gimelshein, *Online normalizer calculation for softmax*, 2018

一维向量（不是按行矩阵）：

$$
\sigma(x)_i=\frac{e^{x_i-m}}{\ell},\quad m=\max_j x_j,\quad \ell=\sum_j e^{x_j-m}
$$

例：`[1,2,3] → ≈[0.090, 0.244, 0.665]`。站点要求用 **max trick** 防溢出。

## 接口

```cuda
void solve(const float* input, float* output, int N);
```

`1 ≤ N ≤ 5e5`，计时 `N=5e5`。

## 为什么归 reduce

要看整段做 max / sum，线程必须通信。element-wise 写不出归一化。

## 3-pass vs Online（面试要能说清）

| | 3-pass | Online |
|---|---|---|
| 扫 x | max → sum(exp) → 写 = **3 读** | 攒 `(m,ℓ)` → 写 = **2 读** |
| 状态 | 先全局 `m` 再 `ℓ` | 流式更新，可分块 |
| 分块 | softmax 本身不结合 | `(m,ℓ)` 用 `onlineMerge` 结合 → FA |

递推（每个元素）：

```text
m' = max(m, x)
ℓ' = ℓ · exp(m - m') + exp(x - m')
```

两段状态：

```text
m' = max(m1, m2)
ℓ' = ℓ1·exp(m1-m') + ℓ2·exp(m2-m')
```

`common.cuh`：`onlineUpdate` / `onlineMerge` / `blockOnlineReduce`。

## 本题写法（N 大、smem 装不下整向量）

1. **K1** 每 block grid-stride + `float4`，online 出 `(m_b, ℓ_b)`
2. **K2** 一个 block 把所有 block 状态 `onlineMerge` 成全局 `(m, ℓ)`
3. **K3** 再读 `x` 写 `y`；**逆序**扫，蹭 K1 留在 L2 里的线（CACHE_OPT）

价值：少一次全局读；代数上给 FlashAttention 铺路。纯速度仍是带宽墙。

## 和 019 / 012 的关系

- `019/02`：按行 3-pass；`019/11`：按行 online（一行一 warp）
- `012`：warp / smem-cache / uncached 三档自动切
- 本目录：LeetGPU **一维** + 面试默写长度的 multi-block online
