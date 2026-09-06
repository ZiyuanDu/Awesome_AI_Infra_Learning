# Softmax Attention：是不是 FlashAttention？

**LeetGPU：** Medium · [Softmax Attention](https://leetgpu.com/challenges/softmax-attention)

## 一句话

**不是。** 站点只要求精确算

\[
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d}}\right)V
\]

（按行 softmax）。**没有**要求 tile、不物化 \(S\)、或 FA 论文里的 IO 最优。  
但约束里 \(N\) 最大 \(10^5\)，若把整行 \(S\in\mathbb{R}^{N}\) 塞进 smem 会炸；计时却是 **\(M=512,N=256\)**——短序列，朴素核也能过。  
面试正确姿势：**先写清 SDPA → 再说为什么要 online → 再口播 FA1/FA2 差在哪。**

## 公式与形状

| 矩阵 | 形状 |
|---|---|
| \(Q\) | \(M\times d\) |
| \(K,V\) | \(N\times d\) |
| 输出 \(O\) | \(M\times d\) |

\(1\le M,N\le 10^5\)，\(1\le d\le 128\)。计时 \(M=512,N=256\)。

对每个 query 行 \(i\)：

\[
S_{ij}=\frac{q_i\cdot k_j}{\sqrt{d}},\quad
P_{i:}=\mathrm{softmax}(S_{i:}),\quad
o_i=\sum_j P_{ij}\,v_j
\]

## 和 FlashAttention 的关系

| | Softmax Attention（本题） | FlashAttention |
|---|---|---|
| 数学 | 精确 SDPA | **同一公式** |
| 接口要求 | 结果对就行 | 不考 |
| \(S=QK^{\top}\) | 可以物化 / 三次扫 | **不写 HBM** |
| Softmax | 3-pass 或 online | **online `(m,ℓ)` + 熔进 \(PV\)** |
| 目的 | 算对 | 降 IO：读 KV 从 \(O(MND)\) 量级到流式一遍 |

FA 换的是 **并行与 IO**，不是换数学（见 019 README）。本题用 FA 合法且更稳（大 \(N\)），但不是题目名字的含义。

## 三个默写档（本目录）

| 目录 | 写法 | 何时讲 |
|---|---|---|
| `01_naive` | 每 query 一线程，三遍扫 \(K\)：max → \(\sum e^{s-m}\) → \(\sum p v\) | 白板第一版；\(S\) 不落地 |
| `02_online` | 同一并行，**一遍**扫 \(K\)，维护 `(m,ℓ,Oacc)` | 接上题 softmax online；FA 代数核 |
| `03_flash_v2` | Q tile × KV tile，片上 `onlineMerge`，最后除一次 \(ℓ\) | 口播 FA2 三条；大 \(M,N\) 才看出 IO |

推荐提交 / 默写终点：**`02_online`**（短、对、大 \(N\) 也活）。`03` 用来讲「和真 FA 还差 occupancy / Tensor Core」。

## 面试口播提纲

1. **SDPA 三步**：\(S=QK^{\top}/\sqrt{d}\) → 行 softmax → \(PV\)。
2. **Max trick**：先减行 max，防 `exp` 溢出（本题要求）。
3. **为什么 online**：3-pass 读三遍 \(K\)；online 把 max 与 sum 合成 `(m,ℓ)`，再把 \(P V\) 熔进同一遍 → 代数上等价，且 **分块可结合**（`onlineMerge`）。
4. **FA1 → FA2**：外 KV 内 Q 且 `(m,ℓ,O)` 回 HBM → 改成 **切 Q 并行**、`Oacc` 留片上、最后除一次。
5. **本题计时很小**：FA2 不一定最快；大 \(N\) 才是 FA 的主场。

## 接口

```cuda
void solve(const float* Q, const float* K, const float* V, float* output,
           int M, int N, int d);
```
