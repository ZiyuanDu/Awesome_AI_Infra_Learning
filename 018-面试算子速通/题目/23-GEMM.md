# General Matrix Multiplication（GEMM）

**LeetGPU：** Medium · [GEMM](https://leetgpu.com/challenges/general-matrix-multiplication-gemm)  
**目录：** `gemm/03_gemm`

和 Easy「矩阵乘法」不是同一题：这里是 **BLAS 语义 + FP16 I/O + FP32 累加**。

$$
C \leftarrow \alpha\,(A B) + \beta\, C_{\mathrm{init}}
$$

| 矩阵 | 形状 | 类型 | 布局 |
|---|---|---|---|
| \(A\) | \(M\times K\) | half | row-major |
| \(B\) | \(K\times N\) | half | row-major |
| \(C\) | \(M\times N\) | half in/out | row-major |
| \(\alpha,\beta\) | 标量 | float | — |

例：\(A{=}[[1,2,3],[4,5,6]]\)，\(B{=}[[1,2],[3,4],[5,6]]\)，\(C_0{=}1\)，\(\alpha{=}1,\beta{=}0\) → \(C{=}[[22,28],[49,64]]\)。

## 接口（CUDA）

```cuda
void solve(const half* A, const half* B, half* C,
           int M, int N, int K, float alpha, float beta);
```

- 允许 **WMMA**，禁止 cuBLAS
- 乘加累加用 **FP32**，写回 **half**
- \(16\le M,N,K\le 4096\)；计时 **\(M{=}N{=}K{=}1024\)**

对照 Easy `01_matrix_mul`：那里是 \(A(M{\times}N)\,B(N{\times}K){\to}C(M{\times}K)\) 全 FP32、无 \(\alpha/\beta\)。

## 性能结论（RTX 5090 D v2 / sm_120）

| 实现 | \(1024^3\) | \(2048^3\) | 相对 cuBLAS |
|---|---|---|---|
| CUDA Core（half→float smem 外积） | ~16–20 TFLOPS | — | ~20% |
| **本仓库 WMMA + cp.async** | **~55–65 TFLOPS** | **~100 TFLOPS** | **~60–70%** |
| cuBLAS `COMPUTE_32F`（同精度语义） | ~95 TFLOPS | ~175 TFLOPS | 100% |
| cuBLAS 默认 TensorOp（可更激进） | ~170 TFLOPS | — | 上限参考 |

**够不够高效？** 对面试题够用：过题 + 能讲清为什么没吃满。距库还有约 30–40%，主要差在：更大 tile / 更好的 smem swizzle / warp-specialized 流水 / CUTLASS 级调度，而不是「有没有用 Tensor Core」。

### 关键坑

本地 `CMAKE_CUDA_ARCHITECTURES` 若落成 `75`，WMMA 会 JIT 在错误 ISA 上，实测可掉到 ~30 TFLOPS 且 `cp.async` 直接编不过。本机应固定 **`120`**。

### 和最新论文 / 教程的关系（口播）

- **数据中心 Blackwell（sm100）**：[Colfax CUTLASS TMEM 教程](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/)、[tcgen05 for dummies](https://gau-nernst.github.io/tcgen05/) 等走 **`tcgen05.mma` + Tensor Memory**，累加器在 TMEM，单线程发 MMA，靠 mbarrier 同步；手写可到接近 cuBLAS。
- **消费级 50 系（sm120）**：**没有 TMEM / 不指望 tcgen05 那套**；CUTLASS 在 Sm120 上仍是 `mma.sync` / 较小 tile。面试写 **WMMA + half smem + cp.async 双缓冲** 就是正确一代。

## 面试写法（仓库版）

`03_gemm` 主路径（`M,N,K % 16 == 0`）：

1. CTA 产出 \(64\times64\) 的 \(C\)；\(BK{=}32\)；**4 warps / 128 thr**（提高 1024³ 下每 SM CTA 数）
2. \(A,B\) tile 以 **half** 进 smem（不要先转 float）；\(B\) 行距 pad
3. **`cp.async` 双缓冲** 与 WMMA `mma_sync` 重叠
4. 每 warp 寄存器持有 \(2\times2\) 个 \(16\times16\) accum（FP32）
5. Epilogue：`alpha * acc + beta * C`，写回 half

非对齐尺寸走 CUDA Core fallback。

## 和 01 的关系

| | `01_matrix_mul` | `03_gemm` |
|---|---|---|
| 公式 | \(C=AB\) | \(C=\alpha AB+\beta C\) |
| 形状 | \(A_{M\times N},B_{N\times K}\) | \(A_{M\times K},B_{K\times N}\) |
| 类型 | float | half I/O，float 累加 |
| 计时 | \(8192{\times}6144{\times}4096\) | \(1024^3\) |
| 本机参考 | ~63 TFLOPS（CUDA Core） | ~60 TFLOPS（WMMA，更小问题） |
