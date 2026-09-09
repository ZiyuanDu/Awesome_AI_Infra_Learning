# 面试算子速通

按 **并行写法** 分目录；每题只有 `solve.cu` + `bench.cu`。  
对照：[LeetGPU](https://leetgpu.com/challenges) · 归类：[题目/00-分类.md](题目/00-分类.md)

```
include/          common.cuh（reduce/scan/online/load4）+ bench.cuh
01_element_wise/   01–10 逐点 map
02_reduce/         reduction · softmax · prefix_sum · dot
03_permute/        reverse · transpose · interleave
04_gemm/           matmul · spmv · gemm(fp16 αβ)
05_stencil/        conv1d/2d/3d
06_attention/      naive · online · flash_v2
题目/ · 科普/
```

阅读 = 目录数字顺序：map → reduce（shuffle/scan/online 地基）→ permute（首个 tile+smem）→ GEMM（经典 tile）→ stencil（halo 变体）→ attention（softmax + GEMM 合流）。

| 目录 | 模板 |
|---|---|
| `01_element_wise` | grid-stride，可 `float4` |
| `02_reduce` | shuffle / atomic / online `(m,ℓ)` / scan |
| `03_permute` | 下标置换；转置要 smem |
| `04_gemm` | tile + smem 外积；SpMV=按行 GEMV |
| `05_stencil` | 输出 tile + 输入 halo |
| `06_attention` | SDPA → online → FA2 |

```bash
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=120 && cmake --build build -j16
./build/02_reduce/01_reduction/bench       # 路径 = 分类/题号
```

> 本机 5090 用 **sm_120**。若 cache 里曾是 75，`04_gemm/03_gemm` 的 `cp.async` 会报 `requires .target sm_80 or higher`——删 build 或显式传 `-DCMAKE_CUDA_ARCHITECTURES=120`。

新题：放进对应分类，根 `CMakeLists.txt` 加一行 `leetgpu_bench(...)`。
