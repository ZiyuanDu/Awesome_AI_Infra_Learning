# 面试算子速通

按 **并行写法** 分目录；每题只有 `solve.cu` + `bench.cu`。  
对照：[LeetGPU](https://leetgpu.com/challenges) · 归类：[题目/00-分类.md](题目/00-分类.md)

```
include/          common.cuh（reduce/scan/online/load4）+ bench.cuh
element_wise/     01–10 逐点 map
permute/          reverse · transpose · interleave
gemm/             matmul · spmv · gemm(fp16 αβ)
stencil/          conv1d/2d/3d
reduce/           reduction · softmax · prefix_sum · dot
attention/        naive · online · flash_v2
题目/ · 科普/
```

| 目录 | 模板 |
|---|---|
| `element_wise` | grid-stride，可 `float4` |
| `permute` | 下标置换；转置要 smem |
| `gemm` | tile + smem 外积；SpMV=按行 GEMV |
| `stencil` | 输出 tile + 输入 halo |
| `reduce` | shuffle / atomic / online `(m,ℓ)` / scan |
| `attention` | SDPA → online → FA2 |

```bash
cmake -B build -S . && cmake --build build -j
./build/reduce/01_reduction/bench          # 路径 = 分类/题号
```

新题：放进对应分类，根 `CMakeLists.txt` 加一行 `leetgpu_bench(...)`。
