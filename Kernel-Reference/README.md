# Kernel Reference

CUDA 算子参考实现，共 27 个题目 / 版本，按并行写法分为 7 类。
配套练习见 [Kernel-from-Scratch](../Kernel-from-Scratch/README.md)，原题见 [LeetGPU](https://leetgpu.com/challenges)。

每题以 `solve.cu` 和 `bench.cu` 为主，公共实现位于 `include/`。
题面和技术笔记保留在 [题目/](题目/) 与 [科普/](科普/)，供练习时对照。

## 目录与阅读顺序

| 目录 | 内容 |
|---|---|
| [01_element_wise](01_element_wise/) | 向量加法、颜色反转、激活函数等逐元素运算 |
| [02_reduce](02_reduce/README.md) | Reduction、Softmax、Prefix Sum、Dot Product |
| [03_permute](03_permute/) | 数组反转、矩阵转置、数组交错 |
| [04_gemm](04_gemm/README.md) | FP32 矩阵乘法、SpMV、FP16 GEMM |
| [05_stencil](05_stencil/README.md) | 1D / 2D / 3D 卷积 |
| [06_attention](06_attention/) | Naive、Online、Flash v2 风格的 Attention |
| [07_sort](07_sort/) | Top-K Selection |
| [include](include/) | Reduce、Scan、Online Softmax、向量化读写和基准测试工具 |

按表格顺序阅读：先熟悉逐元素运算和线程通信，再学习共享内存分块，最后组合 Attention 与 Top-K。

## 构建与运行

需要 CMake 3.18+、支持 C++17 的编译器、CUDA Toolkit 和 NVIDIA GPU。
在本项目目录执行：

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --parallel
./build/02_reduce/01_reduction/bench
```

默认架构为本机 RTX 5090 的 `120`；使用其他 GPU 时传入对应的架构值。FP16 GEMM 中的 `cp.async` 要求 SM 80 或更高。
目录改名后，已有 CMake 缓存仍可能记录旧路径；首次构建可换用新的输出目录，例如将命令中的 `build` 改为 `build/reference`。

新增题目时放入对应分类，并在 `CMakeLists.txt` 中添加 `leetgpu_bench(...)`。

## 博客与题单

- [按并行写法分类的题单](题目/00-分类.md)
- [CUDA Warp Shuffle 教程](科普/01-shuffle.md)
- [Softmax](题目/16-Softmax.md)
- [Softmax Attention](题目/17-SoftmaxAttention.md)
