# Kernel from Scratch

CUDA 算子练习，共 29 个题目 / 版本目录，分为 7 类。配套参考实现见 [Kernel-Reference](../Kernel-Reference/README.md)。

每题固定提供三个文件：

- `README.md`：题目说明、接口、参数含义、示例和约束。
- `solve.cu`：算子接口和练习实现；尚未完成的实现以 `// TODO` 标记。
- `bench.py`：Python 正确性检查和性能测试入口；未编写的测试保留为空文件。

当前向量加法、颜色反转和 ReLU 已有实现及 Python 测试，其余 26 个题目 / 版本保留练习模板。

所有指针参数均为 GPU 设备指针，由调用方分配内存。原地操作及输出形状在各题 README 中说明。
FP16 GEMM 使用 `cuda_fp16.h`；其余模板使用 `cuda_runtime.h`。

阅读顺序：Element-wise → Reduce / Scan → Permute → GEMM → Stencil → Attention → Sort。
Attention 保留 Naive、Online 和 Flash v2 三个版本目录，题面及接口相同。

## 题目索引

### [Element-wise / 逐元素](element-wise/README.md)

| 题目 | 目录 |
|---|---|
| 向量加法 | [element-wise/01-vector-add](element-wise/01-vector-add/README.md) |
| 颜色反转 | [element-wise/02-color-inversion](element-wise/02-color-inversion/README.md) |
| ReLU | [element-wise/03-relu](element-wise/03-relu/README.md) |
| Leaky ReLU | [element-wise/04-leaky-relu](element-wise/04-leaky-relu/README.md) |
| 矩阵拷贝 | [element-wise/05-matrix-copy](element-wise/05-matrix-copy/README.md) |
| SiLU | [element-wise/06-silu](element-wise/06-silu/README.md) |
| SWiGLU | [element-wise/07-swiglu](element-wise/07-swiglu/README.md) |
| Value Clipping | [element-wise/08-value-clipping](element-wise/08-value-clipping/README.md) |
| RGB 转灰度 | [element-wise/09-rgb-to-grayscale](element-wise/09-rgb-to-grayscale/README.md) |
| Sigmoid | [element-wise/10-sigmoid](element-wise/10-sigmoid/README.md) |
| GELU | [element-wise/11-gelu](element-wise/11-gelu/README.md) |
| GEGLU | [element-wise/12-geglu](element-wise/12-geglu/README.md) |

### [Reduce / 归约与扫描](reduce/README.md)

| 题目 | 目录 |
|---|---|
| Reduction | [reduce/01-reduction](reduce/01-reduction/README.md) |
| Softmax | [reduce/02-softmax](reduce/02-softmax/README.md) |
| Prefix Sum | [reduce/03-prefix-sum](reduce/03-prefix-sum/README.md) |
| Dot Product | [reduce/04-dot-product](reduce/04-dot-product/README.md) |

### [Permute / 下标重排](permute/README.md)

| 题目 | 目录 |
|---|---|
| 反转数组 | [permute/01-reverse-array](permute/01-reverse-array/README.md) |
| 矩阵转置 | [permute/02-matrix-transpose](permute/02-matrix-transpose/README.md) |
| 交错合并数组 | [permute/03-interleave-arrays](permute/03-interleave-arrays/README.md) |

### [GEMM / 矩阵乘加](gemm/README.md)

| 题目 | 目录 |
|---|---|
| 矩阵乘法（FP32） | [gemm/01-matrix-mul](gemm/01-matrix-mul/README.md) |
| SpMV | [gemm/02-spmv](gemm/02-spmv/README.md) |
| GEMM（FP16，α / β） | [gemm/03-gemm](gemm/03-gemm/README.md) |

### [Stencil / 卷积](stencil/README.md)

| 题目 | 目录 |
|---|---|
| 1D 卷积 | [stencil/01-conv1d](stencil/01-conv1d/README.md) |
| 2D 卷积 | [stencil/02-conv2d](stencil/02-conv2d/README.md) |
| 3D 卷积 | [stencil/03-conv3d](stencil/03-conv3d/README.md) |

### [Attention / 注意力](attention/README.md)

| 题目 | 目录 |
|---|---|
| Softmax Attention / Naive | [attention/01-naive](attention/01-naive/README.md) |
| Softmax Attention / Online | [attention/02-online](attention/02-online/README.md) |
| Softmax Attention / Flash v2 | [attention/03-flash-v2](attention/03-flash-v2/README.md) |

### [Sort / 排序与选择](sort/README.md)

| 题目 | 目录 |
|---|---|
| Top-K Selection | [sort/01-top-k-selection](sort/01-top-k-selection/README.md) |

## 运行已有测试

需要 NVIDIA GPU、CUDA Toolkit，以及支持本机 CUDA 环境的 PyTorch 和 Ninja。
在本项目目录运行，首次执行会通过 PyTorch 编译对应的 CUDA 扩展：

```bash
python element-wise/01-vector-add/bench.py
python element-wise/02-color-inversion/bench.py
python element-wise/03-relu/bench.py
```

## 补充资料

- [参考实现与学习路线](../Kernel-Reference/README.md)
- [按并行写法分类的题单](../Kernel-Reference/题目/00-分类.md)
- [CUDA Warp Shuffle 教程](../Kernel-Reference/科普/01-shuffle.md)
