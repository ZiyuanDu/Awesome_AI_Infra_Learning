# Kernel from Scratch

CUDA 算子练习，共 **53 个题目 / 版本目录**，按计算方式分为 **9 类**。

## 文件约定

- 题目目录：各模块独立使用两位编号，从 `01` 开始连续排列，格式为 `NN-题目名`。
- `README.md`：公式、布局、接口、参数、示例、约束和前置练习。
- `solve.cu`：新增题目提供必要头文件、`CEIL` 宏和带 `// TODO` 的 `solve` 接口。
- `bench.py`：新增题目为空，后续自行补正确性与性能测试。

## 题目索引

### [Element-wise / 逐元素与成对变换](element-wise/README.md)

| 题目 | 目录 | 层级 |
|---|---|---|
| 向量加法 | [element-wise/01-vector-add](element-wise/01-vector-add/README.md) | 基础 |
| 颜色反转 | [element-wise/02-color-inversion](element-wise/02-color-inversion/README.md) | 基础 |
| ReLU | [element-wise/03-relu](element-wise/03-relu/README.md) | 基础 |
| Leaky ReLU | [element-wise/04-leaky-relu](element-wise/04-leaky-relu/README.md) | 基础 |
| Sigmoid | [element-wise/05-sigmoid](element-wise/05-sigmoid/README.md) | 基础 |
| SiLU | [element-wise/06-silu](element-wise/06-silu/README.md) | 基础 |
| SWiGLU | [element-wise/07-swiglu](element-wise/07-swiglu/README.md) | 基础 |
| GELU | [element-wise/08-gelu](element-wise/08-gelu/README.md) | 基础 |
| GEGLU | [element-wise/09-geglu](element-wise/09-geglu/README.md) | 基础 |
| RoPE / 旋转位置编码 | [element-wise/10-rope](element-wise/10-rope/README.md) | 基础 |
| Weight Dequantization / 分块反量化 | [element-wise/09-weight-dequantization](element-wise/09-weight-dequantization/README.md) | 专项 |
| Fused AdamW / 融合优化器更新 | [element-wise/10-fused-adamw](element-wise/10-fused-adamw/README.md) | 专项 |

### [Reduce / 归约与扫描](reduce/README.md)

| 题目 | 目录 | 层级 |
|---|---|---|
| Reduction | [reduce/01-reduction](reduce/01-reduction/README.md) | 基础 |
| Softmax | [reduce/02-softmax](reduce/02-softmax/README.md) | 基础 |
| Prefix Sum | [reduce/03-prefix-sum](reduce/03-prefix-sum/README.md) | 基础 |
| Dot Product | [reduce/04-dot-product](reduce/04-dot-product/README.md) | 基础 |
| Row-wise Reduction / 行求和与最大值 | [reduce/05-row-reduction](reduce/05-row-reduction/README.md) | 基础 |
| Row-wise Softmax / 按行 Softmax | [reduce/06-row-softmax](reduce/06-row-softmax/README.md) | 基础 |
| Categorical Cross Entropy / 分类交叉熵 | [reduce/07-cross-entropy](reduce/07-cross-entropy/README.md) | 基础 |
| Mean Squared Error / 均方误差 | [reduce/08-mse](reduce/08-mse/README.md) | 基础 |

### [Normalization / 归一化](normalization/README.md)

| 题目 | 目录 | 层级 |
|---|---|---|
| LayerNorm / 按行层归一化 | [normalization/01-layer-norm](normalization/01-layer-norm/README.md) | 基础 |
| Row-wise RMSNorm / 按行均方根归一化 | [normalization/02-row-rms-norm](normalization/02-row-rms-norm/README.md) | 基础 |
| Fused Residual Add + RMSNorm | [normalization/03-fused-add-rms-norm](normalization/03-fused-add-rms-norm/README.md) | 融合 |
| BatchNorm / 二维训练前向 | [normalization/04-batch-norm](normalization/04-batch-norm/README.md) | 专项 |

### [Permute / 下标重排](permute/README.md)

| 题目 | 目录 | 层级 |
|---|---|---|
| 反转数组 | [permute/01-reverse-array](permute/01-reverse-array/README.md) | 基础 |
| 矩阵转置 | [permute/02-matrix-transpose](permute/02-matrix-transpose/README.md) | 基础 |
| 交错合并数组 | [permute/03-interleave-arrays](permute/03-interleave-arrays/README.md) | 基础 |

### [Indexing / 索引访存与聚合](indexing/README.md)

| 题目 | 目录 | 层级 |
|---|---|---|
| Gather / 一维索引读取 | [indexing/01-gather](indexing/01-gather/README.md) | 基础 |
| Embedding Lookup / 基础查表 | [indexing/02-embedding-lookup](indexing/02-embedding-lookup/README.md) | 基础 |
| Scatter Add / 按索引聚合 | [indexing/03-scatter-add](indexing/03-scatter-add/README.md) | 基础 |
| Token Embedding Layer / 查表与 LayerNorm 融合 | [indexing/04-fused-token-embedding](indexing/04-fused-token-embedding/README.md) | 融合 |

### [GEMM / 矩阵与向量乘](gemm/README.md)

| 题目 | 目录 | 层级 |
|---|---|---|
| 矩阵乘法（FP32） | [gemm/01-matrix-mul](gemm/01-matrix-mul/README.md) | 基础 |
| SpMV | [gemm/02-spmv](gemm/02-spmv/README.md) | 基础 |
| GEMM（FP16，α / β） | [gemm/03-gemm](gemm/03-gemm/README.md) | 基础 |
| GEMV / 矩阵向量乘法 | [gemm/04-gemv](gemm/04-gemv/README.md) | 基础 |
| Batched GEMM / FP32 批量矩阵乘 | [gemm/05-batched-matmul](gemm/05-batched-matmul/README.md) | 基础 |
| GEMM + Bias + GELU / 融合后处理 | [gemm/06-fused-bias-gelu](gemm/06-fused-bias-gelu/README.md) | 融合 |
| INT8 Quantized MatMul / 量化矩阵乘 | [gemm/07-int8-quantized-matmul](gemm/07-int8-quantized-matmul/README.md) | 专项 |
| W4A16 GEMM / INT4 权重量化矩阵乘 | [gemm/08-int4-weight-only-matmul](gemm/08-int4-weight-only-matmul/README.md) | 专项 |

### [Stencil / 卷积与池化](stencil/README.md)

| 题目 | 目录 | 层级 |
|---|---|---|
| 1D 卷积 | [stencil/01-conv1d](stencil/01-conv1d/README.md) | 基础 |
| 2D 卷积 | [stencil/02-conv2d](stencil/02-conv2d/README.md) | 基础 |
| 3D 卷积 | [stencil/03-conv3d](stencil/03-conv3d/README.md) | 基础 |
| MaxPool2d / 二维最大池化 | [stencil/04-max-pool2d](stencil/04-max-pool2d/README.md) | 专项 |
| AvgPool2d / 二维平均池化 | [stencil/05-avg-pool2d](stencil/05-avg-pool2d/README.md) | 专项 |
| Depthwise Conv2d / 逐通道卷积 | [stencil/06-depthwise-conv2d](stencil/06-depthwise-conv2d/README.md) | 专项 |

### [Attention / 注意力](attention/README.md)

| 题目 | 目录 | 层级 |
|---|---|---|
| Softmax Attention / Naive | [attention/01-naive](attention/01-naive/README.md) | 基础 |
| Softmax Attention / Online | [attention/02-online](attention/02-online/README.md) | 基础 |
| Softmax Attention / Flash v2 | [attention/03-flash-v2](attention/03-flash-v2/README.md) | 基础 |
| Causal Self-Attention / 因果自注意力 | [attention/04-causal-attention](attention/04-causal-attention/README.md) | 专项 |
| Paged KV-Cache Attention / 分页解码 | [attention/05-paged-kv-attention](attention/05-paged-kv-attention/README.md) | 专项 |

### [Sort / 排序、选择与采样](sort/README.md)

| 题目 | 目录 | 层级 |
|---|---|---|
| Top-K Selection | [sort/01-top-k-selection](sort/01-top-k-selection/README.md) | 基础 |
| Top-p Sampling / 确定性核采样 | [sort/02-top-p-sampling](sort/02-top-p-sampling/README.md) | 专项 |
| MoE Top-K Gating / 专家选择与权重 | [sort/03-moe-topk-gating](sort/03-moe-topk-gating/README.md) | 融合 |

## 参考资料

- [LeetGPU 官方题库](https://github.com/AlphaGPU/leetgpu-challenges)
- [Triton 官方基础算子教程](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [FlashInfer 算子文档](https://docs.flashinfer.ai/)
