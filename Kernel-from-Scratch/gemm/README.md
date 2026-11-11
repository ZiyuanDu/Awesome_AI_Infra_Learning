# GEMM / 矩阵与向量乘

矩阵乘、矩阵向量乘、批处理、低精度与后处理融合。

每题包含 `README.md`、`solve.cu` 和 `bench.py`。前置练习见各题题面；专项题可按目标岗位选做。

| 题目 | 目录 | 层级 |
|---|---|---|
| 矩阵乘法（FP32） | [01-matrix-mul](01-matrix-mul/README.md) | 基础 |
| SpMV | [02-spmv](02-spmv/README.md) | 基础 |
| GEMM（FP16，α / β） | [03-gemm](03-gemm/README.md) | 基础 |
| GEMV / 矩阵向量乘法 | [04-gemv](04-gemv/README.md) | 基础 |
| Batched GEMM / FP32 批量矩阵乘 | [05-batched-matmul](05-batched-matmul/README.md) | 基础 |
| Batched GEMM / FP16 输入输出 | [06-fp16-batched-matmul](06-fp16-batched-matmul/README.md) | 专项 |
| GEMM + Bias + GELU / 融合后处理 | [07-fused-bias-gelu](07-fused-bias-gelu/README.md) | 融合 |
| INT8 Quantized MatMul / 量化矩阵乘 | [08-int8-quantized-matmul](08-int8-quantized-matmul/README.md) | 专项 |
| W4A16 GEMM / INT4 权重量化矩阵乘 | [09-int4-weight-only-matmul](09-int4-weight-only-matmul/README.md) | 专项 |
