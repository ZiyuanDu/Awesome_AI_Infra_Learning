# Normalization / 归一化

统计量归约后再逐元素变换，覆盖 LayerNorm、RMSNorm、残差融合和 BatchNorm。

共 **4 个题目 / 版本目录**。每题包含 `README.md`、`solve.cu` 和 `bench.py`。前置练习见各题题面；专项题可按目标岗位选做。

| 题目 | 目录 | 层级 |
|---|---|---|
| LayerNorm / 按行层归一化 | [01-layer-norm](01-layer-norm/README.md) | 基础 |
| Row-wise RMSNorm / 按行均方根归一化 | [02-row-rms-norm](02-row-rms-norm/README.md) | 基础 |
| Fused Residual Add + RMSNorm | [03-fused-add-rms-norm](03-fused-add-rms-norm/README.md) | 融合 |
| BatchNorm / 二维训练前向 | [04-batch-norm](04-batch-norm/README.md) | 专项 |
