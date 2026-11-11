# Normalization / 归一化

统计量归约后再逐元素变换；分清归约轴、仿射参数和前后向。

每题包含 `README.md`、`solve.cu` 和 `bench.py`。前置练习见各题题面；专项题可按目标岗位选做。

| 题目 | 目录 | 层级 |
|---|---|---|
| LayerNorm / 按行层归一化 | [01-layer-norm](01-layer-norm/README.md) | 基础 |
| RMSNorm / 官方一维标量仿射版本 | [02-vector-rms-norm](02-vector-rms-norm/README.md) | 基础 |
| Row-wise RMSNorm / 按行均方根归一化 | [03-row-rms-norm](03-row-rms-norm/README.md) | 基础 |
| Fused Residual Add + RMSNorm | [04-fused-add-rms-norm](04-fused-add-rms-norm/README.md) | 融合 |
| BatchNorm / 二维训练前向 | [05-batch-norm](05-batch-norm/README.md) | 专项 |
| GroupNorm / 分组归一化 | [06-group-norm](06-group-norm/README.md) | 专项 |
| LayerNorm Backward / 带保存统计量的反向 | [07-layer-norm-backward](07-layer-norm-backward/README.md) | 专项 |
| RMSNorm Backward / 按行反向 | [08-rms-norm-backward](08-rms-norm-backward/README.md) | 专项 |
