# Element-wise

除 `07_swiglu`（2 读 1 写）外均为 1 读 1 写，纯带宽型，答案约等于 memcpy 上限。

| 题目 | 目录 | 难度 | 出题频率 | 备注 |
|---|---|---|---|---|
| Vector Add | `01_vector_add` | 入门 | 热身题 | 讲 grid-stride / 向量化的最小载体 |
| Color Inversion | `02_color_inversion` | 入门 | 极少 | 本质是逐字节取反，考 `uchar4` 语义 |
| ReLU | `03_relu` | 入门 | 偶尔 | 单独考少，多作为融合算子的组件 |
| Leaky ReLU | `04_leaky_relu` | 入门 | 极少 | 与 ReLU 等价，换皮 |
| Matrix Copy | `05_matrix_copy` | 入门 | 极少 |-|
| SiLU | `06_silu` | 简单 | 偶尔 | 引出 `__expf`；是 SwiGLU 的一半 |
| SwiGLU | `07_swiglu` | 简单 | 中 | LLM 高频融合算子，常以"silu(x)·x2"出现 |
| Value Clipping | `08_value_clipping` | 入门 | 少 | fminf/fmaxf，无考点 |
| RGB→Gray | `09_rgb_to_grayscale` | 简单 | 少 |-|
| Sigmoid | `10_sigmoid` | 入门 | 偶尔 | 同上，`__expf` 精度取舍 |



> 锐评: 难度拉完了，知识掌握难易程度给到夯。

面试开头热身。真正的高频考点是它和 reduce/scan 组合出的融合算子（RMSNorm、SwiGLU 门控、flash 类注意力里的 element-wise 部分）。
