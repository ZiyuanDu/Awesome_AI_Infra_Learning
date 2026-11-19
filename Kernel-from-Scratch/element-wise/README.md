# Element-wise / 逐元素与成对变换

本目录收录 10 道 LeetGPU CUDA 练习。元素、像素或固定元素对可以独立计算，重点是把下标映射到线程，并处理好连续访存、向量化对齐和尾部边界。

## 题目索引

| 题目 | 层级 | 练习重点 |
|---|---|---|
| [01 向量加法](01-vector-add/README.md) | 基础 | 一维线程索引、grid-stride loop、`float4` 向量化和标量尾部。 |
| [02 颜色反转](02-color-inversion/README.md) | 基础 | 用 `uchar4` 读取 RGBA 像素，原地反转 RGB 并保留 Alpha。 |
| [03 ReLU](03-relu/README.md) | 基础 | 逐元素将负值截为零。 |
| [04 Leaky ReLU](04-leaky-relu/README.md) | 基础 | 对负值保留固定斜率。 |
| [05 Sigmoid](05-sigmoid/README.md) | 基础 | 逐元素计算指数与倒数。 |
| [06 SiLU](06-silu/README.md) | 基础 | 将 Sigmoid 与输入的乘法放在同一个 kernel 中。 |
| [07 SWiGLU](07-swiglu/README.md) | 基础 | 将前后半段配对，融合 SiLU 门控与乘法。 |
| [08 RoPE / 旋转位置编码](08-rope/README.md) | 基础 | 在每一行内配对前后半维，计算旋转后的两个分量。 |
| [09 Weight Dequantization / 分块反量化](09-weight-dequantization/README.md) | 专项 | 用二维块坐标查找共享的 scale，处理边缘块。 |
| [10 Fused AdamW / 融合优化器更新](10-fused-adamw/README.md) | 专项 | 融合权重衰减、矩更新和偏差校正，原地写回参数与状态。 |

## 本地运行

每道题的 `README.md` 包含题面与接口，`solve.cu` 是 CUDA 实现或起始代码，`bench.py` 用于和 PyTorch 对照。以下命令从仓库根目录运行分块反量化的 benchmark。

```bash
python Kernel-from-Scratch/element-wise/09-weight-dequantization/bench.py
```