# Reduce / 归约与扫描

覆盖全局与按行归约、Softmax、损失计算及扫描。

共 **8 个题目 / 版本目录**。每题包含 `README.md`、`solve.cu` 和 `bench.py`。前置练习见各题题面；专项题可按目标岗位选做。

| 题目 | 目录 | 层级 |
|---|---|---|
| Reduction | [01-reduction](01-reduction/README.md) | 基础 |
| Softmax | [02-softmax](02-softmax/README.md) | 基础 |
| Prefix Sum | [03-prefix-sum](03-prefix-sum/README.md) | 基础 |
| Dot Product | [04-dot-product](04-dot-product/README.md) | 基础 |
| Row-wise Reduction / 行求和与最大值 | [05-row-reduction](05-row-reduction/README.md) | 基础 |
| Row-wise Softmax / 按行 Softmax | [06-row-softmax](06-row-softmax/README.md) | 基础 |
| Categorical Cross Entropy / 分类交叉熵 | [07-cross-entropy](07-cross-entropy/README.md) | 基础 |
| Mean Squared Error / 均方误差 | [08-mse](08-mse/README.md) | 基础 |

## 面试中复用什么

先掌握 **线程局部累积 → warp 归约 → block 归约 → 必要时再归约块结果**。已实现的 [reduce.cuh](reduce.cuh) 提供 `block_sum(value)`、`block_max(value)`，内部用 warp shuffle，只把每个 warp 的结果放进 shared memory。没有 CUB 依赖。

```cuda
float local = 0.0f;
for (int i = blockIdx.x * blockDim.x + threadIdx.x;
     i < N; i += blockDim.x * gridDim.x) {
    local += input[i];  // Dot Product 改成 A[i]*B[i]，MSE 改成平方差。
}
float total = block_sum(local);  // block 内所有线程都调用。
if (threadIdx.x == 0) partial[blockIdx.x] = total;
```

这是全局求和的第一阶段，`partial` 由调用方提供；第二阶段对块结果求和。矩阵按行归约时，把 `blockIdx.x` 映射为行号，把循环改为 `col=threadIdx.x; col<C; col+=blockDim.x`。

模板约定：一维 block，线程数为 32 的整数倍且不超过 1024；所有线程都参与调用。尾部越界元素用 sum 的 `0`、max 的 `-∞` 补齐，不能在归约前提前 `return`。返回结果对 block 内所有线程有效。模板只处理 FP32 sum/max；浮点加法的归约顺序变化可能带来舍入差异。同步规则可参考 [CUDA warp primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)。

## 这八题怎样现场写

除 Softmax 和共享模板外，以下是实现路线分析，尚未替其他题目的 `solve.cu` 填入实现。

| 题目 | 推荐先写出的结构 | 可复用部分 | 接着优化什么 |
|---|---|---|---|
| Reduction | 每线程累加多个元素，block 求和写 partial，再求和 | `block_sum` 与分层归约 | 连续访存、合适的 block 数、多个独立累加器；再测向量化 |
| Softmax | 分块缓存，求局部 max/sum，再合并并归一化 | `block_max`、`block_sum`、小统计量合并 | 减少显存访问、kernel launch 和 workspace 管理开销；已实现 |
| Prefix Sum | 块内 scan，递归 scan 块和，再给每块加偏移 | 分层组织与 shared memory 同步思路 | warp scan、连续元素分工；不能直接拿 `block_sum` 替代 |
| Dot Product | 把局部累加项替换成 `A[i]*B[i]` | 与 Reduction 相同 | 融合乘法和归约，不落地中间乘积数组 |
| Row Reduction | 一 block 一行，同时维护局部 sum/max | `block_sum`、`block_max` | 短行一 warp 一行；长行每线程多元素 |
| Row Softmax | 一 block 一行，寄存器保存该行的一部分，max→exp→sum→写回 | 与 Softmax 相同的块内模板 | 行数足够、行宽较小时一 warp 一行；行很长且行数少时再 split |
| MSE | 累加 `(pred-target)^2`，归约后只除一次总元素数 | Reduction 只改局部表达式和最后处理 | 融合差值、平方、求和，避免中间数组；大输入优先减少访存 |
| Cross Entropy | 每行算稳定 logsumexp，减正确类别 logit，最后对行损失取平均 | `block_max`、`block_sum` 与最终全局归约 | 不生成完整 Softmax；把行统计和标签读取融合 |

Reduction、Dot Product、MSE 属于同一类 **map + sum + 最后处理**；Row Reduction、Row Softmax、归一化属于 **行统计 + 逐元素变换**。优先复用这些结构，现场改动通常集中在局部表达式和最终输出。

### Prefix Sum 单独记 scan

本仓库要求 inclusive scan。warp 内最小骨架如下，要求完整 warp 的所有线程参与，lane 按原数组顺序对应连续元素：

```cuda
int lane = threadIdx.x & 31;
for (int offset = 1; offset < 32; offset <<= 1) {
    float left = __shfl_up_sync(0xffffffff, value, offset);
    if (lane >= offset) value += left;
}
```

block 内先保存每个 warp 的末尾总和，再 scan 这些 warp 总和，把前面 warps 的和加回当前 warp。跨 block 同样扫描块和，再加块偏移。对很长数组要递归处理块和，不能假定一 block 能扫描所有块结果。局部元素顺序必须保持；全局 sum 的任意 grid-stride 分组不能直接套成全局 scan。参考 [GPU Gems 的分层 scan 分析](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)。

### Cross Entropy 省掉整个 Softmax 输出

每行的最大值为 `m`，指数和为 `s`，正确类别的 logit 为 `x_label`：

$$
loss_{row}=(m-x_{label})+\log(s),\qquad
s=\sum_j\exp(x_j-m).
$$

优先按 `(m-x_label)+log(s)` 计算，避免先算大数 `m+log(s)` 再减 `x_label` 导致精度损失。每行只写一个 loss，最后求和并除以行数。不要先生成 Softmax 再做 `-log(p_label)`，那会增加访存，也可能让很小的概率先下溢为零。

## 在其他模块里怎么迁移

| 模块 | 与这套模板的关系 | 面试时优先写什么 |
|---|---|---|
| Element-wise | 通常不需要线程间归约 | 连续访存、grid-stride、尾部处理，再考虑 `float4` |
| Reduce | 当前模板的直接使用场景 | 先写 sum/max；scan 单独处理 |
| Normalization | 每行先统计再变换 | RMSNorm 用平方和；LayerNorm 先均值、再中心化方差；进阶再讲 Welford |
| Permute | 核心是下标与布局变换 | 转置优先讲 shared memory tile 和 bank conflict，通常不用 `block_sum` |
| Indexing | Gather 是读索引，Scatter 是冲突聚合 | 先明确写冲突；Embedding+LayerNorm 的统计部分可用块归约 |
| GEMM | Dot/GEMV 的局部求和可复用思路 | GEMM 的主要结构仍是矩阵分块与数据复用，不能只套一个全局 reduction |
| Stencil | 窗口内也有求和或最大值 | 小窗口先线程内展开，池化和卷积不必为每个小窗口使用 block 归约 |
| Attention | score 的 max/sum 可复用；跨 tile 维护 online 状态 | 先讲稳定 Softmax，再讲输出累加器随最大值变化而重标定及矩阵分块 |
| Sort | sum/max 只覆盖部分步骤 | Top-K 需要选择或比较；Top-p 的累计概率需要 scan |

LayerNorm 不宜直接用 `E[x²]-E[x]²` 作为通用稳定方差方案：大均值、小方差输入会发生消减。能先写对中心化两遍方差，再把 Welford 当进阶讨论。

性能收益要逐题验证，目前这里实测完成的是 Softmax。优先寻找 MSE、Cross Entropy、残差加法与 RMSNorm 等能消除中间数组或重复读取的融合机会。对于已经接近启动延迟或访存限制的逐元素算子，换一套归约模板不会带来普遍加速；GEMM、卷积则需要分别面对成熟的 cuBLAS/cuDNN 路径，不能承诺用简短手写模板在所有形状上明显超越它们。后续每题都应同时记录具体形状、dtype、普通调用和 GPU 侧时间，再决定保留哪些优化。

现场优先顺序：**明确形状和归约轴 → 写稳定且无同步错误的版本 → 减少中间数组 → 改线程分工 → 实测后决定向量化或更复杂的流水**。单一实现无法在所有行宽、行数和 GPU 上同时最优；[OneFlow 的分段设计](https://oneflow2020.medium.com/how-to-implement-an-efficient-softmax-cuda-kernel-oneflow-performance-optimization-sharing-405ad56e9031) 和 [Maharshi 的实作记录](https://maharshi.bearblog.dev/optimizing-softmax-cuda/) 适合用来理解这些取舍，而不是直接照搬其中的性能数字。
