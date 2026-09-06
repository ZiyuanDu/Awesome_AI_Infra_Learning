# Reduce

写法：grid-stride 局部状态 → warp/block shuffle →（可选）atomic / 二次 merge。

| 题目 | 目录 | 要点 |
|---|---|---|
| Reduction | `01_reduction` | sum；`blockReduceSum` + 每 block 一次 `atomicAdd` |
| Softmax | `02_softmax` | online `(m,ℓ)`；2 读 1 写；`onlineMerge` |
| Prefix Sum | `03_prefix_sum` | 面试三步；`common` 的 `blockScan*` + `load4` |
| Dot Product | `04_dot_product` | 融合 `Σ a·b`；同 `01` 骨架，多一路读 |

LeetGPU：Reduction、Softmax、Prefix Sum、Dot Product。按行 softmax / FA 见 019。
