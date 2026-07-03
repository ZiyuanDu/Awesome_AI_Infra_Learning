# 016-Conv — Triton / PyTorch Conv2D

Conv2D (stride=1) 的 Triton 与 PyTorch 对比实现，展示从 naive direct convolution 到
Implicit GEMM 的完整优化历程。递进 5 版，每步只引入一个概念，性能单调递增。

> **关于 GPU 硬件层优化**（shared memory、tiling、double buffering、cp.async、Tensor Core）
> 已在 [015-Matmul](../015-Matmul) 的 CUDA 实现里完整讲解。本章聚焦**算子层**的优化思路
> （数据复用 / 算术强度 / im2col / 隐式 GEMM），用 Triton 表达以避免与矩阵乘的硬件细节重复。

## 项目结构

```
016-Conv/
├── README.md
│
├── ttriton/                         # Triton 实现 (递进优化 5 版)
│   ├── common.py                    # 公共工具:参数解析/计时/显存/校验/表格
│   ├── conv_v0_naive.py             # v0 朴素直接卷积 (基线)
│   ├── conv_v1_spatial.py           # v1 空间分块 → 数据复用
│   ├── conv_v2_tiled.py             # v2 空间+通道分块 → 提高算术强度
│   ├── conv_v3_im2col.py            # v3 im2col + GEMM (cuBLAS)
│   ├── conv_v4_implicit_gemm.py     # v4 Implicit GEMM (cuDNN 主力算法)
│   └── bench.py                     # 正确性 + 性能 + 显存对比 benchmark
│
├── ppytorch/                        # PyTorch 参考实现
│   ├── conv2d.py                    # im2col 原理 + conv2d 等价性验证
│   └── bench.py                     # PyTorch 性能对比
│
└── slides/                          # 配套课件 (卷积/滤波/边缘检测/CNN 等)
```

## Triton Kernel 优化路线

| 版本 | 技术 | 核心思想 |
|------|------|----------|
| **v0** | Naive direct | 每 program 一个输出元素，零复用（基线） |
| **v1** | Spatial tiling | 一 program 算 TILE_H×TILE_W 输出块 → 输入被相邻像素**复用** |
| **v2** | Spatial + channel | 再对 C_out 分块，一块输入服务 BLOCK_K 个通道 → 提高**算术强度**（GEMM 雏形） |
| **v3** | im2col + GEMM | 换思路：`F.unfold` 摊平成矩阵，交给 cuBLAS。代价是 col 显存物化 |
| **v4** | Implicit GEMM | 把 im2col 融进 `tl.dot`，坐标现算不落地 → cuDNN 主力算法 |

两条正交的主线：**v0→v1→v2** 是"直接卷积 + 分块复用"，**v3→v4** 是"归约成矩阵乘"。
v2 的外积累加是 v4 `tl.dot` 的手写雏形，二者串起了从 direct 到 GEMM 的认知桥梁。

## 快速开始

### 环境要求

| 组件 | 要求 |
|------|------|
| PyTorch | `pip install torch` |
| Triton | `pip install triton` (随 PyTorch 附带) |
| rich | `pip install rich` (benchmark 表格渲染) |

### 运行

```bash
# 完整 benchmark：正确性 + 性能 + 显存对比 (5 版 vs cuDNN)
python -m ttriton.bench

# 单独验证某一版的正确性 (每个文件含教学注释)
python -m ttriton.conv_v0_naive
python -m ttriton.conv_v4_implicit_gemm

# PyTorch im2col 原理演示
python ppytorch/conv2d.py
```

## Benchmark 结果 (RTX 4090 D, sm_89, TF32)

compute-bound 问题集，GFLOPS 越高越好。v0→v4 严格单调，v4 Implicit GEMM 在 3×3 上超过 cuDNN：

```
Problem                          v0      v1      v2       v3        v4      cuDNN
ResNet-mid 3×3 C=64  56×56 ×16   23    3777    5081     6213     40473    34898
Deep-layer 3×3 C=128 28×28 ×32   23    3156    4166    10789     54368    47485
Wide-conv  3×3 C=256 28×28 ×8    23    3120    4126    19138     49113    47172
First-layer 7×7 C=3  224   ×8    30    5527    7968     4843*    15795    16021
                                                                  (GFLOPS)
* v3 im2col 在 7×7 上反而慢于 v2：kernel 越大，col 矩阵物化越大 (峰值 648 MB)，
  访存成本压过 GEMM 收益。这是 im2col 的经典短板，v4 隐式 GEMM 不物化故不受影响。
```

> **为什么用 compute-bound 问题集？** 小问题（<1 GFLOP）在现代 GPU 上是延迟/启动瓶颈，
> 算术强度优化（v2）看不出差别，甚至因寄存器压力比 v1 慢，破坏"越优化越快"的教学叙事。
> 只有放大 batch / 通道到计算密集，才能显出 v0→v4 的单调阶梯。

## 关键教学点

1. **v0→v1 是最大跳变 (~100–180x)**：不是算得更少，而是把每输出一次的独立访存摊薄到
   一个 tile，相邻输出复用输入，占用率和带宽利用率大幅提升。

2. **v2 提高算术强度**：一次输入访存服务 BLOCK_K 个输出通道（外积累加），FLOP/Byte 随
   BLOCK_K 线性提高。这是 GEMM 的雏形。

3. **v3 vs v4 的显存差距**：都是"卷积=矩阵乘"，但 v3 把 im2col 的列矩阵物化到显存
   （放大 kH·kW 倍），v4 用坐标现算把展开融进 `tl.dot`，不落地。峰值显存差 10 倍，
   这正是 cuDNN 用 Implicit GEMM 而非朴素 im2col 的原因。

4. **TF32 精度**：`tl.dot` 在 Ampere+ 默认走 TF32 Tensor Core，逐元素绝对误差可达 ~1e-2，
   但归一化误差 (nrmse) <1e-3。正确性校验统一用 `common.assert_close`（nrmse 口径），
   与 benchmark 判定一致。要更高精度可传 `allow_tf32=False`。

## 文件说明

| 文件 | 说明 |
|------|------|
| `ttriton/conv_v0_naive.py` | v0 朴素直接卷积：每 program 一输出元素，零复用 |
| `ttriton/conv_v1_spatial.py` | v1 空间分块：TILE_H×TILE_W 输出块，输入复用 |
| `ttriton/conv_v2_tiled.py` | v2 空间+通道分块：外积累加，提高算术强度 |
| `ttriton/conv_v3_im2col.py` | v3 im2col + cuBLAS GEMM (F.unfold + matmul) |
| `ttriton/conv_v4_implicit_gemm.py` | v4 Implicit GEMM：tl.dot + 隐式 im2col |
| `ttriton/common.py` | 参数解析 / 计时 / 显存 / 正确性校验 / Rich 表格 |
| `ttriton/bench.py` | Triton benchmark：5 版 vs cuDNN 对比 |
| `ppytorch/conv2d.py` | PyTorch conv2d 参考 + im2col 原理讲解 |
| `ppytorch/bench.py` | PyTorch 性能对比 |

## License

MIT
