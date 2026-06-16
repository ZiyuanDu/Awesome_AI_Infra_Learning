# CUDA 矩阵乘法优化全景指南

> 涵盖从 naive 到接近 cuBLAS 的所有标准优化技术  
> 基于 LeetCUDA + 当前 mycuda 实现 + 文献综述

---

## 1. 优化技术分类体系

矩阵乘法优化可分为三层：**数据复用层**、**指令效率层**、**硬件特化层**。

```
                         ┌──────────────────────────┐
                         │   3. 硬件特化层           │
                         │   Tensor Core (WMMA/MMA)  │
                         │   cp.async 异步拷贝       │
                         │   Swizzle (SMEM XOR)      │
                         │   Block Swizzle (L2)      │
                         ├──────────────────────────┤
                         │   2. 指令效率层           │
                         │   float4 向量化           │
                         │   FMA intrinsic           │
                         │   Loop unrolling          │
                         │   Bank Conflict Free      │
                         ├──────────────────────────┤
                         │   1. 数据复用层           │
                         │   Shared Memory Tiling    │
                         │   Warp/Thread Tiling      │
                         │   Double Buffering        │
                         │   Multi-stage Pipeline    │
                         └──────────────────────────┘
```

---

## 2. 数据复用层 — 减少全局内存访问

### 2.1 Shared Memory Tiling（共享内存分块）

**解决的问题**: 每个全局内存元素被重复加载 M×N 次

**原理**: 将 A 和 B 的子块加载到 SMEM，block 内所有线程共享。每个全局元素只加载一次。

```
无 tiling: AI ≈ 0.25 FLOPs/byte → 极端访存受限
有 tiling: AI ≈ 8-32 FLOPs/byte → 大幅改善
```

**关键参数**:
- BM, BN: 输出 tile 大小（典型值 64-256）
- BK: K 维度 tile 大小（CUDA Core: 8-32, Tensor Core: 固定为 WK=8/16）

**tradeoff**: BM×BN 越大，全局访存越少，但 SMEM 和寄存器需求越大。

**标准性**: ⭐⭐⭐⭐⭐（必修）

### 2.2 Warp/Thread Tiling（线程级分块）

**解决的问题**: 每个线程只算一个输出元素 → 计算密度低，指令开销占比大

**原理**: 每个线程维护 TM×TN 个累加器（如 8×8=64 个），大幅减少指令/FLOP 比。

```
每线程 1 个累加器: 1 FMA/2 loads → 50% loads 指令
每线程 64 个累加器: 64 FMA/2 loads → 3% loads 指令
```

**关键参数**:
- TM, TN: 每线程计算的输出子块（典型值 4-16）
- TM×TN 越大 → 寄存器需求越高 → occupancy 越低

**常见配置**:

| 配置 | TM×TN | 寄存器 | Occupancy (256 thr/blk) | 适用场景 |
|------|-------|--------|------------------------|---------|
| 8×4 | 32 | ~80 | 3 blk/SM (50%) | 小矩阵、低寄存器压力 |
| 8×8 | 64 | ~126 | 2 blk/SM (33%) | 通用（当前默认） |
| 8×16 | 128 | 溢出(spill) | — | 不推荐（寄存器溢出） |

**标准性**: ⭐⭐⭐⭐⭐（必修）

### 2.3 Double Buffering（双缓冲）

**解决的问题**: 全局加载和计算串行执行，计算单元等待数据

**原理**: 分配 2× SMEM，加载 tile N+1 的同时计算 tile N。

```
无 DB:  | load0 | B | comp0 | B | load1 | B | comp1 | ...
有 DB:  | load0 | B | comp0     | B | comp1     | ...
                | load1        | load2        |
```

**SMEM 成本**: 2× 基础用量

**标准性**: ⭐⭐⭐⭐（强烈推荐）

### 2.4 Multi-stage Pipeline（多级流水线）

**解决的问题**: 双缓冲只能隐藏 1 个 tile 的加载延迟，对于高延迟的全局加载可能不够

**原理**: 分配 K_STAGE × SMEM，形成更深的流水线。

```
2-stage (双缓冲): 预取深度=1
3-stage:          预取深度=2
4-stage:          预取深度=3
```

**关键洞察**: Stage 不是越多越好！

LeetCUDA 实测数据: **2-stage 比 3-stage 快 10-16%**（在 WMMA TF32 路径上）。

原因:
- 更多 stage → 更多 SMEM → 更低 occupancy
- SMEM 减少 → 更少的 block/SM → 更少的 warp 隐藏延迟
- 在某个点上，SMEM 的边际成本超过了预取的边际收益

**何时 stage 多有益**:
- 全局加载延迟非常大时（如 HBM 带宽低的 GPU）
- 使用 MMA PTX 而非 WMMA（MMA 无 WARPSYNC 开销，流水线更紧凑）
- Occupancy 不受 SMEM 限制时

**标准性**: ⭐⭐⭐（进阶）

---

## 3. 指令效率层 — 提高计算吞吐

### 3.1 float4 向量化加载/存储

**解决的问题**: 标量全局加载只能利用 1/4 的内存事务宽度

**原理**: 一次 128-bit 读取 4 个 float，将 4 次内存事务合并为 1 次。

```cuda
float4 val = *(const float4*)&A[addr];  // 1 条 128-bit LDG 指令
// vs
float a0 = A[addr+0];  // 4 条 32-bit LDG 指令
float a1 = A[addr+1];
float a2 = A[addr+2];
float a3 = A[addr+3];
```

**标准性**: ⭐⭐⭐⭐⭐（必修）

### 3.2 FMA Intrinsic

**解决的问题**: 编译器可能不将 `a*b+c` 识别为 FMA，导致额外的乘法+加法指令

**原理**: 

```cuda
accum += a * b;       // 可能编译为: MUL + ADD (2 条指令)
accum = fmaf(a, b, accum);  // 保证编译为: FFMA (1 条指令)
```

**标准性**: ⭐⭐⭐⭐⭐（必修）

### 3.3 Bank Conflict Free (BCF) SMEM 布局

**解决的问题**: Shared memory 的 32 个 bank 被多个线程同时访问时产生冲突

**Bank Conflict 原理**:
```
SMEM 有 32 个 bank，每个 bank 4 bytes 宽（与 float 大小一致）
同一 cycle 内，不同线程访问同一 bank 的不同地址 → 串行化
同一 cycle 内，多个线程访问同一 bank 的同一地址 → 广播（无冲突）
```

**矩阵乘法的 bank conflict 来源**:

```
A 矩阵（行优先存储在 SMEM 中）:
  计算阶段：每线程读取 sA[ty*TM + m][k] (m=0..TM-1)
  行 stride = BK（如 BK=8 → stride=8 floats = 32 bytes = 8 banks）
  问题：stride 太小，相邻行落在相同 bank 组

B 矩阵（行优先存储在 SMEM 中）:
  计算阶段：每线程读取 sB[k][tx*TN + n] (n=0..TN-1)
  行 stride = BN（如 BN=128 → stride=128 floats = 512 bytes）
  128 % 32 = 0 → 同一列位置每行落在相同的 bank
```

**BCF 解决方案**:

1. **A 矩阵转置存储**: `sA[k][m]` 而非 `sA[m][k]`
   - 加载阶段：每个线程散布写入 sA[k][m]，k 维分散 → 不同 bank
   - 计算阶段：连续读 sA[k][m] for k=0..BK-1 → stride = BM+OFFSET → 更大 stride 减少冲突

2. **SMEM Padding (OFFSET)**: 在行尾添加 padding
   - `sA[BK][BM]` → `sA[BK][BM+OFFSET]` (OFFSET 典型值=4)
   - stride 从 BM 变为 BM+OFFSET，打破 32 的倍数关系
   - 132 % 32 = 4 → 每行偏移 4 banks → 8 行循环回同一 bank

3. **XOR Swizzle**（MMA PTX 级别）:
   - 对列索引做 XOR 变换: `swizzled_col = col ^ (col & mask)`
   - 彻底消除 bank conflict，但需要更多地址计算

**当前实现的状态**:

| 版本 | A 矩阵 SMEM | B 矩阵 SMEM | Bank Conflict |
|------|------------|------------|---------------|
| v2 | sA[BM][BK] (128×8) | sB[BK][BN] (8×128) | A: 2-way, B: 4-way |
| v3/v4 | sA[BK][BM+4] (8×132) | sB[BK][BN+4] (8×132) | A: 2-way, B: ~2-way (avg) |
| v6 (BCF) | sA[BK][BM+4] (16×132) | sB[BK][BN+4] (16×132) | A: 冲突消除（写入阶段），B: ~2-way |

**值得作为新优化点吗？**

短期（CUDA Core v6）: **值得**。BCF + BK=16 组合提供 ~10-15% 额外提升。
中期（WMMA）: **不需要**。WMMA API 内部管理 SMEM 布局，B_PAD 足够。
长期（MMA PTX）: **必须**。MMA PTX 需要 XOR swizzle 达到接近 cuBLAS 的性能。

**标准性**: ⭐⭐⭐（进阶优化，非必修但有价值）

### 3.4 BK 大小选择

**CUDA Core 路径**:

| BK | SMEM/block | tile 迭代数 (K=2048) | Barrier 次数 | 推荐 |
|----|-----------|---------------------|-------------|------|
| 8 | ~8 KB | 256 | 512 | 小 GPU、低 SMEM |
| **16** | **~16 KB** | **128** | **256** | **推荐（最佳平衡）** |
| 32 | ~32 KB | 64 | 128 | 受限于 SMEM（降低 occ） |

**Tensor Core (WMMA) 路径**:

BK 必须与 WK 对齐：
- WK=8: BK=8（1 轮 WMMA）或 BK=16（2 轮 WMMA）
- BK=8 是更优选择：SMEM 最小，WMMA 指令最少

**标准性**: ⭐⭐⭐⭐

---

## 4. 硬件特化层 — 利用专用硬件

### 4.1 cp.async (Async Copy)

**解决的问题**: 全局→共享内存的拷贝占用 LD/ST 流水线，阻塞计算指令发射

**原理**: 使用专用的异步拷贝引擎 (ACE)，全局加载不经过 SM 的 LD/ST 单元。

```cuda
// 同步加载：占用 LD/ST pipeline
float4 val = *(float4*)&A[addr];  sA[r][c] = val.x; ...

// 异步加载：ACE 独立执行
cp.async.cg.shared.global [dst], [src], 16;  // 不占用 LD/ST!
```

**cp.async 变体**:

| 变体 | L1 | L2 | 适用场景 |
|------|----|----|---------|
| `.ca` | ✅ cache | ✅ cache | 数据可能被复用 |
| `.cg` | ❌ bypass | ✅ cache | **流式数据**（GEMM 输入） |
| `.cs` | ❌ bypass | ❌ bypass | 纯 streaming |

**GEMM 中应该用 `.cg`**:
- A 和 B 矩阵的每个元素在每个 tile 中只读一次
- 缓存到 L2 足够（后续 tile 可能命中 L2）
- 不污染 L1（L1 留给栈变量、索引计算等）

**标准性**: ⭐⭐⭐⭐（sm_80+，强烈推荐）

### 4.2 WMMA (Warp Matrix Multiply-Accumulate)

**解决的问题**: CUDA Core 的 FFMA 指令在每个 SM 上受限于 128 FP32 ops/cycle

**原理**: 使用 Tensor Core，每个 SM 可达 512 TF32 ops/cycle（4× 提升）

**WMMA API 的隐藏成本**:
- `_sync` 后缀函数隐含 WARPSYNC → SASS 中的 WARPSYNC 指令 → 流水线停顿
- Fragment 管理有类型系统开销
- SMEM 布局受限于 API 约束

**关键参数**:
```
BM = WM × WARP_M × WTM
BN = WN × WARP_N × WTN

WM=16, WN=16 (WMMA 指令固定)
WARP_M=2, WARP_N=4 → 8 warps, 256 threads
WTM=4, WTN=2 → 每 warp 8 个 fragment
```

**WMMA vs MMA PTX**:

| 特性 | WMMA API | MMA PTX |
|------|----------|---------|
| 易用性 | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| 性能天花板 | ~85% cuBLAS | ~100% cuBLAS |
| WARPSYNC 开销 | 隐式 (每个 _sync) | 显式控制 |
| Fragment 管理 | 模板类型 | 手动寄存器数组 |
| SMEM swizzle | 受限于 API | 完全自由 |
| 编译依赖 | sm_70+ | sm_80+ (ldmatrix) |

**标准性**: ⭐⭐⭐（Tensor Core 入门），MMA PTX ⭐⭐⭐⭐（极致性能）

### 4.3 Block Swizzle（L2 Cache 优化）

**解决的问题**: 默认的 block 调度顺序（row-major）对 L2 cache 不友好

**原理**: 重新排列 block 的执行顺序，使相邻 block 复用 L2 中的数据。

```
默认调度（row-major）:
  block(0,0) → block(1,0) → block(2,0) → ...
  A 矩阵的列方向复用很差

Swizzle 调度 (GROUP_M=8):
  group 0: block(0,0)→(1,0)→(2,0)→...→(7,0)→(0,1)→...→(7,1)
  group 1: block(8,0)→(9,0)→...
  同组内 block 共享 A 的相同行 → L2 友好
```

**标准性**: ⭐⭐⭐（对大量 block 的大矩阵有帮助）

---

## 5. 优化顺序 & 增量收益

### 推荐的优化阶梯

```
Step 1: SMEM Tiling (v1)
  收益: 2-3× | SMEM: +8 KB | 复杂度: +30%

Step 2: Thread Tiling + float4 (v2)  
  收益: 4-5× | 寄存器: 38→123 | 复杂度: +50%

Step 3: BCF SMEM Layout (v6 feature)
  收益: 1.1-1.15× | SMEM: +padding | 复杂度: +15%

Step 4: BK=16 (v6 feature)
  收益: 1.1-1.15× | SMEM: 2× | 复杂度: 0%

Step 5: Double Buffering (v3)
  收益: 1.2× | SMEM: 2× | 复杂度: +20%

Step 6: cp.async.cg (v4/v6)
  收益: 1.05-1.1× | SMEM: 0 | 复杂度: +10%

Step 7: WMMA Tensor Core (v5)
  收益: 0.9-1.2× vs CUDA Core best | SMEM: 2-3× | 复杂度: +80%
  注意: 参数不对可能反而变慢！

Step 8: MMA PTX + Swizzle
  收益: 1.3-1.5× vs WMMA | 复杂度: +200%
```

### 各版本增量收益估算 (M=N=K=2048, RTX 4090 D)

```
v0 (naive)                 4.7 TFLOPS  ████
v1 (+smem)                 6.2 TFLOPS  ██████          +32%
v2 (+tile+vec)            28.5 TFLOPS  ██████████████████████████████  +360%
v2+BCF(+bcf)              32.0 TFLOPS  ████████████████████████████████████  +12%
v6 (+BK=16+BCF+dbuf)      36.0 TFLOPS  ████████████████████████████████████████  +12%
v3 (+dbuf)                33.5 TFLOPS  (从 v2 基准)
v4 (+async)               33.3 TFLOPS  (BK=8 限制)
v5_new (+WMMA BK=8 s=2)   45.0 TFLOPS  ██████████████████████████████████████████████████  +25% vs v3
cuBLAS TF32               56.4 TFLOPS  ██████████████████████████████████████████████████████████████
```

---

## 6. 常见问题 & 陷阱

### 6.1 Occupancy 不是越高越好

```
高 occupancy (100%): 大量 warp → 隐藏延迟好 → 但每个 warp 寄存器少 → 计算密度低
低 occupancy (33%):  少量 warp → 每个 warp 寄存器多 → 计算密度高 → 但延迟隐藏差

对于 GEMM: 33-50% occupancy 通常是甜区
  - 足够多的 warp 隐藏内存延迟
  - 足够多的寄存器支持 TM×TN 展开
```

### 6.2 BK 太小 vs 太大

```
BK 太小 (如 4): tile 迭代太多 → barrier 开销主导
BK 太大 (如 64): SMEM 太多 → occupancy 过低
BK 最优: CUDA Core ~16, WMMA ~8 (对齐 WK)
```

### 6.3 Stage 数量的甜区

对于 WMMA: `stage=2` 通常最优（LeetCUDA 实测）
对于 MMA PTX: `stage=4-5` 可能更好（无 WARPSYNC 开销）
对于 CUDA Core: `stage=2`（双缓冲即 2-stage）通常足够

### 6.4 B vs A 矩阵的 SMEM 访问优化权重

A 矩阵的计算阶段访问更密集（内层循环沿 K 维度，即 A 的宽度方向），因此：
- A 的 BCF 优化（转置存储）比 B 的 BCF 优化更重要
- B 的访问是跨 warp 的（tx×TN），bank conflict 可以通过增大 BN+OFFSET stride 缓解

---

## 7. 当前实现存在的问题 & 改进清单

### 已识别的缺失优化

| # | 优化项 | 当前状态 | 应做 | 预期收益 | 在哪个版本 |
|---|--------|---------|------|---------|-----------|
| 1 | BK=16 (CUDA Core) | BK=8 only | 添加 BK=16 | +10-15% | v6 |
| 2 | BCF A 转置 | 无 | 转置 SMEM 布局 | +3-5% | v6 |
| 3 | cp.async.cg | .ca in v4 | 改为 .cg | +2-3% | v4, v5, v6 |
| 4 | BK=8 (WMMA) | BK=16 | 改为 BK=8 | +15-20% | v5 |
| 5 | stage=2 (WMMA) | stage=3 | 改为 stage=2 | +10-16% | v5 |
| 6 | SMEM 缩减 (WMMA) | 49.5 KB | 16 KB | +25-40% | v5 |
| 7 | TM×TN 变体 | 仅 8×8 | 添加 8×4, 8×16 | 形状适应性 | v6 |
| 8 | __launch_bounds__ | 256,2 | 去除限制 | 允许更高 occ | v5 |

### 优先级排序

```
P0 (立即): 修复 v5 的参数 (BK=8, stage=2) — 预期 +40%
P1 (短期): 实现 v6 (BK=16 + BCF) — 预期 +10-15% on CUDA Core
P2 (中期): 添加 TM×TN 变体 + cp.async.cg 全局替换
P3 (长期): MMA PTX 实现
```

---

## 8. 总结

### 标准 GEMM 优化的"强制性"台阶

```
必须做:
  1. Shared Memory Tiling        ← 否则 AI 太低，完全访存受限
  2. Thread Tiling (TM×TN ≥ 64)  ← 否则指令开销占比太高
  3. float4 向量化               ← 否则浪费 75% 的内存带宽
  4. FMA intrinsic               ← 否则可能生成 MUL+ADD 而非 FMA
  5. Double Buffering            ← 用 SMEM 换重叠，几乎免费的性能

强烈推荐:
  6. BK 调优 (CUDA: BK=16, WMMA: BK=8)
  7. cp.async.cg                 ← sm_80+ 免费性能
  8. Block Swizzle               ← 大矩阵有帮助

进阶:
  9. BCF SMEM 布局               ← 进一步榨取 5-15%
  10. Multi-stage Pipeline        ← 需要仔细调参
  11. WMMA Tensor Core            ← 接口简单但需正确参数
  12. MMA PTX + XOR Swizzle      ← 接近 cuBLAS 的唯一路径
```
