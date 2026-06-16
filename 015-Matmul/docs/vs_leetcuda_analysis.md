# 当前实现 vs LeetCUDA 对比分析

> LeetCUDA: https://github.com/xlite-dev/LeetCUDA — 200+ CUDA kernel 的教学级实现  
> 对比范围: SGEMM (FP32) 优化全路径: naive → CUDA Core 优化 → WMMA Tensor Core  
> 当前 GPU: RTX 4090 D (sm_89, 114 SM) | LeetCUDA 测试 GPU: L20 (sm_89, 同样 Ada 架构)

---

## 目录

1. [整体优化路线对比](#1-整体优化路线对比)
2. [CUDA Core 路径对比 (v0–v4)](#2-cuda-core-路径对比-v0v4)
3. [WMMA Tensor Core 路径对比 (v5)](#3-wmma-tensor-core-路径对比-v5重点)
4. [v5 的 7 个致命问题](#4-v5-的-7-个致命问题)
5. [修复方案 & 预期性能](#5-修复方案--预期性能)
6. [总结](#6-总结)

---

## 1. 整体优化路线对比

| 优化阶段 | 当前实现 (mycuda) | LeetCUDA | 差异 |
|---------|------------------|----------|------|
| Naive | v0 (16×16 block) | sgemm_naive_f32 (32×32 block) | 相同思路 |
| SMEM Tiling | v1 (32×32×32) | sgemm_sliced_k_f32 (32×32×32) | **完全一致** |
| Thread Tiling + float4 | v2 (128×128×8, TM=TN=8) | sgemm_t_8x8 (128×128×8, TM=TN=8) | **完全一致** |
| Bank Conflict Free | — | sgemm_...bcf (A 转置存, OFFSET=4) | **当前缺失!** |
| Double Buffering | v3 (128×128×8) | sgemm_...dbuf (128×128×8) | 相同思路 |
| BK=16 | — | sgemm_...k16 (128×128×16) | **当前缺失 BK=16!** |
| cp.async | v4 (仅 B 异步) | sgemm_...async (仅 B 异步) | 相同思路 |
| TM×TN 变体 | — | 8×4, 8×8, 8×16 | **当前只有 8×8** |
| WMMA TF32 | v5 (BK=16, stage=3) | sgemm_wmma (BK=8, stage=2/3/4/5) | **参数选择完全不同** |
| MMA PTX | — | HGEMM 中有 | **当前缺失** |

### 性能对比

```
                    当前实现 (RTX 4090 D)    LeetCUDA (L20)
                    ─────────────────────    ──────────────
CUDA Core 最佳:     v3: 33.5 TFLOPS          38.9 TFLOPS (@ M=8K,N=4K)
                    占 FP32 峰值: 45.5%       占 FP32 峰值: 65.0%

WMMA TF32 最佳:     v5: 32.2 TFLOPS          47.0 TFLOPS
                    占 Tensor 峰值: 21.9%      占 Tensor 峰值: 39.3%

cuBLAS TF32:        56.4 TFLOPS              57.9 TFLOPS
                    占 Tensor 峰值: 38.4%      占 Tensor 峰值: 48.4%
```

**关键发现**: 
- CUDA Core 路径: LeetCUDA 比我们高 **65% vs 45%** 峰值利用率
- WMMA TF32 路径: LeetCUDA 比我们高 **39% vs 22%** 峰值利用率 → **1.8× 差距！**
- 两者都远未达到 cuBLAS，但 LeetCUDA 更接近

---

## 2. CUDA Core 路径对比 (v0–v4)

### 2.1 当前实现缺失的关键优化

#### 缺失 1: Bank Conflict Free (BCF) SMEM 布局

**LeetCUDA 的做法** (`sgemm_t_8x8_sliced_k_f32x4_bcf_kernel`):

```cuda
// A 矩阵在 SMEM 中按列优先存储: s_a[BK][BM]
// 而不是直观的行优先 s_a[BM][BK]

// 加载 A 到寄存器（行优先从全局读）
float4 r_load_a[2];
r_load_a[0] = *(float4*)&A[gmem_idx];
r_load_a[1] = *(float4*)&A[gmem_idx + K*4];

// 散布写入 SMEM（列优先）：每个线程写不同 bank
int load_a_smem_m = tid / 2;       // 0..63
int load_a_smem_k = (tid % 2) * 4; // 0 or 4
s_a[load_a_smem_k+0][load_a_smem_m] = r_load_a[0].x;
s_a[load_a_smem_k+1][load_a_smem_m] = r_load_a[0].y;
s_a[load_a_smem_k+2][load_a_smem_m] = r_load_a[0].z;
s_a[load_a_smem_k+3][load_a_smem_m] = r_load_a[0].w;
// ... 第二组
```

**为什么重要？** 在 compute 阶段，A 矩阵沿 K 维度连续读取（`s_a[k][m]`），bank = `k % 32`。相邻线程 (m 不同) 读相同 k → 不同 bank → 无 bank conflict。

当 warps 跨越 `BM/2=64` 边界时（第二个半 warp 组读 `s_a[k][m+64]`），bank = `64/4 + k = 16 + k`，如果 k 相同则会产生 conflict。OFFSET=4 的 padding 解决这个问题。

**当前实现**: v2 和 v3 的 SMEM 布局是 `sA[BM][BK]` (行优先)，没有 BCF 考虑。

**预期提升**: ~15%（LeetCUDA 的数据: +bcf 比基础 tile 版本快 ~15%）

#### 缺失 2: BK=16 的 CUDA Core 变体

**LeetCUDA** (`sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_kernel`):

```cuda
// BK=16, BM=128, BN=128, TM=8, TN=8
// SMEM: s_a[2][16][128], s_b[2][16][128]  → ~32 KB (double buffered)
```

vs 当前 v3: `BK=8, BM=128, BN=128` → SMEM ~16.5 KB

**BK=16 的优势**:
- K 维度 tile 迭代数减半 (2048/8=256 → 2048/16=128)
- Barrier 次数减半 (256×2=512 → 128×1=128 sync)
- 每次 tile 迭代的计算密度翻倍

**代价**: SMEM 加倍 (~16.5 KB → ~32 KB)，但 128 KB/SM 仍可放 3-4 blocks。

**预期提升**: ~5-10%（减少 barrier + 增加计算密度）

#### 缺失 3: TM×TN 变体

LeetCUDA 提供 3 种配置:
```
TM=8, TN=4,  BM=64,  BN=64    → 128 threads, ~16 KB smem
TM=8, TN=8,  BM=128, BN=128   → 256 threads, ~32 KB smem
TM=8, TN=16, BM=128, BN=256   → 256 threads, ~24 KB smem
```

当前只有 TM=8, TN=8 一种。不同形状的矩阵（tall/wide）适合不同配置。

#### 缺失 4: cp.async 使用 `.cg` 而非 `.ca`

**LeetCUDA**: `cp.async.cg` (bypass L1, cache in L2 only)
**当前**: `cp.async.ca` (cache at all levels)

对于流式数据（GEMM 的输入矩阵），`.cg` 更合适: 数据不会被重复使用，没必要污染 L1 cache。节省的 L1 可以给其他数据用。

---

## 3. WMMA Tensor Core 路径对比 (v5) — 重点

### 3.1 配置对比

| 参数 | 当前 v5 | LeetCUDA WMMA | 影响 |
|------|---------|---------------|------|
| BM, BN | 128, 128 | 128, 128 | 相同 |
| **BK** | **16** | **8** | ⚠️ 关键差异 |
| WK | 8 | 8 | 相同 (WMMA 指令限制) |
| **WMMA_K** | **2** (BK/WK=16/8) | **1** (BK/WK=8/8) | 每 tile 的 WMMA 轮数 |
| **K_STAGE** | **3** | **2** (最佳) | ⚠️ 关键差异 |
| WARP_M × WARP_N | 2×4 | 2×4 | 相同 |
| WTM × WTN | 4×2 | 4×2 | 相同 |
| **A_PAD** | **0** | **0** | 相同 |
| **B_PAD** | **8** | **0/4/8** | 当前浪费 SMEM |
| Threads | 256 | 256 | 相同 |
| Registers | 104 | — | — |
| **SMEM** | **~49.5 KB** | **~16 KB** (s=2) / **~24 KB** (s=3) | ⚠️ **3× 差距!** |
| cp.async 类型 | `.ca` | `.cg` | L1 vs L2 cache |
| Block Swizzle | ✅ GROUP_M=8 | ✅ BLOCK_SWIZZLE | 相同思路 |
| FP32→TF32 convert | ❌ | ✅ 单独 kernel | 可能影响精度/性能 |
| SWIZZLE stride | 固定 | 可选 512/1024/2048 | LeetCUDA 更灵活 |

### 3.2 SMEM 用量是核心问题

```
当前 v5 (BK=16, stage=3):
  sA: 3 × 128 × 16 × 4B = 24,576 B
  sB: 3 × 16 × 136 × 4B = 26,112 B
  总计: 50,688 B ≈ 49.5 KB

  2 blocks/SM → 99 KB / 128 KB → 仅剩 29 KB L1
  只能 2 blocks/SM → 512 threads/SM → 33% occupancy

LeetCUDA (BK=8, stage=2):
  sA: 2 × 128 × 8 × 4B = 8,192 B
  sB: 2 × 8 × 128 × 4B = 8,192 B
  总计: 16,384 B ≈ 16 KB

  理论上 128/16 = 8 blocks/SM!
  (实际受寄存器限制，但至少 3-4 blocks/SM → 768-1024 threads/SM → 50-66% occupancy)
```

**SMEM 用量的连锁反应**:

```
SMEM 大 → blocks/SM 少 → warp 少 → 无法隐藏 Tensor Core 延迟
                                  → Tensor Core 利用率低
                                  → 性能差
```

### 3.3 Stage 数量的教训

**LeetCUDA 的性能数据**:
> "2-stage pipeline consistently outperforms 3-stage: stage2 variants are typically 10-16% faster than stage3 equivalents"

**我们的 v5 默认 stage=3**，与 LeetCUDA 的最佳实践背道而驰！

为什么 stage=2 更好？
- Stage=3 需要 50% 更多 SMEM → 降低 occupancy
- 3 级流水线的计算/加载重叠收益不足以补偿 SMEM/occupancy 损失
- Tensor Core 指令延迟不够大，3 级预取深度不必要

### 3.4 BK=16 vs BK=8 的权衡

**BK=16 的优势**:
- K tile 迭代数减半 (128 vs 256)
- Barrier 次数减少

**BK=16 的劣势（WMMA 路径）**:
- sA 大小 2× (128×16 vs 128×8)
- sB 大小 2× (16×136 vs 8×128)
- 结合 3-stage → SMEM 爆炸 (49.5 KB vs 16 KB!)
- 每个 tile 需要 2 轮 WMMA (WMMA_K=2)，增加指令数

**对于 WMMA，BK=8 是最优选择**，因为:
- WK=8 是 WMMA 指令的 K 维度
- BK=8 意味着每个 tile 正好 1 轮 WMMA，指令开销最小
- SMEM 占用最小

**对于 CUDA Core，BK=16 是更好的选择**（LeetCUDA 也是这么做的）:
- FFMA 没有固定的 K 维度限制
- 更大的 BK → 更少的 barrier，更高计算密度
- SMEM 成本可接受

---

## 4. v5 的 7 个致命问题

### 问题 1: BK=16 + Stage=3 → SMEM 爆炸 (49.5 KB)

**严重程度**: 🔴 致命

```
LeetCUDA stage=2 BK=8: 16 KB SMEM → 可达 4-8 blocks/SM
当前 v5 stage=3 BK=16: 49.5 KB SMEM → 强制 2 blocks/SM

后果: SMEM 用量的 3× 差异 → Occupancy 差了 2-4×
```

### 问题 2: 默认使用 3-stage pipeline (应该用 2-stage)

**严重程度**: 🔴 致命

LeetCUDA 明确指出 stage=2 比 stage=3 快 10-16%。我们的 v5 默认写死 stage=3。

### 问题 3: 缺少 FP32→TF32 转换

**严重程度**: 🟡 中等

LeetCUDA 在 WMMA GEMM 前单独调用 `f32x4_tf32x4_kernel` 将输入转为 TF32:

```cuda
// 每个线程处理 4 个 float
float4 val = *(float4*)&input[idx];
val.x = wmma::__float_to_tf32(val.x);
val.y = wmma::__float_to_tf32(val.y);
val.z = wmma::__float_to_tf32(val.z);
val.w = wmma::__float_to_tf32(val.w);
*(float4*)&output[idx] = val;
```

我们的 v5 在 `load_matrix_sync` 时使用 `wmma::precision::tf32`，依赖硬件隐式转换。但显式预转换可能更高效（避免在 Tensor Core 指令的关键路径上做转换）。

### 问题 4: 使用 cp.async.ca 而非 cp.async.cg

**严重程度**: 🟡 中等

```
cp.async.ca: 数据经过 L1 + L2 cache → 污染 L1（流数据不需要 L1）
cp.async.cg: 数据只经过 L2 cache → 保留 L1 给其他数据
```

对于 GEMM 的 A/B 矩阵（每个元素只读一次），`.cg` 是正确的选择。

### 问题 5: B_PAD=8 浪费 SMEM

**严重程度**: 🟠 次要

```
当前: PB=8, SB=BN+PB=136 → sB 每行浪费 8×4=32 bytes
      对于 stage=3: 3×16×136×4 = 26112 bytes
      如果用 PB=0:    3×16×128×4 = 24576 bytes → 节省 1536 bytes

节省虽小，但结合其他优化累积起来就重要了
```

LeetCUDA 默认 B_PAD=0，可选 B_PAD=4 或 B_PAD=8，根据需要添加。

### 问题 6: __launch_bounds__(256, 2) 强制低 occupancy

**严重程度**: 🔴 致命

```cuda
__launch_bounds__(256, 2)  // 强制 max 2 blocks/SM!
```

这主动放弃了更高 occupancy 的可能性。虽然当前因为 SMEM 大也只能 2 blocks，但如果修复 SMEM 后，这个限制就成了人为瓶颈。

### 问题 7: WMMA fragment 加载不够高效

**严重程度**: 🟡 中等

```cuda
// 当前 v5: 每次 load_matrix_sync 加载一个 fragment
for (int i = 0; i < WTM; ++i)         // 4 个 A fragment
  wmma::load_matrix_sync(a_frag[i], ...);
for (int j = 0; j < WTN; ++j)         // 2 个 B fragment
  wmma::load_matrix_sync(b_frag, ...);
```

WMMA API 的 `load_matrix_sync` 每次调用都有隐式 WARPSYNC。4+2=6 次 load × 每轮 2 WMMA_K × 所有 tile = 大量 WARPSYNC 开销。

这就是 SASS 中 8 条 WARPSYNC 的来源。用 MMA PTX 的 `ldmatrix` 替代可以大幅减少。

---

## 5. 修复方案 & 预期性能

### 短期修复 (WMMA API 层面，不改架构)

```
修改 1: BK=16 → BK=8
修改 2: K_STAGE=3 → K_STAGE=2
修改 3: B_PAD=8 → B_PAD=0 (或 4)
修改 4: cp.async.ca → cp.async.cg
修改 5: 去掉 __launch_bounds__(256, 2)
修改 6: 添加 BLOCK_SWIZZLE=2/4 可配置
```

**SMEM 变化**:
```
修复前: 3 × 128 × 16 + 3 × 16 × 136 = 49.5 KB → 2 blk/SM
修复后: 2 × 128 × 8  + 2 × 8  × 128 = 16 KB   → 4-8 blk/SM

Occupancy: 33% → 50-66% (受寄存器限制)
```

**预期性能**: 32 TFLOPS → **42-48 TFLOPS** (+31% ~ +50%)

这应该能追平或超过 v3/v4 的 CUDA Core 性能，真正发挥 Tensor Core 的优势。

### 中期修复 (添加 CUDA Core 缺失的优化)

```
修改 7: v3/v4 添加 BK=16 变体 (CUDA Core 路径)
修改 8: v3/v4 添加 BCF (bank conflict free) SMEM 布局
修改 9: 添加 TM=8xv, TN=8x 多种配置 (8×4, 8×8, 8×16)
```

**预期 CUDA Core 最佳**: 33.5 TFLOPS → **38-42 TFLOPS** (+13% ~ +25%)

### 长期修复 (MMA PTX)

```
修改 10: 用 mma.sync.f32.tf32.tf32 替代 wmma::mma_sync
修改 11: 用 ldmatrix 替代 wmma::load_matrix_sync
修改 12: 手动 SMEM swizzle (XOR-based)
修改 13: 4-5 stage 深流水线 (MMA PTX 下 stage 多才有益)
修改 14: Reg double buffering (fragment 预取)
```

**预期 MMA PTX 最佳**: **50-55 TFLOPS** (~90% cuBLAS)

### 完整优化路线图

```
当前 v5:                32 TFLOPS  (22% Tensor 峰值)
  ↓ BK=8, stage=2, cp.async.cg
第一阶段:               42-48 TFLOPS (29-33% Tensor 峰值)
  ↓ BCF + BK=16 CUDA Core
第二阶段:               CUDA Core 38-42 TFLOPS / WMMA ~48 TFLOPS
  ↓ MMA PTX + ldmatrix + swizzle
第三阶段:               50-55 TFLOPS (34-37% Tensor 峰值)
                        目标: ~90% cuBLAS
```

---

## 6. 总结

### 当前实现的核心问题

| # | 问题 | 当前 | 应为 | 影响 |
|---|------|------|------|------|
| 1 | BK (WMMA) | **16** | **8** | SMEM 2× |
| 2 | K_STAGE | **3** | **2** | SMEM 1.5×, Occ ↓ |
| 3 | SMEM 总量 | **49.5 KB** | **16 KB** | Occupancy 差距 2-4× |
| 4 | launch_bounds | **2** | 不限制 | 人为限制 Occ |
| 5 | cp.async 类型 | **.ca** | **.cg** | L1 污染 |
| 6 | FP32→TF32 | **无** | 单独 kernel | 转换开销在关键路径 |
| 7 | CUDA Core BCF | **无** | A 列优先 | ~15% 性能损失 |
| 8 | CUDA Core BK=16 | **无** | 添加 | Barrier 减半 |
| 9 | CUDA Core TM×TN | **仅 8×8** | 8×4, 8×16 | 形状适应性差 |

### 一句话总结

**当前 v5 最大的问题是: BK=16 + 3-stage pipeline 导致 49.5 KB SMEM，强制 2 blocks/SM (33% occupancy)，Tensor Core 利用率仅 22%。而 LeetCUDA 用 BK=8 + 2-stage = 16 KB SMEM，可达 4+ blocks/SM (66%+ occupancy)，Tensor Core 利用率 39%。**

**修复第一步（BK=8 + stage=2）即可获得 ~40% 性能提升，从 32 → 45 TFLOPS。**

### LeetCUDA 做得更好的地方

1. **参数空间的探索**: 提供 stage=2/3/4/5 多个变体，用 benchmark 选出最佳
2. **阶段性优化**: 每个优化都有独立的 kernel，可以精确衡量每个技术的贡献
3. **SMEM 预算意识**: 对每个 kernel 的 SMEM 用量有精确计算和注释
4. **实践驱动的选择**: stage=2 优于 stage=3 的结论来自数据而非直觉
5. **CUDA Core 路径更完整**: BCF + BK=16 + async 三个额外台阶

### 当前实现做得好的地方

1. **代码更紧凑**: header-only .cuh 设计，无外部依赖
2. **benchmark 框架**: bench.cuh 统一计时+验证，方便对比
3. **Block swizzle**: v5 的 GROUP_M swizzle 实现正确
4. **3-stage pipeline 代码**: 虽然参数选择不佳，但代码逻辑是正确的多级流水线实现
5. **文档详尽**: README + 性能分析 + 工具指南，比 LeetCUDA 的文档好
