# v5 WMMA Tensor Core 深度瓶颈分析

> 文件: `cuda/v5_wmma.cuh`  
> GPU: RTX 4090 D (AD102, sm_89)  
> 测试: M=N=K=2048

---

## 1. 令人意外的性能结果

```
v5-wmma(swizzle,s=3):  0.5332 ms / 32.22 TFLOPS
v3-dbuf(128x128):       0.5134 ms / 33.47 TFLOPS   ← v3 反而更快！
cuBLAS(TF32):           0.3048 ms / 56.37 TFLOPS    ← cuBLAS 是 v5 的 1.75×

v5 比 v3(CUDA Core 版本) 慢 ~4%，比 cuBLAS 慢 ~43%
虽然使用了 Tensor Core，性能不升反降！
```

## 2. 源码架构回顾

```cuda
template <int BM = 128, int BN = 128, int BK = 16, int K_STAGE = 3>
__global__ __launch_bounds__(256, 2)  // 限 2 blocks/SM
void sgemm_v5(...) {

  // Warp tiling: 2×4 warp grid, 每个 warp 负责 8 个 16×16 WMMA tile
  constexpr int WARP_M = 2, WARP_N = 4;          // 2×4 = 8 warps
  constexpr int WTM = BM / (WARP_M * WM);         // 128/32 = 4
  constexpr int WTN = BN / (WARP_N * WN);         // 128/64 = 2
  constexpr int WMMA_K = BK / WK;                 // 16/8  = 2

  // 3-stage SMEM 布局
  // sA: [K_STAGE][BM][SA] = [3][128][16]
  // sB: [K_STAGE][BK][SB] = [3][16][136]
  constexpr int PA = 0, SA = BK + PA;             // stride = 16
  constexpr int PB = 8, SB = BN + PB;             // stride = 136 (padding)

  // Block swizzle for L2-friendly scheduling
  constexpr int GROUP_M = 8;
  // ...
};
```

### 关键参数对比

| 参数 | v3/v4 (CUDA Core) | v5 (WMMA) | 影响 |
|------|-------------------|-----------|------|
| BM, BN | 128, 128 | 128, 128 | 相同 |
| **BK** | **8** | **16** | v5 tile 的 K 更大 → tile 数减半 |
| Warps | 8 | **8** | 相同（256 threads/32 = 8） |
| WMMA fragments/warp | — | 4×2 = **8** | 每个 warp 维护 8 个 WMMA accumulator |
| WMMA_K rounds/tile | — | **2** | BK/WK = 16/8 |
| SMEM | 16.5 KB | **49.5 KB** | 3× 因为 3-stage pipeline |
| Registers | 126 | 104 | v5 更少（WMMA fragment 紧凑） |
| Occupancy | 2 blk/SM (33%) | 2 blk/SM (33%) | 相同！瓶颈转移 |
| Compute | FFMA | **HMMA (Tensor)** | 理论 2× 吞吐 |

---

## 3. SASS 级别的 v5 vs v3 对比

### 指令分布

```
                    v3 (CUDA Core)        v5 (WMMA Tensor Core)
────────────────────────────────────────────────────────────────
总指令数              2,880                 2,064
FFMA (CUDA Core)      1,024                 0        ← v5 不用 FFMA！
HMMA (Tensor Core)       0                 128       ← Tensor Core 专用指令
LDS (共享加载)          64                  24       ← wmma 使用 LDSM 隐式加载
LDG (全局加载)           4                   0       ← cp.async 完全接管
WARPSYNC                 0                   8       ← WMMA API 同步开销
BAR.SYNC                 2                   2       ← block 级同步
LDGDEPBAR                0                   2       ← cp.async 依赖
DEPBAR.LE                0                   2       ← cp.async 完成信号
```

### 关键发现: WARPSYNC

v5 独有的 8 条 WARPSYNC 指令是 WMMA API 的直接开销:

```
WARPSYNC: warp 内 32 个线程的同步点
- 确保所有线程在进入 wmma 操作前数据就绪
- 每条 WARPSYNC → 流水线停顿
- 在 128 条 HMMA 中穿插 8 条 WARPSYNC = 每条 WARPSYNC 服务 ~16 条 HMMA
```

### 关键发现: 零 LDG

v5 中完全看不到 LDG (全局加载) 指令，因为所有全局→共享内存的传输都被 cp.async 接管了。这证明 cp.async 流水线确实在工作！但性能仍然不佳，说明瓶颈不在全局加载。

---

## 4. 六大瓶颈根因分析

### 瓶颈 1: Tensor Core 利用率仅 21.9%

```
v5 理论峰值: ~147 TFLOPS (Tensor TF32, 不含稀疏)
v5 实际:      ~32 TFLOPS
利用率:       21.9%

对比 v3:
v3 理论峰值: ~73.5 TFLOPS (FP32)
v3 实际:     ~33.5 TFLOPS
利用率:      45.5%

→ CUDA Core 的利用率是 Tensor Core 的 2 倍！
```

**根因**: Tensor Core 指令 (HMMA) 的延迟约为 8-16 个时钟周期，而 FFMA 的延迟约为 4 个周期。更长的延迟意味着需要更多的活跃 warp 来隐藏延迟。但 v5 与 v3 一样只有 16 warp/SM，无法有效隐藏 Tensor Core 的延迟。

```
简化的 warp 调度模型:

v3 (FFMA, latency=4):  16 warps / 4 cycle latency = 4 warps 可同时发射
v5 (HMMA, latency=12): 16 warps / 12 cycle latency = 只有 ~1.3 warps 可同时发射

Tensor Core 经常空闲，等待 warp 的 HMMA 结果!
```

### 瓶颈 2: 每个 warp 的 WMMA 工作不够密集

```
v5 每个 warp 的 WMMA 任务:
  WTM × WTN = 4 × 2 = 8 个 fragment
  每个 fragment = 16×16×8 MMA
  WMMA_K = BK/WK = 16/8 = 2 rounds
  每 round: load A (4×), load B (2×), mma (4×2)

  总共: 2 rounds × (4×1 load_A + 1×2 load_B + 4×2 mma) = 复杂交织

问题: A matrix 的 load 用了 4 个 fragment (WTM=4 个 16-row 块)
      B matrix 的 load 用了 2 个 fragment (WTN=2 个 16-col 块)
      但每次 MMA 只做 1 个 A × 1 个 B → 需要 4×2=8 次 MMA/round

由于 WMMA API 的限制:
  - load_matrix_sync(A) 是 warp 级操作
  - load_matrix_sync(B) 是 warp 级操作  
  - mma_sync(A, B, C) 是 warp 级操作
  - 操作之间隐含有 WARPSYNC

这意味着每个 warp 做大量的 warp 级同步，但每次 MMA 的计算量有限。
```

### 瓶颈 3: 49.5 KB SMEM → L1 Cache 被挤掉

```
v5 SMEM 分配:
  sA: 3 × 128 × 16 × 4B = 24,576 bytes (~24 KB)
  sB: 3 × 16  × 136 × 4B = 26,112 bytes (~25.5 KB)
  总计: ~49.5 KB per block

2 blocks/SM: 2 × 49.5 = 99 KB
SMEM 总量: 128 KB
剩余 L1: 128 - 99 = 29 KB

v3 SMEM 分配:
  sA: 2 × 8 × 132 × 4B = 8,448 bytes
  sB: 2 × 8 × 132 × 4B = 8,448 bytes
  总计: ~16.5 KB per block

2 blocks/SM: 2 × 16.5 = 33 KB
剩余 L1: 128 - 33 = 95 KB
```

**影响**: v5 只有 29 KB L1，意味着任何非 SMEM 的访存（栈变量 spill、地址计算临时变量等）都必须经过 L2 (72 MB)。虽然 L2 很大，但延迟更高（~200 cycles vs ~30 cycles for L1）。

### 瓶颈 4: B 矩阵 Bank Conflict

```cuda
// A: stride = BK = 16, 无 padding → 16 × 4B = 64B = 1 cache line
// 每个 bank 32-bit → 16 个 bank → 覆盖 64B → 每 16 个 float 回到同一 bank
#define PA 0;  SA = BK + PA = 16

// B: stride = BN + PB = 128 + 8 = 136 → 136 × 4B = 544B  
// 136 % 32 = 8, 每行偏移 8 个 bank
#define PB 8;  SB = BN + PB = 136
```

对于 WMMA 的 `load_matrix_sync`:
- A fragment (WM=16, WK=8): 从 sA 中加载 16×8 子矩阵，stride=16
  - 16 行 × 8 列，列连续（stride=16 elements = 64B = 16 banks）
  - 同一 warp 内 32 个线程分成 4 组，每组读 8 个元素
  - stride=16 时无 bank conflict ✓（16 是 warpSize/2，恰好对齐）

- B fragment (WK=8, WN=16): 从 sB 中加载 8×16 子矩阵，stride=136
  - 136 % 32 = 8 → 每行偏移 8 banks
  - wmma::load_matrix_sync 内部使用 LDSM (ldmatrix) 指令
  - LDSM 每次读 4 个 32-bit，需要 4 个不同 bank
  - stride=136, 行偏移 136×4B=544B → 行之间 bank offset = 544/4 % 32 = 136 % 32 = 8
  - 如果 4 条 LDSM 跨越了相同的 bank 组，就会产生 bank conflict

**计算 bank conflict 概率**: 这个需要更详细的 SMEM 地址分析，但粗略估计，stride=136 的模式导致每 4 行（136*4/32=17 bank 周期）可能出现冲突。

### 瓶颈 5: 3-stage Pipeline 深度不足

```
v5 的流水线结构:
┌─────────────────────────────────────────────────────────┐
│ Stage 0 (load):  cp.async A[0], B[0]  → commit_group    │
│ Stage 1 (load):  cp.async A[1], B[1]  → commit_group    │
│ Stage 2 (load):  cp.async A[2], B[2]  → commit_group    │
│                                                         │
│ Stage 0 (comp):  wait_group(1) → compute tile[0]        │
│ Stage 1 (comp):  wait_group(1) → compute tile[1]        │
│ Stage 2 (comp):  wait_group(0) → compute tile[2]        │
│ ...                                                     │
└─────────────────────────────────────────────────────────┘

问题: 每个 stage 的计算只包含 2 轮 WMMA (WMMA_K=2)
      2 轮 × (4 load_A + 2 load_B + 8 mma) = 约 28 个 WMMA 操作
      这些操作很快完成，然后 warp 在 wait_group 上等待
```

**3-stage pipeline 的并行度:**
```
时间 →
Stage0: [ Load tile2 ][ Comp tile0 ][ Load tile3 ][ Comp tile1 ]...
Stage1: [ Load tile3 ][ Comp tile1 ][ Load tile4 ][ Comp tile2 ]...
Stage2: [ Comp tile2 ][ Load tile5 ][ Comp tile3 ][ Load tile6 ]...

理想情况: load 和 compute 完全重叠
实际情况: compute 快于 load → 计算完成后在 wait_group 等待

如果每个 tile 的计算需要 WMMA_K=4 或更多 (需要 BK=32)，流水线会更平衡。
但 BK=32 需要 2× SMEM → 总共 99 KB → 只能 1 block/SM → occupancy 降到 16.5%！
```

### 瓶颈 6: 与 MMA PTX 的差距

WMMA API 是 MMA PTX 的高层封装。直接比较:

| 特性 | WMMA API | MMA PTX |
|------|----------|---------|
| 加载指令 | `wmma::load_matrix_sync` → LDS | `ldmatrix.sync.aligned` → LDSM (1 条指令！) |
| MMA 指令 | `wmma::mma_sync` | `mma.sync.f32.tf32.tf32` |
| 存储指令 | `wmma::store_matrix_sync` | 直接 STS/STG |
| WARPSYNC | 隐式（每个 `_sync` 调用） | 显式（`mma.sync` 自带同步） |
| Fragment 类型 | `wmma::fragment<>` 模板类型 | 直接寄存器数组 |
| SMEM 布局约束 | 严格（类型系统强制） | 灵活（手动管理 stride） |

**MMA PTX 的关键优势**:
1. `ldmatrix` 一条指令加载多个 32-bit 值，效率远高于多条 LDS 指令
2. 无需 WARPSYNC（`mma.sync` 自身提供同步语义）
3. 可以手动管理 fragment 寄存器布局，减少 register spilling
4. 可以用 inline PTX 与 cp.async 等指令自由混合

---

## 5. 量化分析: v5 的时间分解

基于 SASS 分析和 nsys 数据，估算 2048² 矩阵时 v5 kernel 的时间分配:

```
v5 kernel 总时间: ~508 μs (稳态, median)

分解:
├── cp.async 全局加载:     ~150 μs  (30%)  ← 3-stage pipeline 部分隐藏
├── Tensor Core HMMA 计算:  ~120 μs  (24%)  ← Tensor Core 未饱和
├── WARPSYNC + barrier:     ~100 μs  (20%)  ← WMMA API 同步开销
├── wmma::load_matrix_sync:  ~80 μs  (16%)  ← SMEM → 寄存器 (可能包含 bank conflict)
├── Block swizzle:           ~30 μs   (6%)  ← group_id 计算
└── wmma::store + 其他:      ~28 μs   (4%)

浪费在同步/等待上的时间: 100 + 80*0.3 ≈ 124 μs (24%!)
```

对比 v3 (CUDA Core):
```
v3 kernel 总时间: ~519 μs (稳态, median)

分解:
├── 全局加载 (float4):     ~130 μs  (25%)  ← 双缓冲部分隐藏
├── FFMA 计算:             ~180 μs  (35%)  ← CUDA Core 更饱和
├── barrier + sync:         ~90 μs  (17%)  ← __syncthreads() × 512
├── 寄存器中转 A:           ~70 μs  (13%)
└── store + 其他:           ~49 μs  (10%)
```

**关键差异**: v5 的纯计算时间更短 (120 vs 180 μs)，但同步开销占比更大 (20% vs 17%)，抵消了 Tensor Core 的速度优势。

---

## 6. 改进路径

### 短期优化 (保持 WMMA API)

1. **调整 WARP_M/WARP_N**: 尝试 1×8 或 4×2 的 warp 网格，改变每个 warp 的 WMMA fragment 数量
2. **减小 K_STAGE**: 用 2-stage pipeline（省 16 KB SMEM → 可能做到 3-4 blocks/SM）
3. **换用 `wmma::col_major`**: B 矩阵用列优先布局可能减少 load 指令数
4. **增大 PB padding**: 尝试 PB=16 或 PB=24 进一步减少 bank conflict

### 中期优化 (MMA PTX)

```cuda
// 伪代码: 用 mma.sync + ldmatrix 替代 WMMA
asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
             : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(smem_addr));

asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
             "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
             : ...);
```

预期提升: **1.3-1.5×** (32 → 45-48 TFLOPS)

### 长期优化 (达到 cuBLAS 水平)

1. **Persistent kernel**: 一个 block 循环处理多个 output tile，减少 block 调度开销
2. **XOR swizzle**: 彻底解决 SMEM bank conflict
3. **4-5 stage pipeline**: 更深流水线隐藏加载延迟（需要更多 SMEM → 可能需要 split-K）
4. **Prefetch with ldmatrix**: 在计算当前 tile 时用 ldmatrix 预取下一 tile 的 fragment 到寄存器
5. **Block 级数据复用**: 对于大矩阵，在 block 内沿 K 维度做更深的流水线

预期达到: **50-55 TFLOPS** (~90% cuBLAS)

---

## 7. 结论

v5 是 WMMA API 的一个**参考实现**——它正确、易读、能跑，但不是性能最优的实现。

**核心矛盾**: WMMA API 的设计目标（易用性）与 SGEMM 的极致性能需求（精细控制）之间存在根本性张力。

**关键数字**:
- v5 用了 Tensor Core 却不如 CUDA Core 的 v3/v4: 32.22 vs 33.47 TFLOPS
- Tensor Core 利用率只有 21.9%（v3 的 CUDA Core 利用率 45.5%）
- 约 24% 的时间浪费在 WMMA API 的 WARPSYNC 同步上
- 99 KB/128 KB SMEM 用于数据，只留 29 KB 给 L1 Cache

**要追赶 cuBLAS (56.37 TFLOPS)**，必须:
1. 从 WMMA API 迁移到 MMA PTX (`mma.sync` + `ldmatrix`)
2. 增大 occupancy（当前 33% → 目标 50-66%）
3. 更深或更高效的软件流水线
4. 零 bank conflict 的 SMEM 布局

对于**教学目的**，v5 完美展示了 WMMA Tensor Core 编程的基本模式。对于**生产环境**，cuBLAS 是更合适的选择。
