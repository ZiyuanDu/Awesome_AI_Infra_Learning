# 014-CUDA-Triton-Norm-Kernels

高性能 Normalization Kernel 实现：CUDA C++ 与 Triton 双版本，涵盖 RMSNorm、LayerNorm、BatchNorm。

---

## 目录

- [背景：为什么 Norm 是带宽瓶颈](#背景为什么-norm-是带宽瓶颈)
- [项目结构](#项目结构)
- [Triton 版本](#triton-版本)
  - [RMSNorm](#triton-rmsnorm)
  - [LayerNorm](#triton-layernorm)
  - [BatchNorm / LayerNorm (教学版)](#batchnorm--layernorm-教学版)
- [CUDA 版本](#cuda-版本)
  - [模块架构](#模块架构)
  - [reduce.cuh — 规约原语](#reducecuh--规约原语)
  - [io.cuh — 访存抽象层](#iocuh--访存抽象层)
  - [norm_kernel.cuh — 通用 kernel 骨架](#norm_kernelcuh--通用-kernel-骨架)
  - [两种 Kernel 策略](#两种-kernel-策略)
  - [rms_norm.cuh — RMSNorm 实例化](#rms_normcuh--rmsnorm-实例化)
  - [layer_norm.cuh — LayerNorm 实例化](#layer_normcuh--layernorm-实例化)
- [性能数据](#性能数据)
- [快速开始](#快速开始)
- [设计哲学](#设计哲学)

---

## 背景：为什么 Norm 是带宽瓶颈

LayerNorm 和 RMSNorm 是现代 Transformer 的核心组件。以 RMSNorm 为例，它的计算极其简单：

```
y = x * rsqrt(mean(x²) + ε) * gamma
```

每读取一个元素只做两次乘加和一次 rsqrt，**算术密度极低**。这类 kernel 的性能不由算力决定，而由**内存带宽**决定。优化目标是**让内存带宽跑满**。

对于 RTX 4090D（~1008 GB/s），一个 4096×8192 的 FP16 RMSNorm 理论最优耗时为：

```
数据量 = (读x + 读gamma + 写y) = (2×4096×8192 + 8192) × 2 bytes ≈ 134 MB
理论最优 = 134 MB / 1008 GB/s ≈ 0.133 ms
```

---

## 项目结构

```
014-CUDA-Triton-Norm-Kernels/
├── README.md                    # 本文件
│
├── triton/                      # Triton 语言实现 (Python)
│   ├── rms_norm.py              # RMSNorm: kernel + autotuning + 正确性测试 + 性能对比
│   ├── layer_norm.py            # LayerNorm: kernel + autotuning + 正确性测试 + 性能对比
│   ├── batchnorm.py             # BatchNorm: 手写实现 vs PyTorch nn.BatchNorm1d
│   └── layernorm_torch.py       # LayerNorm: 纯 PyTorch 教学实现 vs nn.LayerNorm
│
└── cuda/                        # CUDA C++ 实现 (header-only)
    ├── CMakeLists.txt           # 构建系统 (sm_89, fast_math, L1 cache 优化)
    ├── reduce.cuh               # warp/block 级规约原语
    ├── io.cuh                   # 向量化访存抽象 (Pack, DirectLoad, AffineStore)
    ├── norm_kernel.cuh          # 通用 kernel 骨架 (WarpImpl + BlockSMemImpl)
    ├── rms_norm.cuh             # RMSNorm: Stats 特化 + dispatch + host API
    ├── layer_norm.cuh           # LayerNorm: Stats 特化 + dispatch + host API
    └── bench/
        ├── reduce_bench.cu      # 规约微基准
        ├── io_bench.cu          # 内存带宽微基准 (scalar vs 向量化)
        ├── rms_norm_bench.cu    # RMSNorm 正确性+性能基准
        └── layer_norm_bench.cu  # LayerNorm 正确性+性能基准
```

**依赖关系**（CUDA 版本）：

```
reduce.cuh  ──→  io.cuh  ──→  norm_kernel.cuh  ──→  rms_norm.cuh
                                                  ──→  layer_norm.cuh
```

---

## Triton 版本

### Triton RMSNorm

**文件**：`triton/rms_norm.py`

**算法**：

```
y = x * rsqrt(mean(x²) + ε) * weight
```

每个 program 处理一行数据。为了保持数值精度，加载数据后立即转换为 `float32` 进行计算，
`rsqrt` 操作在 float32 下完成，最后写回时保持原始 dtype。

**核心设计**：

1. **Autotuning**：`@triton.autotune` 自动搜索最优 `num_warps`（1/2/4/8/16/32），
   以 `N`（归一化维度）为 key 进行缓存。

2. **BLOCK_N 对齐**：使用 `triton.next_power_of_2(N)` 确保 block size 是 2 的幂，
   满足 Triton 的 `tl.constexpr` 约束。

3. **无 weight 场景**：当 `weight=None` 时自动创建全 1 向量，等价于纯 RMSNorm（无仿射参数）。

**正确性验证**：与 `torch.nn.RMSNorm` 对比，fp32 下误差 < 1e-4，fp16 下误差 < 1e-2。

**Benchmark**：内置 `do_bench` 手动 benchmark 和 `@perf_report` 自动生成性能曲线图。

---

### Triton LayerNorm

**文件**：`triton/layer_norm.py`

**算法**：

```
mean = sum(x) / N
var = sum((x - mean)²) / N
y = (x - mean) * rsqrt(var + ε) * weight + bias
```

**核心设计**：

1. **输出 mean/rstd**：除了归一化结果 `y`，还输出每行的 `mean` 和 `rstd`（`float32` 精度），
   方便调用方后续使用（如 backward pass 的中间值）。

2. **Multi-port 存储**：一次 kernel launch 中完成数据加载 → 统计量计算 → 归一化 → 写出，
   避免了多次 kernel launch 的开销。

3. **Autotuning**：与 RMSNorm 相同的 autotune 策略。

---

### BatchNorm / LayerNorm 教学版

**`triton/batchnorm.py`** — 手写 BatchNorm1d 的 forward pass，详细展示：

- 训练/推理模式下的不同行为
- `running_mean` / `running_var` 的滑动平均更新
- 有偏方差（用于归一化）vs 无偏方差（用于更新 running_var）的区别
- 与 `torch.nn.BatchNorm1d` 的逐位对比

**`triton/layernorm_torch.py`** — 极简的纯 PyTorch LayerNorm 实现（~15 行代码），
展示 `torch.var_mean` 的用法，适合作为理解 LayerNorm 公式的起点。

---

## CUDA 版本

### 模块架构

CUDA 版本的设计借鉴了 OneFlow 的 `layer_norm.cuh` 架构，核心思想是**四个分离**：

| 维度 | 分离方式 | 实现 |
|------|---------|------|
| 数据搬运 vs 计算 | Load/Store 仿函数 | `io.cuh` — DirectLoad, AffineStore |
| 规约 vs 归一化 | Stats 模板参数 | `norm_kernel.cuh` — `NormWarpImpl<K, Stats>` |
| 策略 vs 算法 | K ≤ 1024 → WarpImpl, K > 1024 → BlockSMemImpl | `norm_kernel.cuh` |
| RMS vs Layer | 不同的 Stats 结构体 | `rms_norm.cuh`, `layer_norm.cuh` |

新增一个 Norm 变体（如 GroupNorm）只需写一个 ~100 行的 `.cuh` 文件，其余代码零改动。

---

### reduce.cuh — 规约原语

**职责**：提供 warp 级和 block 级的求和规约。

**核心接口**：

```cpp
template <typename T>
__device__ T warp_reduce_sum(T val);

template <const int NUM_THREADS, typename T>
__device__ T block_reduce_sum(T val);
```

**Warp 内规约** — 蝶形 shuffle（butterfly pattern）：

```
初始:  lane0  lane1  lane2  lane3  ...  lane31
mask=16: 每个 lane 与相距16的 lane 交换并求和
mask=8:  每个 lane 与相距8的 lane 交换并求和
...
mask=1:  最终每个 lane 持有全部32个值的和
```

**Block 规约** — 两级结构：

```
第一步: 各 warp 内部独立规约 → lane0 写入 shared memory[warp_id]
第二步: warp0 从 shared memory 读取各 warp 结果 → 再做一次 warp 规约
```

注意：`block_reduce_sum` 的返回值**仅在 warp0 中有效**。这是有意设计——避免不必要的 broadcast。

---

### io.cuh — 访存抽象层

**核心类型**：

| 类型 | 作用 |
|------|------|
| `Pack<T, N>` | N 元素对齐数组，`alignas` 确保 128-bit 对齐，触发 `LDG.128` / `STG.128` |
| `DirectLoad<SRC, DST>` | 加载 + 类型转换（如 half→float） |
| `DirectStore<SRC, DST>` | 类型转换 + 存储 |
| `AffineStore<SRC, DST, do_scale, do_center>` | 存储时融合 `gamma*val + beta` |

**关键设计_1 — Pack 与 128-bit 向量化**：

```cpp
template <typename T, int N>
struct alignas(sizeof(T) * N) Pack { T elem[N]; };

// 单条 128-bit LDG 指令加载 8 个 half
Pack<half, 8> pack = *reinterpret_cast<const Pack<half, 8>*>(src + offset);
```

FP16 用 `pack_size=8`（8×2=16 bytes），FP32 用 `pack_size=4`（4×4=16 bytes），
都是 128-bit 对齐，编译器生成最宽的访存指令。

**关键设计_2 — 编译期分支消除**：

```cpp
template <typename SRC, typename DST, bool do_scale, bool do_center>
struct AffineStore { ... };
```

`do_scale` 和 `do_center` 是编译期常量，编译器直接消除死分支：

- `AffineStore<f, h, true, false>` → 生成 `dst[i] = src[i] * gamma[i]`（RMSNorm）
- `AffineStore<f, h, true, true>` → 生成 `dst[i] = src[i] * gamma[i] + beta[i]`（LayerNorm）
- `AffineStore<f, h, false, false>` → 生成 `dst[i] = src[i]`（纯归一化）

**零运行时开销的抽象**。

---

### norm_kernel.cuh — 通用 kernel 骨架

这是整个 CUDA 版本的核心。提供了两种 kernel 策略，通过模板参数 `Stats` 区分 RMSNorm 和 LayerNorm。

**Stats 模板参数需提供**：

```cpp
struct Stats {
    using accum_t;       // 累加器类型 (RMSNorm=float, LayerNorm=float2)
    using stat_t;        // 统计量类型 (RMSNorm=float, LayerNorm=float2)

    static void init(accum_t& a);
    static void accumulate(accum_t& a, const T* vals, int n);
    static accum_t warp_reduce(accum_t a, int group_width);
    template<int BLK> static accum_t block_reduce(accum_t a);
    static stat_t compute(accum_t a, int K, float eps);
    static void normalize(T* vals, stat_t s, int n);
};
```

RMSNorm 和 LayerNorm 共享 80% 的代码（grid/block 组织、循环迭代、load→accumulate→reduce→normalize→store），
只有 20% 不同（统计量公式、归一化公式）。把骨架抽到 `norm_kernel.cuh`，把血肉留给下游。

---

### 两种 Kernel 策略

#### 策略一：WarpImpl（K ≤ 1024）

**适用场景**：小 hidden_size 的 LLM（如 K=256, 512, 1024）。

**问题**：当 K 很小时，如果每个 block 处理一行，block 只有 32 个线程，只占满一个 warp，
无法充分利用 SM 的多 warp 并行能力。

**解决方案**：2D block 组织。

```
blockDim = (32, 4)        ← 32 线程的 warp group，4 个 group 一个 block
gridDim  = (N/4, 1)       ← N 行，每 4 行一个 block

结构示意:
  ┌────────────────────────────────────┐
  │ block (32, 4)                      │
  │  warp group 0 (lane 0..31) → row A │
  │  warp group 1 (lane 0..31) → row B │
  │  warp group 2 (lane 0..31) → row C │
  │  warp group 3 (lane 0..31) → row D │
  └────────────────────────────────────┘
```

每个 warp group 处理一行，group 之间完全独立。统计量只需 warp 内 reduce（`__shfl_xor_sync`），
不需要 `__syncthreads`。一个 SM 同时运行多个 block，大幅提升利用率。

#### 策略二：BlockSMemImpl（K > 1024）

**适用场景**：大 hidden_size（如 K=4096, 8192）。

**问题**：朴素实现需要读 x 两次——第一次算统计量，第二次做归一化。
对于带宽 bound 的 norm 操作，**两次 global read 是最大的性能杀手**。

**解决方案**：Shared memory 缓存。

```
第一次 global 读:
  读 x[col] → 写入 shared memory[col] → 同时累加到 thread-local acc

Block reduce:
  各线程的 acc 规约 → 得到全局统计量

Shared memory 读:
  从 shared memory[col] 读取 → 归一化 → 写回 global y[col]
```

关键数据流：

```
Global Memory (x) ──read──→ Shared Memory (smem) ──read──→ Compute → Global Memory (y)
                               │
                               └──→ Thread-local Accumulator ──→ Block Reduce ──→ stat
```

Shared memory 带宽约 20 TB/s，Global memory 只有 ~1 TB/s。第二次读取从 shared memory 来，几乎免费。

---

### rms_norm.cuh — RMSNorm 实例化

**RMSNormStats**：

```cpp
struct RMSNormStats {
    using accum_t = float;    // 累加一个平方和
    using stat_t  = float;    // 最终统计量: inv_rms

    accumulate:  a += v * v                          // 只累加平方
    compute:     return rsqrtf(a / K + eps)          // inv_rms
    normalize:   vals[i] = vals[i] * inv_rms          // 乘 inv_rms
};
```

**Host API**：

```cpp
void rms_norm_forward(const void* x, void* y, const void* gamma,
                      int N, int K, float eps,
                      bool is_fp16, cudaStream_t stream = 0);
```

`is_fp16=true` 时内部用 `pack_size=8`，`false` 时用 `pack_size=4`，自动选择最优向量化宽度。

---

### layer_norm.cuh — LayerNorm 实例化

**LayerNormStats**：

```cpp
struct LayerNormStats {
    using accum_t = float2;   // (sum, sq_sum) — 需要两个分量
    using stat_t  = float2;   // (mean, inv_std) — 归一化需要两个值

    accumulate:  a.x += v;  a.y += v * v             // 同时累加 sum 和 sq_sum
    compute:     mean = a.x / K
                 var  = a.y / K - mean * mean         // E[x²] - E[x]²
                 return (mean, rsqrt(var + eps))
    normalize:   vals[i] = (vals[i] - mean) * inv_std
};
```

**float2 规约**：CUDA 不原生支持 float2 的 `+=` 和 `__shfl_xor_sync`，
因此在 `layer_norm.cuh` 头部提供了 `f2_add`、`f2_shfl_xor`、`f2_warp_reduce_sum` 三个辅助函数。

**Host API**：

```cpp
void layer_norm_forward(const void* x, void* y,
                        const void* gamma, const void* beta,
                        int N, int K, float eps,
                        bool is_fp16, cudaStream_t stream = 0);
```

---

## 性能数据

测试环境：**RTX 4090D (1008 GB/s)**, CUDA 12.8, **冷缓存**（每次计时前 L2 eviction）。

### RMSNorm (CUDA)

| Shape | FP32 带宽 | FP16 带宽 |
|-------|----------|----------|
| 1024×256 | 302 GB/s | 155 GB/s |
| 1024×1024 | 887 GB/s | 557 GB/s |
| 2048×4096 | 856 GB/s | 839 GB/s |
| **4096×8192** | **911 GB/s** | **915 GB/s** |

### LayerNorm (CUDA)

| Shape | FP32 带宽 | FP16 带宽 |
|-------|----------|----------|
| 1024×256 | 365 GB/s | 190 GB/s |
| 1024×1024 | 927 GB/s | 561 GB/s |
| 2048×4096 | 871 GB/s | 860 GB/s |
| **4096×8192** | **907 GB/s** | **914 GB/s** |

### 分析

- **大 shape (4096×8192) 达到 90%+ 带宽利用率**。BlockSMemImpl 策略下 shared memory 缓存消除了
  第二次 global read，大 K 场景计算量足以摊薄 reduce 同步开销。

- **小 shape (1024×256) 带宽较低**。WarpImpl 下 grid 规模小（仅 1024 个 warp group），
  不足以完全隐藏内存延迟。

- **FP16 在小 shape 下反而慢于 FP32**。因为 FP16 pack_size=8，每个线程只处理 8 个元素，
  指令级并行度更低。大 shape 下 FP16 的带宽优势才能体现。

---

## 快速开始

### Triton 版本

```bash
# RMSNorm — 正确性测试 + 手工 benchmark + 自动性能曲线
python triton/rms_norm.py

# LayerNorm — 正确性测试 + 自动性能曲线
python triton/layer_norm.py

# BatchNorm — 与 PyTorch 对比
python triton/batchnorm.py

# LayerNorm 教学版
python triton/layernorm_torch.py
```

### CUDA 版本

```bash
cd cuda

# 编译 + 一键运行全部 benchmark
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && cmake --build build --target run_all

# 或单独运行
cmake --build build --target run_rms      # 只测 RMSNorm
cmake --build build --target run_layer    # 只测 LayerNorm
cmake --build build --target run_io       # 只测内存带宽上限
cmake --build build --target run_reduce   # 只测 reduce 原语
```

**CMake 编译优化**：

| Flag | 作用 |
|------|------|
| `-arch=sm_89` | Ada Lovelace 原生代码 |
| `-Xptxas -dlcm=ca` | Global load 走 L1 cache |
| `--use_fast_math` | rsqrt→`__frsqrt_rn`, fma 合并 |
| `-O3 -DNDEBUG` | 最高优化级别 |

---

## 设计哲学

1. **模块化**。每个文件只做一件事，依赖关系清晰。新增 Norm 变体只需加一个 ~100 行的文件。

2. **编译期抽象，零运行时开销**。Stats 模板参数、AffineStore 的 bool 参数、pack_size 的选择——
   全部在编译期决议，生成的 PTX 代码与手写特化版本完全一致。

3. **两种策略覆盖全场景**。WarpImpl 解决"小 K 大 N"的 SM 利用率问题，
   BlockSMemImpl 解决"大 K"的重复读取问题。一种策略无法同时覆盖两种场景。

4. **教学友好**。每个模块的开头有中文注释解释"为什么这样设计"。
   CUDA 版本代码量 ~550 行（含注释），可在一个下午读完并理解全部设计。

### Triton vs CUDA 对比

| 维度 | Triton | CUDA |
|------|--------|------|
| 开发效率 | 高（Python DSL，自动 tiling） | 中（手动管理 shared memory, block/grid） |
| Autotuning | 内置 `@triton.autotune` | 需手动编写或使用 CUTLASS |
| 性能调优粒度 | 粗（block size, num_warps） | 细（pack_size, smem bytes, grid sizing） |
| 代码量 | ~50 行/kernel | ~150 行/kernel |
| 可读性 | 极好（接近数学公式） | 较好（模板抽象后清晰） |
| 适用场景 | 快速原型、中小规模部署 | 极致性能、大规模生产环境 |
