# 012-CUDA-Triton-Softmax-Kernels

高性能 Softmax Kernel 实现，CUDA C++ 与 Triton 双版本，从 Naive 3-pass 到 Online 算法的渐进式教学。

[博客地址](https://dlog.com.cn/posts/leetgpu05/softmax/)

## 项目结构

```
012-Softmax/
├── README.md                          # 本文件
│
├── triton/                            # Triton / PyTorch 版本
│   ├── softmax.py                     # Triton 实现: fuse, tile, online, online_v3
│   └── softmax_torch.py               # 纯 PyTorch 教学实现 (safe softmax)
│
└── cuda/                              # CUDA C++ 版本 (header-only)
    ├── CMakeLists.txt                 # 构建配置
    ├── reduce.cuh                     # 公共原语: warp/block reduce + online reduce
    ├── softmax_naive.cuh              # Naive 3-pass: 每行单线程串行
    ├── softmax_online.cuh             # Online 算法: warp + block SMem + block uncached
    └── bench/
        └── softmax_bench.cu           # 统一 Benchmark: 正确性 + 性能对比
```

## 算法概览

| 策略 | 算法 | Global Reads | 适用场景 |
|------|------|-------------|---------|
| Naive 3-pass | 串行 max → exp → normalize | 3 reads + 1 write | Baseline, 单线程/行 |
| Warp Online | 寄存器内 max+sum 合并 | 1 read + 1 write | cols < 1024 |
| Block SMem Online | shared memory 缓存，online reduce | 1 read + 1 write | cols ≥ 1024, smem 充足 |
| Block Uncached Online | 2-pass global，online reduce | 2 reads + 1 write | cols ≥ 1024, smem 不足 |

### 核心思想: Online Softmax

传统 softmax 需要 3 次遍历（找 max、算 sum、归一化）。Online 算法将 max-finding 和 sum-accumulation 合并为一次遍历：

```
传统:                            Online:
  Pass 1: find max                 Pass 1: for each x:
  Pass 2: exp(x-max) + sum           m_new = max(m, x)
  Pass 3: normalize                  s = s * exp(m - m_new) + exp(x - m_new)
                                     m = m_new
                                   Pass 2: normalize with final (m, s)
```

全局内存读取从 3 次减少到 2 次（33% 减少）。CACHE_OPT 进一步利用逆序遍历的 L2 temporal locality。

## 快速开始

### Triton 版本

```bash
# Triton softmax（含正确性测试 + 性能 benchmark）
python triton/softmax.py

# PyTorch 教学实现
python triton/softmax_torch.py
```

### CUDA 版本

```bash
cd cuda

# 编译 + 运行 benchmark
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && cmake --build build --target run_softmax

# 或手动运行
./build/softmax_bench
```

输出示例:

```
================================================================
  CUDA Softmax Benchmark — Naive vs Online
  GPU: NVIDIA GeForce RTX 4090 D  |  SMs: 114
================================================================

--- rows=1024    cols=32768   (128.0 MB)  |  30 iters ---
  [Correctness]
    Naive 3-pass                        max rel err = 1.187126e-06
    Online (CACHE_OPT=true)             max rel err = 3.163575e-05
    Online (CACHE_OPT=false)            max rel err = 3.163575e-05
  [Performance]
  Naive 3-pass (baseline)               1959.7417 ms  ( 65.3247 ms/iter)
  Online (CACHE_OPT=true)                  8.8778 ms  (  0.2959 ms/iter)
  Online (CACHE_OPT=false)                 8.8612 ms  (  0.2954 ms/iter)
```

## 设计要点

1. **Online 算法**: 将 max-finding 和 sum-accumulation 合并为一次遍历，使用 rescaling 公式维护 running (m, s) 状态。

2. **Warp Shuffle 通信**: 小 cols 时使用 `__shfl_xor_sync` 在 warp 内完成 reduce，避免 shared memory 开销。

3. **向量化访存**: `Pack<T,N>` + `DirectLoad`/`DirectStore` 实现 128-bit 对齐的向量化加载/存储（LDG.128 / STG.128）。

4. **CACHE_OPT 逆序遍历**: Pass 2 逆序读取数据，使 Pass 1 最后加载的数据仍在 L2 cache 中，提升 L2 命中率。

5. **Occupancy-aware Grid Sizing**: `GetNumBlocks()` 根据 SM 数量和 max threads/SM 动态计算 grid 大小，确保足够的 wave 覆盖来隐藏延迟。

## 文件依赖关系

```
bench/softmax_bench.cu
  ├── softmax_naive.cuh
  └── softmax_online.cuh ── reduce.cuh (warp/block reduce 原语)
```
