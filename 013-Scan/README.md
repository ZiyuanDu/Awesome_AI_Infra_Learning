# 013-CUDA-Triton-Scan-Kernels

高性能 Parallel Scan Kernel 实现，CUDA C++ 与 Triton 版本。

[博客地址](https://dlog.com.cn/posts/leetgpu06/scan/)

## 项目结构

```
013-Scan/
├── README.md                       # 本文件
│
├── triton/                         # Triton 版本
│   └── scan.py                     # 三趟法 Scan: kernel + 正确性测试 + 性能对比
│
└── cuda/                           # CUDA C++ 版本 (header-only)
    ├── CMakeLists.txt              # 构建配置
    ├── reduce.cuh                  # 公共原语: warp scan + bank-conflict padding
    ├── scan_v0_naive.cuh           # V0: Naive Hillis-Steele (O(N log N) 工作)
    ├── scan_v1_blelloch.cuh        # V1: Blelloch 单 block (O(N) 工作，≤2048 元素)
    ├── scan_v2_multiblock.cuh      # V2: 多 block 递归 Blelloch (任意 N)
    ├── scan_opt.cuh                # V3: 全优化版 (warp/block/device 三级)
    └── bench/
        └── scan_bench.cu           # 统一 Benchmark: 正确性 + 性能对比
```



## 快速开始

### Triton 版本

```bash
# 运行 Scan kernel
python triton/scan.py
```

### CUDA 版本

```bash
cd cuda

# 编译 + 运行 benchmark
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && cmake --build build --target run_scan

# 或手动运行
./build/scan_bench
```

输出示例:

```
================================================================
  CUDA Scan — From Naive to Optimized
  GPU: NVIDIA GeForce RTX 4090D  |  SMs: 112
================================================================

--- N = 1024     (  4 KB)  |  100 iters ---
  v0-naive        0.0032 ms  |   769.23 GB/s  |  PASS
  v1-blelloch     0.0021 ms  |  1176.47 GB/s  |  PASS
  opt-block       0.0018 ms  |  1365.33 GB/s  |  PASS
  v2-multi        0.0045 ms  |   546.13 GB/s  |  PASS
  opt-device      0.0031 ms  |   793.55 GB/s  |  PASS

--- N = 16777216  ( 64 MB)  |  20 iters ---
  v2-multi        2.3456 ms  |    57.12 GB/s  |  PASS
  opt-device      1.1234 ms  |   119.45 GB/s  |  PASS
```

## 设计要点

1. **Bank Conflict Padding**: Blelloch 算法中 shared memory 访问 stride 为 2 的幂次，使用 `pad(i) = i + (i >> 5)` 插入偏移，消除 32-bank 冲突。

2. **Warp Shuffle**: Opt 版本使用 `__shfl_up_sync` 在 warp 内完成 O(log 32) 步 scan，无需 shared memory，延迟极低。

3. **向量化访存**: Opt Device 版本每线程处理 8 个元素 (`OPT_ITEMS=8`)，将全局内存访问次数减少 8 倍，更好地隐藏访存延迟。

4. **递归分解**: 多 block 版本采用经典的 "tile scan → recursive on aggregates → add back" 三阶段策略，将大问题递归分解为可并行的子问题。

## 文件依赖关系

```
bench/scan_bench.cu
  ├── scan_v0_naive.cuh
  ├── scan_v1_blelloch.cuh  ──┐
  ├── scan_v2_multiblock.cuh ──┤── reduce.cuh  (公共原语)
  └── scan_opt.cuh ───────────┘
```
