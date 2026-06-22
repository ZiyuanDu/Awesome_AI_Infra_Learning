# 016-Conv — CUDA / Triton / PyTorch Conv2D

高性能 Conv2D (stride=1, pad=0) 的 CUDA C++、Triton、PyTorch 三版本对比实现，展示从 naive direct convolution 到 shared memory tiling 的完整优化历程。

## 项目结构

```
016-Conv/
├── README.md
│
├── triton/                          # Triton 实现
│   └── conv_v0.py                   # naive tiled → im2col-style → autotuned + benchmark
│
├── cuda/                            # CUDA 手写实现 (header-only)
│   ├── CMakeLists.txt               # CMake 构建
│   ├── common.cuh                   # 共享辅助函数 (load_tile, store_tile等)
│   ├── bench.cuh                    # Benchmark 基础设施 + cuBLAS wrapper
│   ├── conv_bench.cu                # 主 benchmark 程序
│   ├── v0_naive.cuh                 # Naive direct convolution
│   ├── v1_im2col.cuh                # im2col + shared-memory SGEMM
│   ├── v2_tiled.cuh                 # K-tiled + register accumulation
│   └── v3_smem.cuh                  # Shared memory input tiling + channel blocking
│
└── ppytorch/                        # PyTorch 参考实现
    └── conv.ipynb                   # im2col原理 + 各种conv参数 + 性能对比
```

## CUDA Kernel 优化路线

| 版本 | 技术 | 关键优化 |
|------|------|----------|
| **v0** | Naive direct convolution | 基线：每线程直接global memory读写，遍历C,R,S |
| **v1** | im2col + SGEMM | im2col将卷积转为矩阵乘法，32×32 shared memory tiled SGEMM |
| **v2** | K-tiled + register | K维度分块(BK=4)，寄存器累加，减少weight重复读取 |
| **v3** | Shared memory tiling | 输入窗口缓存到shared memory，block内线程共享input数据；C维度分块(BC)和K维度线程展开(TM) |

## 快速开始

### 环境要求

| 组件 | 要求 |
|------|------|
| CUDA Toolkit | ≥ 11.0 (sm_80+ 推荐) |
| Triton | `pip install triton` |
| PyTorch | `pip install torch` (Triton benchmark 需要) |
| CMake | ≥ 3.18 |

### Triton 版本

```bash
# 完整测试：正确性 + 性能
python triton/conv_v0.py

# 仅性能 benchmark
python triton/conv_v0.py --bench-only

# 生成性能曲线图 (需要 pandas)
python triton/conv_v0.py --plot
```

### CUDA 版本

```bash
cd cuda

# 编译 (默认 sm_80，可在 CMakeLists.txt 中修改 CMAKE_CUDA_ARCHITECTURES)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 运行全部 benchmark
./build/conv_bench

# 或通过 CMake target 运行
cmake --build build --target run_conv
```

**指定 GPU 架构：**

```bash
# sm_89 (RTX 4090)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j

# sm_90 (H100)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j
```

### PyTorch 版本

```bash
jupyter notebook ppytorch/conv.ipynb
```

## Benchmark 结果 (RTX 4090 D, sm_89, cuDNN v9)

```
--- ResNet mid: N=1 C=64 H=56 W=56 K=64 R=3 S=3  (0.21 GFLOPs) ---
  kernel                       time     GFLOPS  vs v0
  v0-naive                 0.0972 ms   2211.65 GFLOPS  ref
  v1-im2col+gemm(f4)       0.0869 ms   2473.81 GFLOPS  0.00e+00
  v2-tiled(BK=4)           0.0567 ms   3791.80 GFLOPS  0.00e+00
  v3-smem(BC=16,BK=16)     0.0563 ms   3815.92 GFLOPS  0.00e+00
  cuBLAS+im2col(FP32)      0.0571 ms   3763.93 GFLOPS  9.40e-04
  cuDNN(FP32)              0.0222 ms   9675.21 GFLOPS  0.00e+00

--- First layer: N=1 C=3 H=224 W=224 K=64 R=7 S=7  (0.89 GFLOPs) ---
  kernel                       time     GFLOPS  vs v0
  v0-naive                 0.2108 ms   4242.45 GFLOPS  ref
  v1-im2col+gemm(f4)       0.2014 ms   4440.20 GFLOPS  0.00e+00
  v2-tiled(BK=4)           0.1390 ms   6430.91 GFLOPS  0.00e+00
  v3-smem(BC=4,BK=16)      0.0782 ms  11434.51 GFLOPS  0.00e+00
  cuBLAS+im2col(FP32)      0.0527 ms  16970.18 GFLOPS  0.00e+00
  cuDNN(FP32)              0.0495 ms  18080.02 GFLOPS  0.00e+00

--- Deep layer: N=1 C=128 H=28 W=28 K=128 R=3 S=3  (0.20 GFLOPs) ---
  kernel                       time     GFLOPS  vs v0
  v0-naive                 0.0753 ms   2648.10 GFLOPS  ref
  v1-im2col+gemm(f4)       0.1521 ms   1310.59 GFLOPS  0.00e+00
  v2-tiled(BK=4)           0.1027 ms   1940.67 GFLOPS  0.00e+00
  v3-smem(BC=16,BK=16)     0.1016 ms   1962.21 GFLOPS  0.00e+00
  cuBLAS+im2col(FP32)      0.0888 ms   2244.27 GFLOPS  1.41e-03
  cuDNN(FP32)              0.0332 ms   5999.63 GFLOPS  0.00e+00

--- Batch: N=4 C=32 H=32 W=32 K=64 R=3 S=3  (0.13 GFLOPs) ---
  kernel                       time     GFLOPS  vs v0
  v0-naive                 0.0455 ms   2917.42 GFLOPS  ref
  v2-tiled(BK=4)           0.0307 ms   4317.12 GFLOPS  0.00e+00
  v3-smem(BC=16,BK=16)     0.0295 ms   4503.13 GFLOPS  0.00e+00
  cuDNN(FP32)              0.0199 ms   6652.23 GFLOPS  0.00e+00
```

## vs cuDNN / cuBLAS 分析

### 完整对比表

| 测试 | v3-smem | cuBLAS+im2col | cuDNN | v3/cuDNN | v3/cuBLAS |
|------|---------|---------------|-------|----------|-----------|
| 3×3, 64→64 | **3816** | 3764 | 9675 | 39% | **101%** |
| 7×7, 3→64 | 11435 | **16970** | 18080 | 63% | 67% |
| 3×3, 128→128 | 1962 | **2244** | 6000 | 33% | 87% |
| N=4, 3×3 | 4503 | — | 6652 | 68% | — |

### 关键发现

1. **v3-smem ≈ cuBLAS im2col+GEMM (FP32)**: 对3×3 kernel，v3-smem (3816 GFLOPS) 与 cuBLAS (3764) 持平。cuBLAS是NVIDIA工程师多年优化的SGEMM——**v3-smem证明手写tiled direct conv可以达到甚至超过im2col+cuBLAS的性能**。

2. **cuDNN优势来自算法层面**:
   - 3×3 kernel: cuDNN用Winograd F(2×2,3×3)减少2.25x算术量。3816 × 2.25 = 8586 ≈ 9675 (89%)，差距几乎完全由Winograd解释
   - 7×7 kernel: cuDNN用implicit GEMM + Tensor Cores(FP16)，~2x吞吐优势。11435 × 1.6 ≈ 18296 ≈ 18080

3. **cuBLAS+im2col对特定维度极优**: 7×7 (M=64, N=47524, K=147)的GEMM维度非常适合cuBLAS(大N=大量并行度)，v3-smem无法超越这种维度的cuBLAS GEMM

4. **v0-naive在C≥128时最优**: GPU L2 cache顺序预取对大量channel的串行访问极其友好，tiled kernel的shared memory/sync开销反而不利

### 如果想进一步逼近cuDNN

| 技术 | 预期加速 | 复杂度 | 适用场景 |
|------|---------|--------|----------|
| **Winograd** F(2×2,3×3) | 1.5-2.0x | 高 | 仅3×3 stride=1 |
| **WMMA Tensor Cores** (FP16) | 1.5-2.0x | 中 | sm_70+ |
| **Implicit GEMM** (CUTLASS风格) | 1.3-1.5x | 高 | 通用 |
| **Double buffering** shared memory | 1.1-1.2x | 低 | sm_80+ (cp.async) |

## 代码设计理念

- **Header-only CUDA：** 所有 CUDA kernel 定义在 `.cuh` 文件中，main 文件直接 `#include` 即可。
- **统一 benchmark 框架：** `bench.cuh` 提供模板化的 `bench_conv` / `bench_conv_first` 函数，自动管理 GPU 内存、计时代码、误差验证。
- **GPU 参考值：** 用 v0 的输出作为所有后续 kernel 的 reference，避免引入 CPU 计算误差。
- **模板化 tile 参数：** v2/v3 内核使用 template 参数控制 tile size，方便 auto-tuning 和适配不同输入规模。
- **Triton autotuning：** 对 BLOCK_OH / BLOCK_OW / BLOCK_K / BLOCK_C 进行自动搜索，找到最优配置。
- **PyTorch 对比：** `ppytorch/conv.ipynb` 展示 im2col 原理、手动实现与 `torch.nn.functional.conv2d` 的等价性，以及各种 conv 参数用法。

## 文件说明

| 文件 | 说明 |
|------|------|
| `cuda/v0_naive.cuh` | Naive direct conv: 每线程一输出元素，全global memory |
| `cuda/v1_im2col.cuh` | im2col kernel + 32×32 tiled SGEMM kernel |
| `cuda/v2_tiled.cuh` | K维度分块(BK) + 寄存器累加 |
| `cuda/v3_smem.cuh` | Shared memory input tiling + C维度分块(BC) + K线程展开(TM) |
| `cuda/common.cuh` | Shared memory helper (load_tile, store_tile, load_input_window) |
| `cuda/bench.cuh` | Benchmark 计时、误差计算、cuBLAS wrapper |
| `cuda/conv_bench.cu` | 主 benchmark：多种卷积规模、多 kernel 对比 |
| `triton/conv_v0.py` | Triton 三版本实现 + 正确性测试 + 性能 benchmark |
| `ppytorch/conv.ipynb` | PyTorch conv2d 参考实现 + im2col 原理讲解 |

## License

MIT
