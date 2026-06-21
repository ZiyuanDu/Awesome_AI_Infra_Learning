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

## Benchmark 结果示例

运行 `conv_bench` 后，典型的输出格式如下（数值随 GPU 而异）：

```
--- Conv N=1 C=64 H=56 W=56 K=64 R=3 S=3  (3.7 GFLOPs) ---
  kernel                       time     GFLOPS  vs ref
  v0-naive                23.4567 ms    1.57 GFLOPS  ref
  v1-im2col+sgemm          5.1234 ms    7.20 GFLOPS  1.23e-07
  v2-tiled                 3.4567 ms   10.67 GFLOPS  1.45e-07
  v3-smem                  2.3456 ms   15.72 GFLOPS  2.10e-07

--- Conv N=1 C=3 H=224 W=224 K=64 R=7 S=7  (5.0 GFLOPs) ---
  kernel                       time     GFLOPS  vs ref
  v0-naive                45.6789 ms    1.10 GFLOPS  ref
  v1-im2col+sgemm          8.9012 ms    5.62 GFLOPS  1.23e-07
  v2-tiled                 6.7890 ms    7.37 GFLOPS  1.45e-07
  v3-smem                  5.4321 ms    9.21 GFLOPS  2.10e-07
```

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
