# 015-Matmul — CUDA / Triton / CUTLASS SGEMM

高性能 SGEMM (Single-Precision General Matrix Multiply) 的 CUDA C++、Triton、CUTLASS 三版本对比实现，展示从 naive global memory 到 Tensor Core 的完整优化历程。

## 项目结构

```
015-Matmul/
├── README.md
│
├── triton/                          # Triton 实现
│   └── matmul.py                    # naive → tiled → autotuned + benchmark
│
├── cuda/                            # CUDA 手写实现 (header-only)
│   ├── CMakeLists.txt               # CMake 构建
│   ├── matmul_kernels.cuh           # SGEMM v0→v5 全部内核
│   ├── bench.cuh                    # Benchmark 基础设施 + cuBLAS wrapper
│   └── matmul_bench.cu              # 主 benchmark 程序
│
└── cutlass/                         # CUTLASS 对比实现
    ├── CMakeLists.txt               # CUTLASS 构建 (需 CUTLASS 源码)
    └── cutlass_sgemm.cu             # SIMT + TensorOp 两种配置
```

## CUDA Kernel 优化路线

| 版本 | 技术 | 关键优化 |
|------|------|----------|
| **v0** | Naive global memory | 基线：每线程直接读写全局内存 |
| **v1** | Shared memory tiling | 32×32 共享内存分块，减少全局内存访问 |
| **v2** | 1D block tiling + float4 | 128×128×8 分块，256 线程协作搬运，向量化读写 |
| **v3** | Double buffering | 双缓冲 SMEM + warp tiling，预取与计算流水线重叠 |
| **v4** | cp.async (sm_80+) | B 矩阵异步拷贝，与计算并行执行 |
| **v5** | WMMA Tensor Core | 3-stage cp.async 流水线 + TF32 MMA 指令 |

## 快速开始

### 环境要求

| 组件 | 要求 |
|------|------|
| CUDA Toolkit | ≥ 11.0 (sm_80+ 推荐) |
| Triton | `pip install triton` |
| PyTorch | `pip install torch` (Triton benchmark 需要) |
| CUTLASS | `git clone https://github.com/NVIDIA/cutlass.git` (仅 cutlass/ 需要) |
| CMake | ≥ 3.18 |

### Triton 版本

```bash
# 完整测试：正确性 + 性能
python triton/matmul.py

# 仅性能 benchmark
python triton/matmul.py --bench-only

# 生成性能曲线图 (需要 pandas)
python triton/matmul.py --plot
```

### CUDA 版本

```bash
cd cuda

# 编译 (默认 sm_80，可在 CMakeLists.txt 中修改 CMAKE_CUDA_ARCHITECTURES)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 运行全部 benchmark
./build/matmul_bench

# 或通过 CMake target 运行
cmake --build build --target run_all
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

### CUTLASS 版本

```bash
# 1. 克隆 CUTLASS
git clone https://github.com/NVIDIA/cutlass.git

# 2. 编译
cd cutlass
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCUTLASS_DIR=/path/to/cutlass
cmake --build build -j

# 3. 运行
./build/cutlass_sgemm
```

## Benchmark 结果示例

运行 `matmul_bench` 后，典型的输出格式如下（数值随 GPU 而异）：

```
--- SGEMM M=N=K=2048  (17.2 GFLOPs) ---
  kernel                    time       TFLOPS    vs ref
  v0-naive              12.3456 ms    0.57 TFLOPS  ref
  v1-smem                2.1234 ms    3.30 TFLOPS  1.23e-07
  v2-tile                0.8765 ms    8.01 TFLOPS  1.45e-07
  v3-dbuf                0.6543 ms   10.73 TFLOPS  2.10e-07
  v4-async               0.5432 ms   12.92 TFLOPS  2.50e-07
  v5-wmma(s=3)           0.3210 ms   21.86 TFLOPS  5.00e-05  ← TF32
  v5-wmma(bk16)          0.2987 ms   23.49 TFLOPS  6.00e-05  ← TF32
  cuBLAS(TF32)           0.2756 ms   25.46 TFLOPS  5.00e-05  ← TF32 baseline
```

**注意：** v5 WMMA 和 cuBLAS 使用 TF32 精度（Ampere+ Tensor Core），速度显著快于 FP32 版本，但误差约为 5e-5 量级（在 TF32 精度范围内）。

## 代码设计理念

- **Header-only CUDA：** 所有 CUDA kernel 定义在 `.cuh` 文件中，无编译依赖，main 文件直接 `#include` 即可。
- **统一 benchmark 框架：** `bench.cuh` 提供模板化的 `bench_matmul` 函数，自动管理 GPU 内存、计时代码、误差验证。
- **GPU 参考值：** 用 v0 的输出作为所有后续 kernel 的 reference，避免引入 CPU 计算误差。
- **自适应 dispatch：** `sgemm_dispatch` 根据矩阵形状（tall/wide/square）和 GPU 能力自动选择最优 kernel。
- **Triton autotuning：** 对 block size / num_warps / num_stages 进行自动搜索，找到最优配置。
- **CUTLASS 对比：** 展示 SIMT 和 TensorOp 两种 OpClass，作为手写 kernel 的性能上界参考。

## 文件说明

| 文件 | 说明 |
|------|------|
| `cuda/matmul_kernels.cuh` | 所有 SGEMM 内核 (v0–v5) + 自适应 dispatch |
| `cuda/bench.cuh` | Benchmark 计时、误差计算、cuBLAS wrapper |
| `cuda/matmul_bench.cu` | 主 benchmark：多矩阵形状、多 kernel 对比 |
| `triton/matmul.py` | Triton 三版本实现 + 正确性测试 + 性能 benchmark |
| `cutlass/cutlass_sgemm.cu` | CUTLASS SIMT + TensorOp GEMM wrapper |
| `cutlass/CMakeLists.txt` | CUTLASS 独立构建脚本 |

## License

MIT
