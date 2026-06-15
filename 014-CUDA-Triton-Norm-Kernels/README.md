# 014-CUDA-Triton-Norm-Kernels

高性能 Normalization Kernel 实现：CUDA C++ 与 Triton 双版本，涵盖 RMSNorm、LayerNorm、BatchNorm。

[博客地址](https://dlog.com.cn/posts/leetgpu07/norm/)
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


