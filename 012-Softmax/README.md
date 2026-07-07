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
│   └── softmax_torch.py               # 纯 PyTorch 教学实现
│
└── cuda/                              # CUDA C++ 版本
    ├── CMakeLists.txt                 # 构建配置
    ├── reduce.cuh                     # 公共原语: warp/block reduce + online reduce
    ├── softmax_naive.cuh              # Naive 3-pass: 每行单线程串行
    ├── softmax_online.cuh             # Online 算法: warp + block SMem + block uncached
    └── bench/
        └── softmax_bench.cu           # 统一 Benchmark: 正确性 + 性能对比
```

### 核心思想: Online Softmax

传统 softmax 需要 3 次遍历。
Online 算法将 max-finding 和 sum-accumulation 合并为一次遍历：

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
python triton/softmax.py

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
