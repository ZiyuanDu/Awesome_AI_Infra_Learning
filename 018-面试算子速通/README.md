# 面试算子速通

核心代码都在`solve.cu`中，这些代码都基本在100行以内，并且保证其核心性能。

```
include/bench.cuh      CPU / GPU 回调 + 数据，对答案；bytes>0 再计时
include/common.cuh     device 原语
01_vector_add/
02_matrix_transpose/
03_matrix_mul/
题目/
```

新题：建目录，根 `CMakeLists.txt` 加一行 `leetgpu_bench(03_xxx)`。

```bash
cmake -B build -S .
cmake --build build
./build/01_vector_add/bench
./build/02_matrix_transpose/bench
./build/03_matrix_mul/bench
```
