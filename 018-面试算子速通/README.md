# LeetGPU 速通

每题一个目录：`solve.cu` 提交，`bench.cu` 调 `bench()`。根目录一份 CMake，子目录不要再放 CMakeLists。

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
