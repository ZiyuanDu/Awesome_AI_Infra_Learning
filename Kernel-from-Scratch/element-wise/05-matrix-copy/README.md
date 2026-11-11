# 矩阵拷贝

**LeetGPU：** Easy · [Matrix Copy](https://leetgpu.com/challenges/matrix-copy)  

把行优先的 \(N\times N\) 矩阵 \(A\) 拷到 \(B\)：

$$
B[i,j]=A[i,j]
$$

存储是连续的，下标不变，等价于拷 `N*N` 个 float。例：`[[1,2],[3,4]]` 原样到 \(B\)。

## 接口

```cuda
void solve(const float* A, float* B, int N);
```

所有指针参数均指向 **GPU 设备内存**；数组由调用方分配，尺寸参数和其他标量按值传入。

| 参数 | 含义 |
|---|---|
| `A` | 行优先的 N × N 只读 float 矩阵。 |
| `B` | 行优先的 N × N float 输出矩阵，保存 A 的拷贝。 |
| `N` | 方阵的边长；数组元素总数为 N × N。 |

`1 ≤ N ≤ 4096`，计时 `N=4096`（64MB×2）。
