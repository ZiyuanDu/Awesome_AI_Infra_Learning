# CUDA Warp Shuffle 简单教程


Warp Shuffle 指令允许同一 Warp 内的线程（Lane）直接交换寄存器数据，无需经过共享内存，亦无需线程同步 `__syncthreads()`。

从 Volta 架构开始，使用Warp Shuffle需要使用带 `_sync` 后缀的版本，并需指定 `mask`（通常设为 `0xffffffff` 表示 Warp 内所有 32 个线程均参与）。

以下以 8 线程为例说明数据流转，32 线程场景逻辑相同。初始状态下，各线程寄存器值为其 Lane ID：$v_i = i$。

## 1. 蝶形规约
蝶形归约`__shfl_xor_sync`是实现 Warp 级全归约的标准范式。通过按位异或操作 `xor` 配对线程，数据在对数步数内完成全局聚合。
以下代码展示 8 线程宽度下的累加过程，初始状态各线程持有其 Lane ID。经过偏移量为 4、2、1 的三次异或交换与累加，所有线程最终均获得完整的总和 28。

这种全员持有完整结果的特性，使其成为 Softmax 行归约、RMSNorm 以及 FlashAttention 中 Warp 内合并的首选方案。

```cpp
// 初始状态: v = [0, 1, 2, 3, 4, 5, 6, 7]
// 目标: warp reduce sum 

// 第一步: offset = 4, 线程 i 与 i ^ 4 交换并累加
// 配对: (0,4), (1,5), (2,6), (3,7)
// 结果: v = [4, 6, 8, 10, 4, 6, 8, 10]
v += __shfl_xor_sync(0xffffffff, v, 4, 8);

// 第二步: offset = 2, 线程 i 与 i ^ 2 交换并累加
// 配对: (0,2), (1,3), (4,6), (5,7)
// 结果: v = [12, 16, 12, 16, 12, 16, 12, 16]
v += __shfl_xor_sync(0xffffffff, v, 2, 8);

// 第三步: offset = 1, 线程 i 与 i ^ 1 交换并累加
// 配对: (0,1), (2,3), (4,5), (6,7)
// 结果: v = [28, 28, 28, 28, 28, 28, 28, 28]
v += __shfl_xor_sync(0xffffffff, v, 1, 8);
```

完整的实现 `Warp Reduce Sum` 为：
```cpp
__device__ __forceinline__ float warpReduceSum(float v) {

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, off);
    }
    return v;
}
```


## 2. 向下规约

向下归约`__shfl_down_sync`通过指定正向偏移量，使低索引线程获取高索引线程的数据。超出当前宽度的线程其值保持不变。在相同的累加逻辑下，经过 4、2、1 的偏移操作，最终仅 Lane 0 获得完整总和 28，其余线程保留部分聚合结果。

该模式适用于 Block 级归约的最后阶段，即仅需单一代表线程执行原子操作或写入全局内存的场景。


```cpp
// 初始状态: v = [0, 1, 2, 3, 4, 5, 6, 7]
// 目标: 仅 lane 0 得到 warp reduce sum

// 第一步: delta = 4, 线程 i 读取 i+4 并累加；越界则不变
// 流向: 4→0, 5→1, 6→2, 7→3
// 结果: v = [4, 6, 8, 10, 4, 5, 6, 7]
{
    float t = __shfl_down_sync(0xffffffff, v, 4, 8);
    if (lane + 4 < 8) v += t;
}

// 第二步: delta = 2, 流向: 2→0, 3→1, 6→4, 7→5
// 结果: v = [12, 16, 14, 17, 10, 12, 6, 7]
{
    float t = __shfl_down_sync(0xffffffff, v, 2, 8);
    if (lane + 2 < 8) v += t;
}

// 第三步: delta = 1, 流向: 1→0, 3→2, 5→4, 7→6
// 结果: v = [28, 16, 31, 17, 22, 12, 13, 7]
{
    float t = __shfl_down_sync(0xffffffff, v, 1, 8);
    if (lane + 1 < 8) v += t;
}
```

完整实现里常省略 `if`，生产代码只保证 Lane 0：

```cpp
__device__ __forceinline__ float warpReduceSumDown(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;  // 完整和仅 lane 0 有效
}
```


## 3. 前缀和

前缀和`__shfl_up_sync`沿 lane 号增大的方向传播。通过负向偏移量实现，常用于 Hillis-Steele 扫描算法。低索引线程若偏移越界则保持原值。经过 1、2、4 的偏移累加，Lane i 最终持有从 Lane 0 到 Lane i 的包含性前缀和。

此指令专用于局部扫描计算，如混合专家模型中统计各 Expert 的 Token 偏移量。

```cpp
// 初始状态: v = [0, 1, 2, 3, 4, 5, 6, 7]
// 目标: inclusive scan，lane i 得到 0..i 的和

// 第一步: delta = 1, 线程 i 读取 i-1 并累加；lane 0 没有左边，保持 0
// 流向: 0→1, 1→2, 2→3, 3→4, 4→5, 5→6, 6→7
// 结果: v = [0, 1, 3, 5, 7, 9, 11, 13]
{
    float t = __shfl_up_sync(0xffffffff, v, 1, 8);
    if (lane >= 1) v += t;
}

// 第二步: delta = 2, 流向: 0→2, 1→3, 2→4, 3→5, 4→6, 5→7
// 结果: v = [0, 1, 3, 6, 10, 14, 18, 22]
{
    float t = __shfl_up_sync(0xffffffff, v, 2, 8);
    if (lane >= 2) v += t;
}

// 第三步: delta = 4, 流向: 0→4, 1→5, 2→6, 3→7
// 结果: v = [0, 1, 3, 6, 10, 15, 21, 28]
{
    float t = __shfl_up_sync(0xffffffff, v, 4, 8);
    if (lane >= 4) v += t;
}
```

`if (lane >= delta)` 不能省。越界时 `__shfl_up_sync` 返回的是自己，裸写 `v +=` 会把 Lane 0 变成 $2v$。

```cpp
__device__ __forceinline__ float warpInclusiveScan(float v) {
    int lane = threadIdx.x & 31;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        float t = __shfl_up_sync(0xffffffff, v, d);
        if (lane >= d) v += t;
    }
    return v;
}
```


## 4. 广播与 Gather

`__shfl_sync` 定向广播允许 Warp 内所有活跃线程直接读取指定源线程的寄存器值。该操作通常作为向下归约的补充，用于将 Lane 0 的归约结果分发给全员。然而，完成一次归约加广播需要六次 Shuffle 操作，在全员均需结果的场景下，其指令开销高于蝶形归约的五次操作。


```cpp
// 前置条件: 经过向下归约后，v[0] = 28
// 目标: 将 Lane 0 的结果广播给全员

// 所有线程读取 srcLane = 0 的寄存器值
// 结果: v = [28, 28, 28, 28, 28, 28, 28, 28]
v = __shfl_sync(0xffffffff, v, 0, 8);
```

```cpp
__device__ __forceinline__ float warpBcast(float v, int src = 0) {
    return __shfl_sync(0xffffffff, v, src);
}
```



## 选型

| 计算模式 | 接口 | 典型场景 |
|---|---|---|
| 全员都要完整和 | `__shfl_xor_sync` | Softmax / RMSNorm / FlashAttention 的 warp 内合并 |
| 只要 Lane 0 去写内存 | `__shfl_down_sync` | Block reduce 末尾的 `atomicAdd` |
| 每人一个前缀和 | `__shfl_up_sync` | inclusive scan、MoE 的 token 偏移 |
| 点名某条 lane | `__shfl_sync` | 广播、gather、取固定 lane 的标量 |

