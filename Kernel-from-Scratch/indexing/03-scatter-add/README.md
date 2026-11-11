# Scatter Add / 按索引聚合

**LeetGPU：** Medium · [官方题目与参考](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/125_scatter_add)

**练习层级：** 基础

把每个源行累加到指定目标行，重复目标需要合并贡献。

$$
output_{m,d}=\sum_{i:\,index_i=m}src_{i,d}
$$

## 接口

```cuda
void solve(const float* src, const int* index, float* output, int N, int D, int M);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `src` | float32 输入，形状 (N, D)。 |
| `index` | int32 目标行索引，长度 N。 |
| `output` | float32 输出，形状 (M, D)，初始内容未定义。 |
| `N` | 源行数。 |
| `D` | 每行通道数。 |
| `M` | 目标行数。 |

## 示例

src=[[1,2],[3,4],[5,6]]、index=[0,1,0]、M=3，输出 [[6,8],[3,4],[0,0]]。

## 约束与对照

1≤N≤2000000，1≤D≤1024，1≤M≤1000000，0≤index[i]<M。没有任何源行映射到的目标行也必须写 0。

**PyTorch 对照：** `output.zero_()；output.index_add_(0, index.long(), src)。`

不能依赖调用方预清零。原子加的浮点累加次序可能不同，正确性对照应使用合理容差。

**前置练习：** [indexing/01-gather](../01-gather/README.md)
