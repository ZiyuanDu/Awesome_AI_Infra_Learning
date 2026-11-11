# Mean Squared Error / 均方误差

**LeetGPU：** Medium · [官方题目与参考](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/27_mean_squared_error)

**练习层级：** 基础

融合逐点平方差和全局平均。

$$
mse=\frac1N\sum_i(predictions_i-targets_i)^2
$$

## 接口

```cuda
void solve(const float* predictions, const float* targets, float* mse, int N);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `predictions` | float32 预测值，长度 N。 |
| `targets` | float32 目标值，长度 N。 |
| `mse` | float32 输出，长度 1。 |
| `N` | 元素个数。 |

## 示例

predictions=[1,3]，targets=[0,1]，mse=2.5。

## 约束与对照

1≤N≤1e8；输入值在 [-1000,1000]。官方计时 N=50000000。

**PyTorch 对照：** `torch.nn.functional.mse_loss(predictions, targets)`

**前置练习：** [reduce/01-reduction](../01-reduction/README.md)
