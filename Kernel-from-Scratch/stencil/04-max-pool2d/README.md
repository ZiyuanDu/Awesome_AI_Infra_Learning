# MaxPool2d / 二维最大池化

**LeetGPU：** Medium · [官方题目与参考](https://github.com/AlphaGPU/leetgpu-challenges/tree/main/challenges/medium/42_2d_max_pooling)

**练习层级：** 专项

对每个通道分别取局部窗口最大值，填充区域按负无穷处理。

$$
OH=\lfloor(H+2p-k)/s\rfloor+1,\quad OW=\lfloor(W+2p-k)/s\rfloor+1
$$

## 接口

```cuda
void solve(
    const float* input,
    float* output,
    int N,
    int C,
    int H,
    int W,
    int kernel_size,
    int stride,
    int padding
);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `input` | float32 输入，形状 (N, C, H, W)，NCHW 连续布局。 |
| `output` | float32 输出，形状 (N, C, OH, OW)。 |
| `N` | batch 大小。 |
| `C` | 通道数。 |
| `H` | 输入高度。 |
| `W` | 输入宽度。 |
| `kernel_size` | 方形窗口边长 k。 |
| `stride` | 高宽共同的正步长 s。 |
| `padding` | 高宽两侧共同的非负填充 p。 |

## 示例

input 单通道 [[1,2],[3,4]]，k=2、s=2、p=0，输出 [[4]]。

## 约束与对照

N,C,H,W≥1，k,s≥1，p≥0，OH,OW>0；padding 还需满足 PyTorch 的合法池化约束 p≤floor(k/2)。dilation=1，ceil_mode=False。

**PyTorch 对照：** `torch.nn.functional.max_pool2d(input, k, s, p)`

**前置练习：** [stencil/02-conv2d](../02-conv2d/README.md)
