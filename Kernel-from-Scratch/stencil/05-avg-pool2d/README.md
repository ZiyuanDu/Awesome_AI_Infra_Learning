# AvgPool2d / 二维平均池化

**来源：** 面试扩展练习；形状、接口和边界行为由本题明确约定。

**练习层级：** 专项

同样的窗口布局，对窗口内实际存在的输入值求平均；填充位置不计入分母。

$$
Y_{n,c,i,j}=\frac{\sum_{(h,w)\in\mathcal W_{ij}\cap input}X_{n,c,h,w}}{|\mathcal W_{ij}\cap input|}
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

input 单通道 [[1,2],[3,4]]，k=2、s=2、p=0，输出 [[2.5]]。

## 约束与对照

OH=floor((H+2p-k)/s)+1，OW 同理；输出尺寸为正；0≤p≤floor(k/2)，ceil_mode=False，count_include_pad=False，每个输出窗口至少覆盖一个有效元素。

**PyTorch 对照：** `torch.nn.functional.avg_pool2d(input, k, s, p, count_include_pad=False)`

**前置练习：** [stencil/04-max-pool2d](../04-max-pool2d/README.md)
