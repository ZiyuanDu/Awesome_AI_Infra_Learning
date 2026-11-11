# Depthwise Conv2d / 逐通道卷积

**来源：** 面试扩展练习；形状、接口和边界行为由本题明确约定。

**练习层级：** 专项

每个输入通道使用自己的卷积核，通道间不混合。基础版无偏置、步长 1、无 padding。

$$
Y_{n,c,i,j}=\sum_{u=0}^{KH-1}\sum_{v=0}^{KW-1}X_{n,c,i+u,j+v}W_{c,u,v}
$$

## 接口

```cuda
void solve(
    const float* input,
    const float* kernel,
    float* output,
    int N,
    int C,
    int H,
    int W,
    int KH,
    int KW
);
```

所有指针均指向 **GPU 设备内存**，由调用方分配；标量按值传入。除题面另有说明外，张量按给定形状行优先连续存储，输入和输出不别名。

| 参数 | 含义 |
|---|---|
| `input` | float32 输入，形状 (N, C, H, W)，NCHW 布局。 |
| `kernel` | float32 卷积核，形状 (C, KH, KW)；每通道一个核。 |
| `output` | float32 输出，形状 (N, C, H-KH+1, W-KW+1)。 |
| `N` | batch 大小。 |
| `C` | 输入及输出通道数。 |
| `H` | 输入高度。 |
| `W` | 输入宽度。 |
| `KH` | 卷积核高度。 |
| `KW` | 卷积核宽度。 |

## 示例

N=C=1，input=[[1,2],[3,4]]，kernel=[[1]]，输出与输入相同。

## 约束与对照

所有尺寸为正，KH≤H、KW≤W；depth multiplier=1，stride=1，padding=0，dilation=1；按深度学习惯例做互相关，不翻转核。

**PyTorch 对照：** `torch.nn.functional.conv2d(input, kernel[:,None,:,:], groups=C)`

**前置练习：** [stencil/02-conv2d](../02-conv2d/README.md)
