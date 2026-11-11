# RGB 转灰度

**LeetGPU：** Easy · [RGB to Grayscale](https://leetgpu.com/challenges/rgb-to-grayscale)  

输入是 `height × width × 3` 的 float，按像素交错 `R,G,B,R,G,B,…`。输出 `height × width` 灰度：

$$
\mathrm{gray}=0.299\,R+0.587\,G+0.114\,B
$$

系数必须用这三个数。例：`[255,0,0] → 76.245`，`[100,150,200] → 140.75`。

## 接口

```cuda
void solve(const float* input, float* output, int width, int height);
```

所有指针参数均指向 **GPU 设备内存**；数组由调用方分配，尺寸参数和其他标量按值传入。

| 参数 | 含义 |
|---|---|
| `input` | 长度为 width × height × 3 的只读 float 数组，按像素交错存储 R、G、B。 |
| `output` | 长度为 width × height 的 float 灰度输出数组。 |
| `width` | 图像宽度，单位为像素。 |
| `height` | 图像高度，单位为像素。 |

`1 ≤ width,height ≤ 4096`，`width×height ≤ 2^{22}`，RGB ∈ [0, 255]。计时 `2048×2048`。
