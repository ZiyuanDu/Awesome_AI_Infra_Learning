# 颜色反转

**LeetGPU：** Easy · Color Inversion  

图像是一维 `unsigned char` 数组，按像素打包 **RGBA**（每通道 8 bit）。原地反转颜色，**A 不动**：

$$
R'=255-R,\quad G'=255-G,\quad B'=255-B,\quad A'=A
$$

`image` 长度 `width × height × 4`。前 4 个数是左上角像素。

## 接口

```cuda
void solve(unsigned char* image, int width, int height);
```

所有指针参数均指向 **GPU 设备内存**；数组由调用方分配，尺寸参数和其他标量按值传入。

| 参数 | 含义 |
|---|---|
| `image` | 长度为 width × height × 4 的 unsigned char RGBA 数组；原地反转 RGB，保留 Alpha。 |
| `width` | 图像宽度，单位为像素。 |
| `height` | 图像高度，单位为像素。 |

- `image` 是 **device** 指针，**原地**写回
- 不得改签名

## 示例

输入 1×1：`[10, 20, 30, 255]` → `[245, 235, 225, 255]`
