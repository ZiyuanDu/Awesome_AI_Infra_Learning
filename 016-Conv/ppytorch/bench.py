import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image 
from torchvision.transforms import functional as TF
from conv2d import im2col_conv2d

def gaussian_kernel_2d(size, sigma=1.0):
    """生成一个 2D 高斯核，形状为 (size, size)"""
    ax = torch.arange(size, dtype=torch.float32) - size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def compare_conv(image_path, kernel_size=3, sigma=1.0):

    # 1. 读取图片并转为灰度单通道张量 [1, 1, H, W]
    img = Image.open(image_path).convert('L')          # 'L' 是灰度
    img_tensor = TF.to_tensor(img).unsqueeze(0)        # [1,1,H,W] 值域 [0,1]

    # 2. 生成高斯核，形状为 [1, 1, kH, kW]
    kernel_2d = gaussian_kernel_2d(kernel_size, sigma)
    weight = kernel_2d.view(1, 1, kernel_size, kernel_size)   # C_out=1, C_in=1

    # 3. 确定 padding 以保持尺寸不变
    pad = kernel_size // 2 

    # 4. 官方卷积
    with torch.no_grad():
        official_res = F.conv2d(img_tensor, weight, bias=None, stride=1, padding=pad)

    # 5. 自定义卷积
    with torch.no_grad():
        naive_res = im2col_conv2d(img_tensor, weight, bias=None, stride=1, padding=pad)

    # 6. 检查数值是否一致
    is_close = torch.allclose(official_res, naive_res, atol=1e-5)
    print(f"官方卷积与自定义卷积结果是否一致？ {is_close}")
    if not is_close:
        max_diff = (official_res - naive_res).abs().max().item()
        print(f"最大误差: {max_diff:.6e}")

    # 7. 可视化：原图、官方结果、自定义结果
    # 将张量转为 numpy 用于 matplotlib
    original_np = img_tensor.squeeze().numpy()                # [H, W]
    official_np = official_res.squeeze().numpy()              # [H_out, W_out]
    naive_np = naive_res.squeeze().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_np, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(official_np, cmap='gray')
    axes[1].set_title('Official Conv2d (Gaussian)')
    axes[1].axis('off')

    axes[2].imshow(naive_np, cmap='gray')
    axes[2].set_title('Naive Conv2d (Gaussian)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig("output.png")

    return is_close


if __name__ == "__main__":
    compare_conv('input.png', kernel_size=5, sigma=1.5)