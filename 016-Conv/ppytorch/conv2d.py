import torch
import torch.nn.functional as F

def naive_conv2d(input_tensor, weight, bias=None, stride=1, padding=0):
    """
    input:  [B, C_in, H, W]
    weight: [C_out, C_in, KH, KW]
    """
    # 1. 获取各个维度的尺寸
    B, C_in, H_in, W_in = input_tensor.shape
    C_out, _, kH, kW = weight.shape
    
    stride_h = stride if isinstance(stride, int) else stride[0]
    stride_w = stride if isinstance(stride, int) else stride[1]
    
    pad_h = padding if isinstance(padding, int) else padding[0]
    pad_w = padding if isinstance(padding, int) else padding[1]
    
    if pad_h > 0 or pad_w > 0:
        input_padded = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0.0)
    else:
        input_padded = input_tensor
        
    _, _, H_pad, W_pad = input_padded.shape
    
    # 3. 计算输出的宽高尺寸
    H_out = (H_pad - kH) // stride_h + 1
    W_out = (W_pad - kW) // stride_w + 1
    
    # 4. 初始化一个全零的输出张量
    output = torch.zeros(B, C_out, H_out, W_out, device=input_tensor.device)
    
    for b in range(B):                   # 遍历 Batch 中的每一张图
        for co in range(C_out):          # 遍历每一个卷积核
            for h in range(H_out):       # 遍历输出特征图的高
                for w in range(W_out):   # 遍历输出特征图的宽
                    
                    # 计算当前滑动窗口在 input_padded 上的左上角坐标
                    h_start = h * stride_h
                    w_start = w * stride_w
                    h_end = h_start + kH
                    w_end = w_start + kW
                    
                    # 提取当前窗口内所有输入通道的数据，形状为 (C_in, kH, kW)
                    current_window = input_padded[b, :, h_start:h_end, w_start:w_end]
                    
                    # 获取当前输出通道对应的卷积核权重，形状为 (C_in, kH, kW)
                    current_kernel = weight[co]
                    
                    # 核心数学操作：对应位置元素相乘，然后对所有元素求和 (C_in * kH * kW 个值相加)
                    # torch.sum(current_window * current_kernel) 会自动处理 C_in 维度的相加
                    pixel_value = torch.sum(current_window * current_kernel)
                    
                    # 如果有偏置，加上偏置
                    if bias is not None:
                        pixel_value += bias[co]
                        
                    # 将计算结果存入输出张量
                    output[b, co, h, w] = pixel_value
                    
    return output


B, C_in, H_in, W_in = 1, 2, 5, 5
C_out, kH, kW = 2, 3, 3
stride = 1
padding = 1

input_tensor = torch.randn(B, C_in, H_in, W_in)
weight = torch.randn(C_out, C_in, kH, kW)
bias = torch.randn(C_out)

# 官方对比
official_res = F.conv2d(input_tensor, weight, bias=bias, stride=stride, padding=padding)

# Naive 版本
naive_res = naive_conv2d(input_tensor, weight, bias=bias, stride=stride, padding=padding)

print("输出形状:", naive_res.shape)
print("Naive实现是否正确:", torch.allclose(official_res, naive_res, atol=1e-5))