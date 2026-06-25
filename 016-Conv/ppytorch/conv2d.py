import torch
import torch.nn.functional as F


def naive_conv2d(input_tensor, weight, bias=None, stride=1, padding=0):
    """
    input:  [B, C_in, H, W]
    weight: [C_out, C_in, KH, KW]
    """

    B, _, _, _ = input_tensor.shape
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
    
    # 计算输出的宽高尺寸
    H_out = (H_pad - kH) // stride_h + 1
    W_out = (W_pad - kW) // stride_w + 1
    
    # 初始化一个全零的输出张量
    output = torch.zeros(B, C_out, H_out, W_out, device=input_tensor.device)
    
    for b in range(B):                   # 遍历 Batch 中的每一张图
        for co in range(C_out):          # 遍历每一个卷积核
            for h in range(H_out):       # 遍历输出特征图的高
                for w in range(W_out):   # 遍历输出特征图的宽

                    # 上述for循环都是从输出的视角计算的
                    # 输出特征图的每一个像素，都对应输入图像上的一个窗口

                    h_start = h * stride_h # 窗口在高度方向的起始行
                    w_start = w * stride_w # 窗口在宽度方向的起始行
                    h_end = h_start + kH   # 窗口在高度方向的结束行（不包含）
                    w_end = w_start + kW   # 窗口在宽度方向的结束行（不包含） 
                    
                    # 提取当前窗口内所有输入通道的数据，形状为 (C_in, kH, kW)
                    current_window = input_padded[b, :, h_start:h_end, w_start:w_end]
                    
                    # 获取当前输出通道对应的卷积核权重，形状为 (C_in, kH, kW)
                    current_kernel = weight[co]
                    
                    # pixel_value = torch.sum(current_window * current_kernel)
                    pixel_value = torch.einsum('cij,cij->', current_window, current_kernel)

                    # 如果有偏置，加上偏置
                    if bias is not None:
                        pixel_value += bias[co]
                        
                    # 将计算的标量写入到对应位置
                    output[b, co, h, w] = pixel_value
                    
    return output


def im2col_conv2d(input_tensor, weight, bias=None, stride=1, padding=0):
    B, C_in, H_in, W_in = input_tensor.shape
    C_out, _, kH, kW = weight.shape
    stride_h = stride if isinstance(stride, int) else stride[0]
    stride_w = stride if isinstance(stride, int) else stride[1]
    pad_h = padding if isinstance(padding, int) else padding[0]
    pad_w = padding if isinstance(padding, int) else padding[1]


    if pad_h > 0 or pad_w > 0:
        input_padded = F.pad(input_tensor, (pad_w, pad_w, pad_h, pad_h),
                             mode='constant', value=0.0)
    else:
        input_padded = input_tensor


    # im2col的核心差异
    # [B, C_in * kH * kW, H_out * W_out]
    cols = F.unfold(input_padded, kernel_size=(kH, kW), stride=(stride_h, stride_w))

    H_out = (input_padded.shape[2] - kH) // stride_h + 1
    W_out = (input_padded.shape[3] - kW) // stride_w + 1
    L = H_out * W_out

    W_mat = weight.view(C_out, -1) # [C_out, C_in * kH * kW]

    # [B, L, C_in*kH*kW] * [C_in*kH*kW, C_out]
    out = torch.matmul(cols.transpose(1, 2), W_mat.transpose(0, 1))
    # 输出形状:[B, L, C_out]
    out = out.transpose(1, 2)

    if bias is not None:
        out += bias.view(1, -1, 1)

    out = out.view(B, C_out, H_out, W_out)

    return out 