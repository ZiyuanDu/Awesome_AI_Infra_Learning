import torch 
from torch import nn

def manual_batchnorm(x, running_mean, running_var, weight, bias, training, momentum=0.1, eps=1e-5):

    
    """
        x.dim() 返回张量的维度数，例如对于一个形状为 (3, 4, 5) 的张量，x.dim() 将返回 3。
        range(x.dim()) 生成一个从 0 到 x.dim() - 1 的序列，例如生成 [0, 1, 2]。
        [d for d in range(x.dim()) if d != 1] 是列表推导式，它遍历上述生成的序列，并将不等于 1 的维度索引添加到列表中。
    """
    dims = [d for d in range(x.dim()) if d != 1]

    """
        [1: -1]是一个列表，第一个元素是1，第二个元素是-1
        [1] * (x.dim() - 2) 是一个列表，包含 x.dim() - 2 个元素，每个元素都是1。
        shape表示前两个维度是1和-1，后边的维度全都是1
    """
    shape = [1, -1] + [1] * (x.dim() - 2)   

    # 训练模式
    if training:
        mean = x.mean(dim=dims, keepdim=True)                       # shape: (1, C, 1, 1)        
        var_biased = x.var(dim=dims, keepdim=True, correction=0)    # 有偏估计，用于当前批次的归一化
        var_unbiased = x.var(dim=dims, keepdim=True, correction=1)  # 无偏估计，用于更新全局统计量

        with torch.no_grad():
            # running_mean 和running_var 形状是(C,)， 用squeeze()挤掉多余的维度
            running_mean.data = (1 - momentum) * running_mean.data + momentum * mean.squeeze()          
            running_var.data  = (1 - momentum) * running_var.data  + momentum * var_unbiased.squeeze()
        var = var_biased
    else:
        mean = running_mean.view(shape)
        var = running_var.view(shape)

    # 归一化与仿射变换
    x_norm = (x - mean) / torch.sqrt(var + eps)
    if weight is not None:
        w = weight.view(shape)
        b = bias.view(shape)
        return x_norm * w + b
    return x_norm


def manual_batchnorm_v2(x, running_mean, running_var, weight, bias, training, momentum=0.1, eps=1e-5):
    dims = [d for d in range(x.dim()) if d != 1]
    shape = [1, -1] + [1] * (x.dim() - 2)

    if training:
        # 计算参与归约的元素总数
        n = 1
        for d in dims:
            n *= x.shape[d]
        
        # 一次性计算均值和方差（无偏方差）
        var_unbiased, mean = torch.var_mean(x, dim=dims, keepdim=True, correction=1)
        # 转为有偏方差用于归一化
        var_biased = var_unbiased * ((n - 1) / n)
        
        with torch.no_grad():
            running_mean.data = (1 - momentum) * running_mean.data + momentum * mean.squeeze()
            running_var.data  = (1 - momentum) * running_var.data  + momentum * var_unbiased.squeeze()
        var = var_biased
    else:
        mean = running_mean.view(shape)
        var = running_var.view(shape)

    x_norm = (x - mean) / torch.sqrt(var + eps)
    if weight is not None:
        return x_norm * weight.view(shape) + bias.view(shape)
    return x_norm

if __name__ == "__main__":
    torch.manual_seed(0)

    bn = nn.BatchNorm1d(4, momentum=0.1, eps=1e-5)
    x = torch.randn(3, 4)

    r_mean = bn.running_mean.clone()
    r_var  = bn.running_var.clone()

    out_native = bn(x)

    out_manual = manual_batchnorm(
        x, r_mean, r_var, bn.weight, bn.bias,
        training=bn.training, momentum=bn.momentum, eps=bn.eps
    )
    out_manual_v2 = manual_batchnorm_v2(
        x, r_mean, r_var, bn.weight, bn.bias,
        training=bn.training, momentum=bn.momentum, eps=bn.eps
    )



    print("最大差异:", (out_native - out_manual_v2).abs().max().item())