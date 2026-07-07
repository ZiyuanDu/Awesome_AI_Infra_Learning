import torch

def softmax(x, dim=-1):
    """
    softmax(z)_i = exp(z_i) / sum_j exp(z_j)
    """
    # 第 1 步：求最大值
    # 沿目标维度 dim 求最大值，keepdim=True 保持原维度数，
    # 便于后续广播减法。x_max 的形状会在 dim 维度变成 1，其他维不变。
    x_max = x.max(dim=dim, keepdim=True)[0]   # [0] 取出最大值

    # 第 2 步：平移输入数据，减去最大值
    # 数学上等价于将分子分母同除以 exp(x_max)，防止指数爆炸。
    # 因为 x_max 是沿 dim 维的最大值，x - x_max 的所有元素 ≤ 0，
    safe_x = x - x_max                     # 广播机制 

    # 第 3 步：指数计算
    # 对平移后的张量逐元素求 exp。
    exp_x = torch.exp(safe_x)

    # 第 4 步：求和
    # 沿 dim 维度对 exp_x 求和，同样 keepdim=True 保持维度以便广播。
    sum_exp = exp_x.sum(dim=dim, keepdim=True)

    # 第 5 步：归一化得到最终概率
    output = exp_x / sum_exp

    return output

def naive_softmax(x, dim=-1):
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

if __name__ == "__main__":
    # 一维向量
    x1 = torch.tensor([1.0, 2.0, 3.0])
    p1 = softmax(x1)
    print("一维 softmax:\n",p1)
    print(f"Torch Softmax:\n{torch.softmax(x1, dim=-1)}")
    print("求和 =", p1.sum().item())  # 应为 1

    # 二维矩阵
    x2 = torch.tensor([[1., 2., 3.],
                       [4., 5., 6.]])
    p2 = softmax(x2, dim=-1)          # 对每一行做 softmax
    print("\n二维 softmax:\n",p2)
    print(f"Torch Softmax:\n{torch.softmax(x2, dim=-1)}")
    print("每行求和 =", p2.sum(dim=-1))  # 结果 [1, 1]
