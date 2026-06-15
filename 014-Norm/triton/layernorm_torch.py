import torch 
import torch.nn as nn 



class layernorm(nn.Module):
    def __init__(self, embed_size, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embed_size))
        self.beta = nn.Parameter(torch.zeros(embed_size))
        self.eps = eps

    def forward(self, x):
        var, mean = torch.var_mean(x, dim=-1, keepdim=True, correction=0)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.gamma + self.beta

if __name__ == "__main__":
    x = torch.randn(2, 4, 3)
    embed_size = x.size(-1)
    official = nn.LayerNorm(embed_size, eps=1e-5)
    my  = layernorm(embed_size, eps=1e-5)

    # 复制相同权重
    my.gamma.data = official.weight.data.clone()
    my.beta.data  = official.bias.data.clone()

    diff = (official(x) - my(x)).abs().max().item()
    print(f"Max difference: {diff:.2e}")

