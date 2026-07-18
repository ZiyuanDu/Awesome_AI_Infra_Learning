"""
LayerNorm — 被 Encoder/Decoder 子层调用。

    y = (x - mean) / sqrt(var + eps) * gamma + beta

沿最后一维 (d_model) 归一化, 与 batch / 序列长度无关。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.gamma + self.beta


def main():
    torch.manual_seed(0)
    d_model = 16
    ours, official = LayerNorm(d_model), nn.LayerNorm(d_model)
    x = torch.randn(2, 5, d_model)
    out_ours, out_ref = ours(x), official(x)

    from ppytorch.console import ModuleResult
    return ModuleResult(
        "layers.layer_norm", "层归一化: 沿特征维均值0方差1",
        info=[("输出形状", tuple(out_ours.shape))],
        checks=[
            ("== 官方 nn.LayerNorm", torch.allclose(out_ours, out_ref, atol=1e-5)),
            ("归一化后均值≈0",
             torch.allclose(out_ours.mean(-1), torch.zeros(2, 5), atol=1e-5)),
        ],
    )


if __name__ == "__main__":
    from ppytorch.console import render_report
    render_report(main())
