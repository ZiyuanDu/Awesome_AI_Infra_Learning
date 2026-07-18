"""
Position-wise FFN — 被 Encoder/Decoder 子层调用。

    FFN(x) = W2(act(W1(x)))
    d_ff 默认 4 * d_model; activation: relu | gelu
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None,
                 activation: str = "relu", bias: bool = True):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)
        self.act = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)))


def main():
    torch.manual_seed(0)
    d_model = 16
    ffn = FeedForward(d_model, activation="relu")
    x = torch.randn(2, 5, d_model)
    out = ffn(x)
    ref = ffn.w2(F.relu(ffn.w1(x)))

    from ppytorch.console import ModuleResult
    return ModuleResult(
        "layers.feed_forward", "前馈: 升维→激活→降维",
        info=[("输出", tuple(out.shape)), ("d_ff", ffn.w1.out_features)],
        checks=[("== w2(relu(w1(x)))", torch.allclose(out, ref, atol=1e-6))],
    )


if __name__ == "__main__":
    from ppytorch.console import render_report
    render_report(main())
