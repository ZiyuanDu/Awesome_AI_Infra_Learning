"""
位置编码 — 被 Transformer 顶层调用。

  SinusoidalPE: 论文原版, sin/cos
  LearnedPE:    可学习绝对位置查找表 (BERT/GPT 风格)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


class LearnedPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.pe(positions)


def main():
    torch.manual_seed(0)
    d_model, L = 32, 10
    pe = SinusoidalPE(d_model)
    out = pe(torch.zeros(1, L, d_model))

    diff = not torch.allclose(out[0, 0], out[0, 1])
    norm_sq = (out[0, 5] ** 2).sum().item()
    ev0 = torch.allclose(out[0, 0, 0::2], torch.zeros(d_model // 2), atol=1e-6)
    od1 = torch.allclose(out[0, 0, 1::2], torch.ones(d_model // 2), atol=1e-6)

    # LearnedPE 接口一致: 形状不变
    learned = LearnedPE(d_model, max_len=L)
    learned_out = learned(torch.zeros(1, L, d_model))

    from ppytorch.console import ModuleResult
    return ModuleResult(
        "embeddings.positional", "正弦 / 可学习位置编码",
        info=[("Sinusoidal 输出", tuple(out.shape)),
              ("Learned 输出", tuple(learned_out.shape)),
              ("位置5 范数²", f"{norm_sq:.4f} (应≈{d_model / 2})")],
        checks=[
            ("相邻位置编码不同", diff),
            (f"范数² = d_model/2", abs(norm_sq - d_model / 2) < 1e-3),
            ("PE(0) 偶数维=0", ev0),
            ("PE(0) 奇数维=1", od1),
            ("LearnedPE 形状正确", tuple(learned_out.shape) == (1, L, d_model)),
        ],
    )


if __name__ == "__main__":
    from ppytorch.console import render_report
    render_report(main())
