"""
Token Embedding — 被 Transformer 顶层调用。

查表 [vocab, d_model] 后乘 √d_model, 使嵌入量级与位置编码相当。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, padding_idx: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        self.padding_idx = padding_idx
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """[B, L] → [B, L, d_model]"""
        return self.weight[token_ids] * math.sqrt(self.d_model)


def main():
    torch.manual_seed(0)
    vocab_size, d_model = 100, 16
    ours = TokenEmbedding(vocab_size, d_model, padding_idx=0)
    official = nn.Embedding(vocab_size, d_model, padding_idx=0)
    official.weight.data.copy_(ours.weight.data)

    token_ids = torch.randint(0, vocab_size, (2, 5))
    out_ours = ours(token_ids)
    out_ref = official(token_ids) * math.sqrt(d_model)

    from ppytorch.console import ModuleResult
    return ModuleResult(
        "embeddings.token", "词嵌入: 查表 + √d_model",
        info=[("输入", "[2, 5]"), ("输出", tuple(out_ours.shape))],
        checks=[("== nn.Embedding(×√d)", torch.allclose(out_ours, out_ref, atol=1e-6))],
    )


if __name__ == "__main__":
    from ppytorch.console import render_report
    render_report(main())
