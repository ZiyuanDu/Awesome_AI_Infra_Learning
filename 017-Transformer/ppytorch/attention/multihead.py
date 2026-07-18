"""
多头注意力 — 调用 sdpa。

    MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W^O
    head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)

工程上用一次大投影 + reshape 分头, 而非真开 h 个矩阵。
同一类支持自注意力 (q=k=v) 与交叉注意力 (q≠k=v)。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .sdpa import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, bias: bool = True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, d_model] → [B, h, L, d_k]"""
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, h, L, d_k] → [B, L, d_model]"""
        B, _, L, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, self.d_model)

    def forward(self, query, key, value, mask=None):
        """
        query: [B, Lq, d]  key/value: [B, Lk, d]
        mask: 可广播到 [B, h, Lq, Lk]
        → [B, Lq, d]
        """
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))
        out = scaled_dot_product_attention(Q, K, V, mask=mask)
        return self.W_o(self._merge_heads(out))


def main():
    torch.manual_seed(0)
    d_model, num_heads, B, L = 16, 4, 2, 5
    ours = MultiHeadAttention(d_model, num_heads).eval()

    official = nn.MultiheadAttention(d_model, num_heads, bias=True, batch_first=True).eval()
    with torch.no_grad():
        official.in_proj_weight.copy_(
            torch.cat([ours.W_q.weight, ours.W_k.weight, ours.W_v.weight], dim=0))
        official.in_proj_bias.copy_(
            torch.cat([ours.W_q.bias, ours.W_k.bias, ours.W_v.bias], dim=0))
        official.out_proj.weight.copy_(ours.W_o.weight)
        official.out_proj.bias.copy_(ours.W_o.bias)

    x = torch.randn(B, L, d_model)
    out_ours = ours(x, x, x)
    out_ref, _ = official(x, x, x, need_weights=False)

    q, kv = torch.randn(B, 3, d_model), torch.randn(B, L, d_model)
    out_cross = ours(q, kv, kv)
    out_cross_ref, _ = official(q, kv, kv, need_weights=False)

    causal = torch.triu(torch.full((L, L), float("-inf")), diagonal=1)
    out_m = ours(x, x, x, mask=causal)
    out_m_ref, _ = official(x, x, x, attn_mask=causal, need_weights=False)

    from ppytorch.console import ModuleResult
    return ModuleResult(
        "attention.multihead", f"多头注意力 (h={num_heads}, d_k={d_model // num_heads})",
        info=[("自注意力", tuple(out_ours.shape)), ("交叉注意力", tuple(out_cross.shape))],
        checks=[
            ("自注意力 == 官方", torch.allclose(out_ours, out_ref, atol=1e-5)),
            ("交叉注意力 == 官方", torch.allclose(out_cross, out_cross_ref, atol=1e-5)),
            ("因果掩码 == 官方", torch.allclose(out_m, out_m_ref, atol=1e-5)),
        ],
    )


if __name__ == "__main__":
    from ppytorch.console import render_report
    render_report(main())

