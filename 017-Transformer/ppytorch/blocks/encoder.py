"""
Encoder — 调用 MultiHeadAttention / FeedForward / LayerNorm。

每层: 自注意力 + FFN, 各套残差与 LN。
  Post-LN: x = LN(x + Sublayer(x))
  Pre-LN:  x = x + Sublayer(LN(x))
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ppytorch.attention import MultiHeadAttention
from ppytorch.layers import FeedForward, LayerNorm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.0,
                 activation="relu", norm_first=False):
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, activation)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask=None) -> torch.Tensor:
        if self.norm_first:
            nx = self.norm1(x)
            x = x + self.dropout(self.self_attn(nx, nx, nx, mask=src_mask))
            x = x + self.dropout(self.ffn(self.norm2(x)))
        else:
            x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask=src_mask)))
            x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff=None,
                 dropout=0.0, activation="relu", norm_first=False):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, activation, norm_first)
            for _ in range(num_layers)
        ])
        self.final_norm = LayerNorm(d_model) if norm_first else None

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        if self.final_norm is not None:
            x = self.final_norm(x)
        return x


def _copy_encoder_layer_weights(ours: EncoderLayer, official: nn.TransformerEncoderLayer):
    with torch.no_grad():
        official.self_attn.in_proj_weight.copy_(torch.cat(
            [ours.self_attn.W_q.weight, ours.self_attn.W_k.weight, ours.self_attn.W_v.weight]))
        official.self_attn.in_proj_bias.copy_(torch.cat(
            [ours.self_attn.W_q.bias, ours.self_attn.W_k.bias, ours.self_attn.W_v.bias]))
        official.self_attn.out_proj.weight.copy_(ours.self_attn.W_o.weight)
        official.self_attn.out_proj.bias.copy_(ours.self_attn.W_o.bias)
        official.linear1.weight.copy_(ours.ffn.w1.weight)
        official.linear1.bias.copy_(ours.ffn.w1.bias)
        official.linear2.weight.copy_(ours.ffn.w2.weight)
        official.linear2.bias.copy_(ours.ffn.w2.bias)
        official.norm1.weight.copy_(ours.norm1.gamma)
        official.norm1.bias.copy_(ours.norm1.beta)
        official.norm2.weight.copy_(ours.norm2.gamma)
        official.norm2.bias.copy_(ours.norm2.beta)


def main():
    torch.manual_seed(0)
    d_model, num_heads, B, L = 16, 4, 2, 5
    x = torch.randn(B, L, d_model)

    ours = EncoderLayer(d_model, num_heads, d_ff=64, activation="relu", norm_first=False).eval()
    official = nn.TransformerEncoderLayer(
        d_model, num_heads, dim_feedforward=64, dropout=0.0,
        activation="relu", batch_first=True, norm_first=False).eval()
    _copy_encoder_layer_weights(ours, official)

    out_ours, out_ref = ours(x), official(x)
    enc = Encoder(3, d_model, num_heads, d_ff=64).eval()

    from ppytorch.console import ModuleResult
    return ModuleResult(
        "blocks.encoder", "编码器: 自注意力 + FFN (残差&LN)",
        info=[("单层", tuple(out_ours.shape)), ("Encoder(3)", tuple(enc(x).shape))],
        checks=[("EncoderLayer == 官方", torch.allclose(out_ours, out_ref, atol=1e-5))],
    )


if __name__ == "__main__":
    from ppytorch.console import render_report
    render_report(main())
