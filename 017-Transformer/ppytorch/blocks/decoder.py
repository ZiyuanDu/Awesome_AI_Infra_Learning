"""
Decoder — 调用 MultiHeadAttention / FeedForward / LayerNorm。

每层三子层: 掩码自注意力 → 交叉注意力 → FFN。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ppytorch.attention import MultiHeadAttention
from ppytorch.layers import FeedForward, LayerNorm


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.0,
                 activation="relu", norm_first=False):
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, activation)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        """
        x: [B, Lt, d]  memory: [B, Ls, d]
        self_attn:  Q=K=V=x (+ 因果掩码)
        cross_attn: Q=x, K=V=memory
        """
        if self.norm_first:
            nx = self.norm1(x)
            x = x + self.dropout(self.self_attn(nx, nx, nx, mask=tgt_mask))
            nx = self.norm2(x)
            x = x + self.dropout(self.cross_attn(nx, memory, memory, mask=memory_mask))
            x = x + self.dropout(self.ffn(self.norm3(x)))
        else:
            x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask=tgt_mask)))
            x = self.norm2(x + self.dropout(
                self.cross_attn(x, memory, memory, mask=memory_mask)))
            x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff=None,
                 dropout=0.0, activation="relu", norm_first=False):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, activation, norm_first)
            for _ in range(num_layers)
        ])
        self.final_norm = LayerNorm(d_model) if norm_first else None

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        if self.final_norm is not None:
            x = self.final_norm(x)
        return x


def _copy_decoder_layer_weights(ours: DecoderLayer, official: nn.TransformerDecoderLayer):
    with torch.no_grad():
        official.self_attn.in_proj_weight.copy_(torch.cat(
            [ours.self_attn.W_q.weight, ours.self_attn.W_k.weight, ours.self_attn.W_v.weight]))
        official.self_attn.in_proj_bias.copy_(torch.cat(
            [ours.self_attn.W_q.bias, ours.self_attn.W_k.bias, ours.self_attn.W_v.bias]))
        official.self_attn.out_proj.weight.copy_(ours.self_attn.W_o.weight)
        official.self_attn.out_proj.bias.copy_(ours.self_attn.W_o.bias)
        official.multihead_attn.in_proj_weight.copy_(torch.cat(
            [ours.cross_attn.W_q.weight, ours.cross_attn.W_k.weight, ours.cross_attn.W_v.weight]))
        official.multihead_attn.in_proj_bias.copy_(torch.cat(
            [ours.cross_attn.W_q.bias, ours.cross_attn.W_k.bias, ours.cross_attn.W_v.bias]))
        official.multihead_attn.out_proj.weight.copy_(ours.cross_attn.W_o.weight)
        official.multihead_attn.out_proj.bias.copy_(ours.cross_attn.W_o.bias)
        official.linear1.weight.copy_(ours.ffn.w1.weight)
        official.linear1.bias.copy_(ours.ffn.w1.bias)
        official.linear2.weight.copy_(ours.ffn.w2.weight)
        official.linear2.bias.copy_(ours.ffn.w2.bias)
        for o, m in [(official.norm1, ours.norm1),
                     (official.norm2, ours.norm2),
                     (official.norm3, ours.norm3)]:
            o.weight.copy_(m.gamma)
            o.bias.copy_(m.beta)


def main():
    torch.manual_seed(0)
    d_model, num_heads, B, Lt, Ls = 16, 4, 2, 4, 5
    ours = DecoderLayer(d_model, num_heads, d_ff=64, norm_first=False).eval()
    official = nn.TransformerDecoderLayer(
        d_model, num_heads, dim_feedforward=64, dropout=0.0,
        activation="relu", batch_first=True, norm_first=False).eval()
    _copy_decoder_layer_weights(ours, official)

    x, memory = torch.randn(B, Lt, d_model), torch.randn(B, Ls, d_model)
    causal = torch.triu(torch.full((Lt, Lt), float("-inf")), diagonal=1)
    out_ours = ours(x, memory, tgt_mask=causal)
    out_ref = official(x, memory, tgt_mask=causal)

    from ppytorch.console import ModuleResult
    return ModuleResult(
        "blocks.decoder", "解码器: 掩码自注意力 + 交叉注意力 + FFN",
        info=[("输出", tuple(out_ours.shape)),
              ("x / memory", f"[{B},{Lt},{d_model}] / [{B},{Ls},{d_model}]")],
        checks=[("DecoderLayer == 官方", torch.allclose(out_ours, out_ref, atol=1e-5))],
    )


if __name__ == "__main__":
    from ppytorch.console import render_report
    render_report(main())
