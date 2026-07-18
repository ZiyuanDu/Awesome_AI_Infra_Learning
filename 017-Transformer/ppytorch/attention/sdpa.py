"""
缩放点积注意力 — 调用层级最底层。

    Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V

两种实现:
  1) naive_attention              三重循环, 对照公式 (仅教学)
  2) scaled_dot_product_attention 矩阵化 (实际使用)

掩码: 加性 float, 0 保留 / -inf 屏蔽, 可广播到 [..., Lq, Lk]。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def naive_attention(Q, K, V, mask=None):
    """
    循环版。Q/K/V: [B, Lq|Lk, d_k|d_v] → [B, Lq, d_v]
    """
    B, Lq, d_k = Q.shape
    Lk = K.shape[1]
    out = torch.zeros(B, Lq, V.shape[2], device=Q.device, dtype=Q.dtype)
    scale = 1.0 / math.sqrt(d_k)
    for b in range(B):
        for i in range(Lq):
            scores = torch.empty(Lk, device=Q.device, dtype=Q.dtype)
            for j in range(Lk):
                scores[j] = torch.dot(Q[b, i], K[b, j]) * scale
            if mask is not None:
                scores = scores + mask[b, i]
            w = torch.softmax(scores, dim=-1)
            out[b, i] = (w.unsqueeze(1) * V[b]).sum(dim=0)
    return out


def scaled_dot_product_attention(Q, K, V, mask=None, return_weights=False):
    """
    矩阵化版。支持任意前置维 (batch / head ...)。
    Q: [..., Lq, d_k]  K: [..., Lk, d_k]  V: [..., Lk, d_v] → [..., Lq, d_v]
    """
    d_k = Q.size(-1)
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    weights = torch.softmax(scores, dim=-1)
    out = weights @ V
    if return_weights:
        return out, weights
    return out


def main():
    torch.manual_seed(0)
    B, L, d_k = 2, 4, 8
    Q = K = V = torch.randn(B, L, d_k)

    out_naive = naive_attention(Q, K, V)
    out_mat = scaled_dot_product_attention(Q, K, V)
    out_ref = F.scaled_dot_product_attention(Q, K, V)

    causal = torch.triu(torch.full((L, L), float("-inf")), diagonal=1)
    out_mask = scaled_dot_product_attention(Q, K, V, mask=causal)
    out_ref_causal = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

    from ppytorch.console import ModuleResult, matrix_panel
    return ModuleResult(
        "attention.sdpa", "缩放点积注意力: softmax(QK^T/√d_k)·V",
        info=[("Q/K/V 形状", f"[{B}, {L}, {d_k}]"), ("输出形状", tuple(out_mat.shape))],
        checks=[
            ("naive == 矩阵化", torch.allclose(out_naive, out_mat, atol=1e-5)),
            ("矩阵化 == 官方 F.sdpa", torch.allclose(out_mat, out_ref, atol=1e-5)),
            ("因果掩码 == 官方 is_causal", torch.allclose(out_mask, out_ref_causal, atol=1e-5)),
        ],
        extra=[matrix_panel(causal, "因果掩码 (0=可见, -∞=屏蔽)")],
    )


if __name__ == "__main__":
    from ppytorch.console import render_report
    render_report(main())
