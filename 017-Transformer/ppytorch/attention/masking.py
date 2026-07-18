"""
注意力掩码 — 与 sdpa 同层, 被 Transformer / MHA 调用。

  因果掩码: 位置 i 不能看 j>i (防偷看未来)
  填充掩码: 屏蔽 <pad> key
  合并: 加性掩码直接相加 (任一 -inf 即屏蔽)
"""

from __future__ import annotations

import torch


def causal_mask(L: int, device=None) -> torch.Tensor:
    """[1, 1, L, L], 严格上三角为 -inf。"""
    mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
    out = torch.zeros(L, L, device=device)
    out.masked_fill_(mask, float("-inf"))
    return out.view(1, 1, L, L)


def padding_mask(pad_positions: torch.Tensor) -> torch.Tensor:
    """
    pad_positions: [B, Lk] bool (True=pad)
    → [B, 1, 1, Lk]
    """
    B, Lk = pad_positions.shape
    out = torch.zeros(B, 1, 1, Lk, device=pad_positions.device)
    out.masked_fill_(pad_positions[:, None, None, :], float("-inf"))
    return out


def combine_masks(*masks) -> torch.Tensor | None:
    """合并多个加性掩码; 忽略 None。"""
    valid = [m for m in masks if m is not None]
    if not valid:
        return None
    total = valid[0]
    for m in valid[1:]:
        total = total + m
    return total


def main():
    cm = causal_mask(4)
    pad = torch.tensor([[False, False, False, False],
                        [False, False, False, True]])
    pm = padding_mask(pad)
    merged = combine_masks(cm, pm)

    from ppytorch.console import ModuleResult, matrix_panel
    return ModuleResult(
        "attention.masking", "因果 / 填充 / 合并 (加性 -inf)",
        info=[("因果", tuple(cm.shape)), ("填充", tuple(pm.shape)), ("合并", tuple(merged.shape))],
        checks=[
            ("因果: 第0行只能看自己",
             cm[0, 0, 0, 0] == 0 and cm[0, 0, 0, 1] == float("-inf")),
            ("填充: pad 列被屏蔽",
             pm[1, 0, 0, 3] == float("-inf") and pm[0, 0, 0, 3] == 0),
            ("合并: 因果与填充同时生效",
             merged[1, 0, 0, 3] == float("-inf") and merged[1, 0, 3, 3] == float("-inf")),
        ],
        extra=[matrix_panel(cm[0, 0], "causal_mask(4)")],
    )


if __name__ == "__main__":
    from ppytorch.console import render_report
    render_report(main())
