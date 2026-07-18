"""attention 子系统: masking / sdpa → multihead。"""

from __future__ import annotations

from typing import Any

__all__ = [
    "naive_attention",
    "scaled_dot_product_attention",
    "causal_mask",
    "padding_mask",
    "combine_masks",
    "MultiHeadAttention",
]

_LAZY = {
    "naive_attention": (".sdpa", "naive_attention"),
    "scaled_dot_product_attention": (".sdpa", "scaled_dot_product_attention"),
    "causal_mask": (".masking", "causal_mask"),
    "padding_mask": (".masking", "padding_mask"),
    "combine_masks": (".masking", "combine_masks"),
    "MultiHeadAttention": (".multihead", "MultiHeadAttention"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from importlib import import_module
        mod_path, attr = _LAZY[name]
        value = getattr(import_module(mod_path, __name__), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
