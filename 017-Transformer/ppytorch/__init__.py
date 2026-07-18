"""
ppytorch: 教学版 Transformer (按调用层级组织)

    transformer.model
        ├── embeddings (token / positional)
        ├── blocks.encoder → attention.multihead → attention.sdpa
        │                  → layers (LN / FFN)
        ├── blocks.decoder → (同上 + 交叉注意力)
        └── attention.masking

自包上一级目录运行:
    python3 -m ppytorch.verify
    python3 -m ppytorch.attention.sdpa

也可: from ppytorch import Transformer
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "TokenEmbedding", "SinusoidalPE", "LearnedPE",
    "naive_attention", "scaled_dot_product_attention",
    "causal_mask", "padding_mask", "combine_masks",
    "MultiHeadAttention", "LayerNorm", "FeedForward",
    "Encoder", "EncoderLayer", "Decoder", "DecoderLayer", "Transformer",
]

# 惰性导出: 避免 import ppytorch 时预加载全部子模块,
# 也避免 python -m ppytorch.attention.sdpa 时的重复导入告警。
_LAZY: dict[str, tuple[str, str]] = {
    "TokenEmbedding": ("ppytorch.embeddings", "TokenEmbedding"),
    "SinusoidalPE": ("ppytorch.embeddings", "SinusoidalPE"),
    "LearnedPE": ("ppytorch.embeddings", "LearnedPE"),
    "naive_attention": ("ppytorch.attention", "naive_attention"),
    "scaled_dot_product_attention": ("ppytorch.attention", "scaled_dot_product_attention"),
    "causal_mask": ("ppytorch.attention", "causal_mask"),
    "padding_mask": ("ppytorch.attention", "padding_mask"),
    "combine_masks": ("ppytorch.attention", "combine_masks"),
    "MultiHeadAttention": ("ppytorch.attention", "MultiHeadAttention"),
    "LayerNorm": ("ppytorch.layers", "LayerNorm"),
    "FeedForward": ("ppytorch.layers", "FeedForward"),
    "Encoder": ("ppytorch.blocks", "Encoder"),
    "EncoderLayer": ("ppytorch.blocks", "EncoderLayer"),
    "Decoder": ("ppytorch.blocks", "Decoder"),
    "DecoderLayer": ("ppytorch.blocks", "DecoderLayer"),
    "Transformer": ("ppytorch.transformer", "Transformer"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib
        mod_name, attr = _LAZY[name]
        value = getattr(importlib.import_module(mod_name), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals()) + __all__)
