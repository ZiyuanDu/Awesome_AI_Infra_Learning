"""输入侧: TokenEmbedding + 位置编码。"""

from __future__ import annotations

from typing import Any

__all__ = ["TokenEmbedding", "SinusoidalPE", "LearnedPE"]

_LAZY = {
    "TokenEmbedding": (".token", "TokenEmbedding"),
    "SinusoidalPE": (".positional", "SinusoidalPE"),
    "LearnedPE": (".positional", "LearnedPE"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from importlib import import_module
        mod_path, attr = _LAZY[name]
        value = getattr(import_module(mod_path, __name__), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
