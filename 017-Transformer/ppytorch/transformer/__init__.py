"""顶层整机。"""

from __future__ import annotations

from typing import Any

__all__ = ["Transformer"]


def __getattr__(name: str) -> Any:
    if name == "Transformer":
        from .model import Transformer
        globals()["Transformer"] = Transformer
        return Transformer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
