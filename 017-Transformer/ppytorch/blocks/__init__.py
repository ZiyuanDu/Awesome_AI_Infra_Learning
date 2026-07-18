"""中间层: Encoder / Decoder 堆叠。"""

from __future__ import annotations

from typing import Any

__all__ = ["Encoder", "EncoderLayer", "Decoder", "DecoderLayer"]

_LAZY = {
    "Encoder": (".encoder", "Encoder"),
    "EncoderLayer": (".encoder", "EncoderLayer"),
    "Decoder": (".decoder", "Decoder"),
    "DecoderLayer": (".decoder", "DecoderLayer"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from importlib import import_module
        mod_path, attr = _LAZY[name]
        value = getattr(import_module(mod_path, __name__), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
