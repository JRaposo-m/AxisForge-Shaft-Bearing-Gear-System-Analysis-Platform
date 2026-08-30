# families/roller_bearing/thrust/subtypes/__init__.py  (LEAF)
"""
axisforge/mesh/shaft/element_type/timoshenko_selective_integration/__init__.py

"""
from __future__ import annotations
from typing import TYPE_CHECKING

__all__ = ["TimoshenkoBeam"]

_LAZY = {
    "TimoshenkoBeam": ".timoshenko",
}

def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        module = importlib.import_module(_LAZY[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY.keys()))

if TYPE_CHECKING:  # pragma: no cover
    from .timoshenko import TimoshenkoBeam

