"""
axisforge/mesh/shaft/mesh_generation/__init__.py

"""
from __future__ import annotations
from typing import TYPE_CHECKING

__all__ = ["Mesh1D",
            "Grader",  ]

_LAZY = {
    "Mesh1D": ".mesh_1D",
    "Grader": ".mesh_grade",
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
    from .mesh_1D import Mesh1D
    from .mesh_grade import Grader

