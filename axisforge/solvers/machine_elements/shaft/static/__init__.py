
"""
axisforge/solvers/shaft/static/__init__.py

"""
from __future__ import annotations
from typing import TYPE_CHECKING

__all__ = ["BearingNodeData", "ShaftResults", "SimpleFEMResultsLibrary", "ShaftResultsReader" ]

_LAZY = {
    "BearingNodeData": ".static_analysis",
    "ShaftResults": ".static_analysis",
    "SimpleFEMResultsLibrary": ".static_analysis",
    "ShaftResultsReader": ".static_analysis",

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
    from .results_reader import BearingNodeData, ShaftResults, SimpleFEMResultsLibrary, ShaftResultsReader

