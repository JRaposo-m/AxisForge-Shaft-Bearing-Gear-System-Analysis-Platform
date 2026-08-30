# families/ball_bearing/thrust/subtypes/__init__.py  (LEAF)
"""
axisforge/core/machine_elements/Bearings/families/ball_bearing/thrust/subtypes/__init__.py

Public surface: as tres BearingFamily de contacto pontual, duty thrust.
Lazy-loaded -- ver families/ (topo) para a explicacao do mecanismo.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

__all__ = ["MultiRowThrustBallFamily", "SingleRowThrustBallFamily"]

_LAZY = {
    "MultiRowThrustBallFamily": ".thrust_multirow",
    "SingleRowThrustBallFamily": ".thrust_single",
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
    from .thrust_multirow import MultiRowThrustBallFamily
    from .thrust_single import SingleRowThrustBallFamily
