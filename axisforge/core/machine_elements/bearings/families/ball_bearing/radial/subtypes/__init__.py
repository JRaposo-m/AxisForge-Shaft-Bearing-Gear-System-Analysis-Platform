# families/ball_bearing/radial/subtypes/__init__.py  (LEAF)
"""
axisforge/core/machine_elements/bearings/families/ball_bearing/radial/subtypes/__init__.py

Public surface: as tres BearingFamily de contacto pontual, duty radial.
Lazy-loaded -- ver families/ (topo) para a explicacao do mecanismo.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

__all__ = ["DeepGrooveBallFamily", "AngularContactFamily", "SelfAligningBallFamily"]

_LAZY = {
    "DeepGrooveBallFamily": ".deep_groove_ball",
    "AngularContactFamily": ".angular_contact",
    "SelfAligningBallFamily": ".self_aligning",
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
    from .deep_groove import DeepGrooveBallFamily
    from .angular_contact import AngularContactFamily
    from .self_aligning import SelfAligningBallFamily