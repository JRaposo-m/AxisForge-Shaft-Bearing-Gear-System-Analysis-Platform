# families/roller_bearing/__init__.py  (ROLLUP de radial + thrust)
"""
axisforge/core/machine_elements/bearings/families/roller_bearing/__init__.py
"""
from __future__ import annotations
from typing import TYPE_CHECKING

__all__ = ["CylindricalRollerFamily", "ThrustCylindricalRollerFamily", "MultiRowThrustCylindricalRollerFamily", "ThrustNeedleRollerFamily"]

_LAZY = {
    "CylindricalRollerFamily": ".radial",
    "ThrustCylindricalRollerFamily": ".thrust",
    "MultiRowThrustCylindricalRollerFamily": ".thrust",
    "ThrustNeedleRollerFamily": ".thrust",
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
    from .radial import CylindricalRollerFamily
    from .thrust import ThrustCylindricalRollerFamily, MultiRowThrustCylindricalRollerFamily, ThrustNeedleRollerFamily
