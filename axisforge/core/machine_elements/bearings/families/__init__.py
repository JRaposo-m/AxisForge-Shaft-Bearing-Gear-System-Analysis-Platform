# families/__init__.py  (ROLLUP de ball_bearing + roller_bearing — a base para a seleção)
"""
axisforge/core/machine_elements/bearings/families/__init__.py

Roll-up de todas as BearingFamily disponiveis (ball_bearing/ e
roller_bearing/, cada um radial+thrust por baixo). E' a partir daqui que
bearing.py e' capaz de listar/selecionar familias, e mais tarde de onde
um capabilities.py para bearings vai ler -- nunca ao contrario.

Lazy-loaded: nunca importa os .py finais diretamente, reencaminha
sempre para o __init__.py do subpacote (rollup em cascata).
"""
from __future__ import annotations
from typing import TYPE_CHECKING

__all__ = [
    "DeepGrooveBallFamily", "AngularContactFamily", "SelfAligningBallFamily",
    "MultiRowThrustBallFamily", "SingleRowThrustBallFamily",
    "CylindricalRollerFamily",
    "ThrustCylindricalRollerFamily", "ThrustNeedleRollerFamily",
]

_LAZY = {
    "DeepGrooveBallFamily": ".ball_bearing",
    "AngularContactFamily": ".ball_bearing",
    "SelfAligningBallFamily": ".ball_bearing",
    "MultiRowThrustBallFamily": ".ball_bearing",
    "SingleRowThrustBallFamily": ".ball_bearing",
    "CylindricalRollerFamily": ".roller_bearing",
    "ThrustCylindricalRollerFamily": ".roller_bearing",
    "ThrustNeedleRollerFamily": ".roller_bearing",
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
    from .ball_bearing import DeepGrooveBallFamily, AngularContactFamily, SelfAligningBallFamily, MultiRowThrustBallFamily, SingleRowThrustBallFamily
    from .roller_bearing import CylindricalRollerFamily, ThrustCylindricalRollerFamily, ThrustNeedleRollerFamily