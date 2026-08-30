
"""
axisforge/solvers/shaft/FEM_solvers/__init__.py

"""
from __future__ import annotations
from typing import TYPE_CHECKING

__all__ = ["SimpleFEMSolver",
           "SubmodelResult", "SubmodelSolver", ]

_LAZY = {
    "SimpleFEMSolver": ".simple_fem_solver",
    "SubmodelResult": ".submodel_solver",
    "SubmodelSolver": ".submodel_solver"
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
    from .simple_fem_solver import SimpleFEMSolver
    from .submodel_solver import SubmodelResult, SubmodelSolver

