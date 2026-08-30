"""
axisforge/solvers/machine_elements/bearings/ISO_16281/__init__.py

Public surface for the ISO/TS 16281 orchestration layer only -- dispatch.py
(solver lookup by capability + required-attrs), library.py (the per-bearing
results registry) and rolling_bearing_solver.py (RollingBearingSolver, the
single entry point that groups bearings by resolved solver and merges
results). register_contact_solver is intentionally NOT re-exported here --
it's the registration decorator used internally by
Ball_Bearing/single_row_solver.py and Roller_Bearing/single_row_solver.py
at import time; nothing outside this package registers a contact solver.

Ball_Bearing/ and Roller_Bearing/ are NOT rolled up here. They are two
different contact geometries (point vs line) with their own result types
and postprocessing -- mixing ISO16281BallSolver/BallBearingResult and
ISO16281RollerSolver/RollerBearingResult into this flat namespace would
hide which contact model a name belongs to, the same reason ISO_281 and
ISO_16281 are kept apart one level up. Import directly from
axisforge.solvers.machine_elements.bearings.ISO_16281.Ball_Bearing or
.Roller_Bearing when you need a family-specific solver instead of going
through RollingBearingSolver.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "RollingBearingSolver",
    "BearingResultsLibrary",
    "warn_if_floating_loaded",
    "resolve_solver_cls",
    "resolve_solver_cls_for_attrs",
    "SolverDispatchError",
]

_LAZY = {
    "RollingBearingSolver": ".rolling_bearing_solver",
    "BearingResultsLibrary": ".library",
    "warn_if_floating_loaded": ".library",
    "resolve_solver_cls": ".dispatch",
    "resolve_solver_cls_for_attrs": ".dispatch",
    "SolverDispatchError": ".dispatch",
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
    from .rolling_bearing_solver import RollingBearingSolver
    from .library import BearingResultsLibrary, warn_if_floating_loaded
    from .dispatch import (
        resolve_solver_cls,
        resolve_solver_cls_for_attrs,
        SolverDispatchError,
    )