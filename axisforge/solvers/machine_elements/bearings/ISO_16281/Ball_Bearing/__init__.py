"""
axisforge/solvers/machine_elements/bearings/ISO_16281/Ball_Bearing/__init__.py

Public surface for the point-contact (ball bearing) ISO/TS 16281 solver
package. Mirrors the split kept by the sibling Roller_Bearing package:
solving lives in single_row_solver.py, the LOCAL library -- both the
result shapes a solve produces (BallLoadDistributionResult, the per-row
raw result; BallBearingResult, the unified per-bearing container, single-
row or multi-row through the SAME type) and the registry that holds one
BallBearingResult per bearing label (BallLoadDistributionLibrary) -- lives
together in results.py, and everything that consumes an already-solved
result lives in postprocessing.py.

Shared-displacement multi-row solver
-------------------------------------
ISO16281MultiRowBallSolverSharedDisplacement (in multirow_solver.py) is
re-exported here. NOT part of RollingBearingSolver's dispatch table
(_SOLVER_MAP / _POSTPROC_MAP in rolling_bearing_solver.py) -- call it
directly. It returns the exact same BallBearingResult type as the
single-row solver (via the same .multirow() classmethod), so nothing
downstream needs a new type import to consume its output.

"""
from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "ISO16281BallSolver",
    "debug_radial_capacity",
    "ISO16281MultiRowBallSolverSharedDisplacement",
    "BallLoadDistributionResult",
    "BallBearingResult",
    "BallLoadDistributionLibrary",
    "Q_j",
    "phi_j_global",
    "contact_distribution",
    "bearing_stiffness",
    "BallBearingStiffness",
    "DynamicEquivalentRollingElementLoad",
]

_LAZY = {
    "ISO16281BallSolver": ".single_row_solver",
    "debug_radial_capacity": ".single_row_solver",
    "ISO16281MultiRowBallSolverSharedDisplacement": ".multirow_solver",
    "BallLoadDistributionResult": ".results",
    "BallBearingResult": ".results",
    "BallLoadDistributionLibrary": ".results",
    "Q_j": ".postprocessing",
    "phi_j_global": ".postprocessing",
    "contact_distribution": ".postprocessing",
    "bearing_stiffness": ".postprocessing",
    "BallBearingStiffness": ".postprocessing",
    "DynamicEquivalentRollingElementLoad": ".postprocessing",
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
    from .single_row_solver import ISO16281BallSolver, debug_radial_capacity
    from .multirow_solver import ISO16281MultiRowBallSolverSharedDisplacement
    from .results import (
        BallLoadDistributionResult,
        BallBearingResult,
        BallLoadDistributionLibrary,
    )
    from .postprocessing import (
        Q_j,
        phi_j_global,
        contact_distribution,
        bearing_stiffness,
        BallBearingStiffness,
        DynamicEquivalentRollingElementLoad,
    )