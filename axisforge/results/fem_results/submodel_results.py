"""
axisforge/results/fem_results/submodel_results.py

Pure-data, POST-postprocessing result for one submodel solve (one grade,
one interval). Built by SubmodelPostProcessor.process()
(fem_solvers/submodel_solver/postprocessing.py) FROM a SubmodelSolution
(fem_solvers/submodel_solver/lagrange_multipliers.py's raw solver
output) -- never built directly by SubmodelSolver.solve() itself.
Mirrors the split already used for the full-shaft pipeline:
RigidSupportFEMSolver (solver) -> Timoshenko/EulerBernoulliPostProcessing
(postprocessing) -> ShaftResults (results/); this is that same split one
level down, for a single submodel grade instead of a whole shaft.

CHANGED (this pass): this used to hold `metric_values: dict[str, float]`
-- a handful of named, pre-aggregated numbers (v_xz/v_xy/v_res, produced
via the now-removed MetricSpec/eval_strategy machinery). It now holds
the full per-node arrays instead (u, v_xz, v_xy, theta_xz, theta_xy,
M_xz, M_xy, V_xz, V_xy, plus the Lagrange reactions at the cut nodes) --
the same shape ShaftResults exposes for the whole shaft, just restricted
to [x_lo, x_hi]. Whoever consumes this now (the convergence study) picks
whatever field/point it needs directly from these arrays instead of the
postprocessing step pre-deciding a fixed set of named metrics. No
torsion, no stress concentration -- SubmodelSolver never modelled
torsion, and stress-concentration is a layer on top of a full ShaftResults,
not something this submodel path replicates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["SubmodelResult"]


def _empty() -> np.ndarray:
    return np.array([])


@dataclass
class SubmodelResult:
    """
    One grade's postprocessed FEM data for one interval.

    x_lo, x_hi, grade : identity, carried over from the SubmodelSolution
        this was built from.
    x_nodes : node positions of the submodel mesh at this grade -- kept
        because RichardsonGCI needs node COUNT per grade
        (x_nodes_coarse/medium/fine) to compute the refinement ratio r,
        not just any single field value.
    u, v_xz, v_xy, theta_xz, theta_xy : displacement/rotation per node,
        read straight off the solved DOFs -- no recovery math involved.
    M_xz, M_xy, V_xz, V_xy : internal-effort arrays, recovered via
        element_postprocessing.py's EulerBernoulliPostProcessing/
        TimoshenkoPostProcessing (same classes the global pipeline uses).
    lam_xz, lam_xy : Lagrange-multiplier reactions at the two cut nodes
        -- not physical shaft quantities, but useful for sanity-checking
        the submodel BC coupling itself (e.g. should shrink toward the
        cut-node reaction implied by the global solve as the submodel
        widens).
    """
    x_lo: float
    x_hi: float
    grade: str
    x_nodes: list[float] = field(default_factory=list)

    u: np.ndarray = field(default_factory=_empty)
    v_xz: np.ndarray = field(default_factory=_empty)
    v_xy: np.ndarray = field(default_factory=_empty)
    theta_xz: np.ndarray = field(default_factory=_empty)
    theta_xy: np.ndarray = field(default_factory=_empty)

    M_xz: np.ndarray = field(default_factory=_empty)
    M_xy: np.ndarray = field(default_factory=_empty)
    V_xz: np.ndarray = field(default_factory=_empty)
    V_xy: np.ndarray = field(default_factory=_empty)

    lam_xz: np.ndarray = field(default_factory=_empty)
    lam_xy: np.ndarray = field(default_factory=_empty)