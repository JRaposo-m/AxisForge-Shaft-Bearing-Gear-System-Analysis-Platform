"""
axisforge/solvers/machine_elements/shaft/fem_solvers/submodel_solver/submodel_postprocessing.py

Turns a SubmodelSolution (raw DOFs from SubmodelSolver.solve(), see
lagrange_multipliers.py in this same folder) into a SubmodelResult
(results/fem_results/submodel_results.py) -- the same kind of
displacement/internal-force data ShaftResultsReader produces for the
whole shaft, just restricted to [solution.x_lo, solution.x_hi]. No
torsion, no stress concentration -- SubmodelSolver never modelled
torsion, and stress-concentration is a separate layer on top
(static_solvers/postprocessing.py) that this does not replicate.

REPLACES the previous SubmodelConvergencePostProcessing entirely --
that class dispatched on MetricSpec.kind ("field" vs the since-removed
"moment_centroid"/"moment_point"), sampling and aggregating a handful of
named metrics via spec.eval_strategy.points()/spec.field_fn. This module
no longer imports axisforge.solvers.mesh.metric_spec at all: it just
recovers and returns everything for every node in the submodel, and
lets whoever consumes SubmodelResult (the convergence study) pick
whatever field it needs directly. This is the actual fix for the
problem that motivated the metric_spec detour in the first place --
the old convergence criterion wasn't valid for every analysis/variable
because the postprocessing step was pre-deciding what "the metric" was;
now it isn't deciding anything, it just produces the full FEM-shaped
result and pushes that decision to the caller.

M/V recovery reuses element_postprocessing.py's
EulerBernoulliPostProcessing/TimoshenkoPostProcessing directly -- the
same classes the global pipeline (ShaftResultsReader) uses -- via a
thin view adapter (_SubmodelSolverView below), so there is exactly one
place in the codebase that knows how to turn (elements, d_total_xz,
d_total_xy) into M(x)/V(x).

Theory dispatch (if beam_theory == "timoshenko": ... elif ==
"euler_bernoulli": ...) is duplicated here rather than shared via a
factory function or a registry/decorator -- deliberate, discussed and
left this way: the goal of this pass is untangling the metric_spec
dependency, not improving the dispatch mechanism. See
ShaftResultsReader._recover_internal_forces in static_solvers/results_reader.py
for the twin copy of this same if/elif.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from axisforge.results.fem_results.submodel_results import SubmodelResult

from axisforge.solvers.machine_elements.shaft.fem_solvers.element_theories.element_postprocessing import (
    EulerBernoulliPostProcessing,
    TimoshenkoPostProcessing,
)

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.solvers.machine_elements.shaft.fem_solvers.submodel_solver.lagrange_multipliers import (
        SubmodelSolution,
    )


class _SubmodelSolverView:
    """
    Adapts a SubmodelSolution to the (.elements, .d_total_xz,
    .d_total_xy, ._kGA_override) shape
    ElementTheoryPostProcessor.recover_internal_forces() expects --
    lets the submodel reuse the exact same M/V recovery code the global
    solve uses, without SubmodelSolution itself needing to carry those
    (longer, solver-flavoured) attribute names. Read-only, built fresh
    per process() call -- not meant to be kept around.
    """

    __slots__ = ("elements", "d_total_xz", "d_total_xy", "_kGA_override")

    def __init__(self, solution: "SubmodelSolution"):
        self.elements = solution.elements
        self.d_total_xz = solution.d_xz
        self.d_total_xy = solution.d_xy
        self._kGA_override = solution.kGA_override


class SubmodelPostProcessor:
    """
    process(solution) -> SubmodelResult. No shaft_system, no metrics
    list, no eval_strategy -- everything it needs is already on
    solution (elements, d_xz, d_xy, kGA_override, x_nodes).
    """

    def process(self, solution: "SubmodelSolution") -> SubmodelResult:
        n = len(solution.x_nodes)
        view = _SubmodelSolverView(solution)

        beam_theory = solution.elements[0].beam_theory
        if beam_theory == "timoshenko":
            postproc = TimoshenkoPostProcessing()
        elif beam_theory == "euler_bernoulli":
            postproc = EulerBernoulliPostProcessing()
        else:
            raise ValueError(
                f"SubmodelPostProcessor: unknown beam_theory '{beam_theory}'"
            )

        M_xz, M_xy, V_xz, V_xy = postproc.recover_internal_forces(
            view, solution.x_nodes, n,
        )

        # u is read only from d_xz, never d_xy -- same convention
        # ShaftResultsReader._recover_axial_and_rotation() already uses,
        # for the same reason: the axial DOF is shared between planes,
        # but only the xz system's force vector ever receives the axial
        # load (see lagrange_multipliers.py's solve(), f_xy never gets
        # lc["axial"]) -- d_xy's own axial slot is solved but undriven,
        # and would silently disagree with d_xz's if an axial load falls
        # inside [x_lo, x_hi]. Reading it from d_xy here would reintroduce
        # exactly that trap.
        u        = np.array([solution.d_xz[3 * i]     for i in range(n)])
        v_xz     = np.array([solution.d_xz[3 * i + 1] for i in range(n)])
        theta_xz = np.array([solution.d_xz[3 * i + 2] for i in range(n)])
        v_xy     = np.array([solution.d_xy[3 * i + 1] for i in range(n)])
        theta_xy = np.array([solution.d_xy[3 * i + 2] for i in range(n)])

        return SubmodelResult(
            x_lo=solution.x_lo,
            x_hi=solution.x_hi,
            grade=solution.grade,
            x_nodes=solution.x_nodes,
            u=u,
            v_xz=v_xz, v_xy=v_xy,
            theta_xz=theta_xz, theta_xy=theta_xy,
            M_xz=M_xz, M_xy=M_xy,
            V_xz=V_xz, V_xy=V_xy,
            lam_xz=solution.lam_xz, lam_xy=solution.lam_xy,
        )