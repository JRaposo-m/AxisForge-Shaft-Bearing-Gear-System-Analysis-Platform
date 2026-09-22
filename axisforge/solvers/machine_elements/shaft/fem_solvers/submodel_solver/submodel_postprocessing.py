"""
axisforge/solvers/machine_elements/shaft/fem_solvers/submodel_solver/postprocessing.py

Turns a SubmodelSolution (raw DOFs from SubmodelSolver.solve(), see
lagrange_multipliers.py in this same folder) into a SubmodelResult
(results/fem_results/submodel_results.py) -- the same kind of
displacement/internal-force data ShaftResultsReader produces for the
whole shaft, just restricted to [solution.x_lo, solution.x_hi]. No
torsion, no stress concentration -- SubmodelSolver never modelled
torsion, and stress-concentration is a separate layer on top
(static_solvers/postprocessing.py) that this does not replicate.

CHANGED (this pass): replaces the previous generic SubmodelPostProcessor
+ standalone _SubmodelSolverView. Instead of one class taking a "solver"
shaped like anything, this folder now defines its own subclasses of
EulerBernoulliPostProcessing/TimoshenkoPostProcessing
(element_postprocessing.py), specialized to read a SubmodelSolution
directly:

    SubmodelEulerBernoulliPostProcessing(EulerBernoulliPostProcessing)
    SubmodelTimoshenkoPostProcessing(TimoshenkoPostProcessing)

Each inherits bending_moment()/shear_force() unchanged -- the beam-
theory math never duplicates -- and only overrides
recover_internal_forces() to translate a SubmodelSolution's own
attribute names (d_xz/d_xy/kGA_override) into the (.elements,
.d_total_xz, .d_total_xy, ._kGA_override) shape the inherited
_sweep_plane()/_element_nodal_displacements() expect, via the small
_SubmodelSolverView adapter (kept private to this module -- it's
plumbing, not part of the public shape).

postprocessor_for_submodel(solution) is the factory: checks
solution.elements[0].beam_theory once and returns the matching
subclass instance. This is the "check which subclass feeds it, then
create" step -- deliberately a plain if/elif here, not a
decorator/registry (same call made for the theory dispatch in
element_postprocessing.py and results_reader.py: not the goal of this
pass). global_solver/postprocessing.py will define the mirror pair
(GlobalEulerBernoulliPostProcessing/GlobalTimoshenkoPostProcessing) for
RigidSupportSolution when that folder is done -- those won't need a
view adapter, since rigid_support.py's own attribute names already
match what element_postprocessing.py expects.

element_postprocessing.py itself imports neither this module nor
lagrange_multipliers.py, in either direction -- it doesn't know
SubmodelSolution exists. This module is where that knowledge lives.
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
    .d_total_xy, ._kGA_override) shape the inherited
    _sweep_plane()/_element_nodal_displacements() expect. Private to
    this module -- only the two subclasses below construct one, and
    only for the duration of a single recover_internal_forces() call.
    """

    __slots__ = ("elements", "d_total_xz", "d_total_xy", "_kGA_override")

    def __init__(self, solution: "SubmodelSolution"):
        self.elements = solution.elements
        self.d_total_xz = solution.d_xz
        self.d_total_xy = solution.d_xy
        self._kGA_override = solution.kGA_override


class SubmodelEulerBernoulliPostProcessing(EulerBernoulliPostProcessing):
    """
    Euler-Bernoulli M/V recovery, specialized to a SubmodelSolution.
    bending_moment()/shear_force() are inherited unchanged from
    EulerBernoulliPostProcessing -- only the entry point that used to
    take an untyped `solver` now takes the real SubmodelSolution type
    and does the view translation itself.
    """

    def recover_internal_forces(self, solution: "SubmodelSolution", x_nodes: list[float], n: int):
        view = _SubmodelSolverView(solution)
        return super().recover_internal_forces(view, x_nodes, n)


class SubmodelTimoshenkoPostProcessing(TimoshenkoPostProcessing):
    """Timoshenko counterpart of SubmodelEulerBernoulliPostProcessing --
    same relationship, see that class's docstring."""

    def recover_internal_forces(self, solution: "SubmodelSolution", x_nodes: list[float], n: int):
        view = _SubmodelSolverView(solution)
        return super().recover_internal_forces(view, x_nodes, n)


def postprocessor_for_submodel(
    solution: "SubmodelSolution",
) -> "SubmodelEulerBernoulliPostProcessing | SubmodelTimoshenkoPostProcessing":
    """
    Picks the right subclass by reading solution.elements[0].beam_theory
    -- every Elem in one submodel solve shares the same beam_theory (set
    once from the BeamModelSettings passed into SubmodelSolver.solve()),
    same assumption ShaftResultsReader._recover_internal_forces() already
    makes for the global case.
    """
    beam_theory = solution.elements[0].beam_theory
    if beam_theory == "timoshenko":
        return SubmodelTimoshenkoPostProcessing()
    elif beam_theory == "euler_bernoulli":
        return SubmodelEulerBernoulliPostProcessing()
    raise ValueError(f"postprocessor_for_submodel: unknown beam_theory '{beam_theory}'")


def build_submodel_result(solution: "SubmodelSolution") -> SubmodelResult:
    """
    The actual entry point lagrange_multipliers.py's SubmodelSolver.solve()
    calls. Picks the right postprocessor via postprocessor_for_submodel(),
    recovers M/V through it, reads u/v/theta straight off the DOFs (no
    theory-dependent math needed for those), and assembles the
    SubmodelResult.
    """
    n = len(solution.x_nodes)
    postproc = postprocessor_for_submodel(solution)

    M_xz, M_xy, V_xz, V_xy = postproc.recover_internal_forces(solution, solution.x_nodes, n)

    # u is read only from d_xz, never d_xy -- same convention
    # ShaftResultsReader._recover_axial_and_rotation() already uses, for
    # the same reason: the axial DOF is shared between planes, but only
    # the xz system's force vector ever receives the axial load (see
    # lagrange_multipliers.py's solve(), f_xy never gets lc["axial"]) --
    # d_xy's own axial slot is solved but undriven, and would silently
    # disagree with d_xz's if an axial load falls inside [x_lo, x_hi].
    # Reading it from d_xy here would reintroduce exactly that trap.
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