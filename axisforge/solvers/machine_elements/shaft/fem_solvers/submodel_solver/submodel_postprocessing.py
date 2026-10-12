"""Post-processing of the sub-model solve: turns a ``SubmodelSolution`` (raw DOFs from
``SubmodelSolver.solve``) into a ``SubmodelResult`` with the same displacement and internal-force
data as the global result, restricted to [x_lo, x_hi]. No torsion and no stress concentration. The
moment and shear recovery reuses the element post-processors through a small adapter that renames
the attributes of ``SubmodelSolution``.
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
    """Adapter that gives a ``SubmodelSolution`` the attribute names read by the element post-processor.

    Maps ``elements``, ``d_xz``, ``d_xy`` and ``kGA_override`` to ``elements``, ``d_total_xz``,
    ``d_total_xy`` and ``_kGA_override``. Private to this module; built for one
    ``recover_internal_forces`` call.

    Parameters
    ----------
    solution : SubmodelSolution
        Raw sub-model solution (displacements [mm, rad], ``kGA_override`` [N]).
    """

    __slots__ = ("elements", "d_total_xz", "d_total_xy", "_kGA_override")

    def __init__(self, solution: "SubmodelSolution"):
        """Store the arguments; see the class docstring."""
        self.elements = solution.elements
        self.d_total_xz = solution.d_xz
        self.d_total_xy = solution.d_xy
        self._kGA_override = solution.kGA_override


class SubmodelEulerBernoulliPostProcessing(EulerBernoulliPostProcessing):
    """Euler-Bernoulli moment and shear recovery, typed for a ``SubmodelSolution``.

    ``bending_moment`` and ``shear_force`` are inherited unchanged; only the entry point translates
    the solution through the adapter.
    """

    def recover_internal_forces(self, solution: "SubmodelSolution", x_nodes: list[float], n: int):
        """Moment and shear force at every sub-model node (see ``ElementTheoryPostProcessor.recover_internal_forces``).

        Parameters
        ----------
        solution : SubmodelSolution
            Raw sub-model solution.
        x_nodes : list of float
            Sub-model node positions [mm].
        n : int
            Number of nodes.

        Returns
        -------
        M_xz, M_xy : ndarray, shape (n,)
            Bending moments [N mm].
        V_xz, V_xy : ndarray, shape (n,)
            Shear forces [N].
        """
        view = _SubmodelSolverView(solution)
        return super().recover_internal_forces(view, x_nodes, n)


class SubmodelTimoshenkoPostProcessing(TimoshenkoPostProcessing):
    """Timoshenko counterpart of ``SubmodelEulerBernoulliPostProcessing``.
    """

    def recover_internal_forces(self, solution: "SubmodelSolution", x_nodes: list[float], n: int):
        """Moment and shear force at every sub-model node (see ``ElementTheoryPostProcessor.recover_internal_forces``).

        Parameters
        ----------
        solution : SubmodelSolution
            Raw sub-model solution.
        x_nodes : list of float
            Sub-model node positions [mm].
        n : int
            Number of nodes.

        Returns
        -------
        M_xz, M_xy : ndarray, shape (n,)
            Bending moments [N mm].
        V_xz, V_xy : ndarray, shape (n,)
            Shear forces [N].
        """
        view = _SubmodelSolverView(solution)
        return super().recover_internal_forces(view, x_nodes, n)


def postprocessor_for_submodel(
    solution: "SubmodelSolution",
) -> "SubmodelEulerBernoulliPostProcessing | SubmodelTimoshenkoPostProcessing":
    """Select the moment/shear post-processor that matches the beam theory of the sub-model.

    Parameters
    ----------
    solution : SubmodelSolution
        Raw solution; ``solution.elements[0].beam_theory`` decides (all elements of one solve share
        the theory).

    Returns
    -------
    SubmodelEulerBernoulliPostProcessing or SubmodelTimoshenkoPostProcessing

    Raises
    ------
    ValueError
        If the beam theory is neither ``"timoshenko"`` nor ``"euler_bernoulli"``.
    """
    beam_theory = solution.elements[0].beam_theory
    if beam_theory == "timoshenko":
        return SubmodelTimoshenkoPostProcessing()
    elif beam_theory == "euler_bernoulli":
        return SubmodelEulerBernoulliPostProcessing()
    raise ValueError(f"postprocessor_for_submodel: unknown beam_theory '{beam_theory}'")


def build_submodel_result(solution: "SubmodelSolution") -> SubmodelResult:
    """Assemble the ``SubmodelResult`` from the raw sub-model solution.

    Parameters
    ----------
    solution : SubmodelSolution
        Raw output of ``SubmodelSolver.solve`` (displacements [mm, rad], multipliers [N, N mm]).

    Returns
    -------
    SubmodelResult
        Nodal ``u``, ``v``, ``theta`` per plane [mm, rad], moments [N mm], shear forces [N] and the
        multipliers ``lam_xz``, ``lam_xy``.

    Notes
    -----
    Moment and shear force come from the post-processor selected by ``postprocessor_for_submodel``;
    the displacements are read straight from the DOFs. ``u`` is read from the XZ solution only: only
    the XZ load vector receives the axial load, so the axial slot of ``d_xy`` is undriven inside the
    interval.
    """
    n = len(solution.x_nodes)
    postproc = postprocessor_for_submodel(solution)

    M_xz, M_xy, V_xz, V_xy = postproc.recover_internal_forces(solution, solution.x_nodes, n)

    # u is read only from d_xz, never d_xy -- same convention as
    # _recover_axial_and_rotation() in global_postprocessing.py uses, for
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