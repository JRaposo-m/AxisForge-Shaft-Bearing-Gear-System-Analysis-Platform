"""
axisforge/solvers/machine_elements/shaft/fem_solvers/global_solver/global_postprocessing.py

Turns a RigidSupportSolution (raw DOFs from RigidSupportFEMSolver.solve(),
see rigid_support.py in this same folder) into a ShaftResults
(results/fem_results/shaft_results.py) for the whole shaft.

REPLACES static_solvers/results_reader.py's ShaftResultsReader class --
same job (M/V recovery + deflections + axial/rotation readback +
section properties + bearing reactions/node data + torsion, assembled
into ShaftResults), restructured as a plain function
(build_shaft_result) instead of a class holding a stashed
(solver, shaft_system) pair, to mirror
submodel_solver/postprocessing.py's build_submodel_result(). "The
results reader becomes the postprocessing" -- same job, this is its new
home and shape.

M/V recovery reuses element_theories/element_postprocessing.py's
EulerBernoulliPostProcessing/TimoshenkoPostProcessing directly, via
origin-specific subclasses (GlobalEulerBernoulliPostProcessing/
GlobalTimoshenkoPostProcessing below) -- same pattern as
submodel_solver/postprocessing.py's SubmodelEulerBernoulliPostProcessing/
SubmodelTimoshenkoPostProcessing. Unlike those, these two need NO view
adapter: RigidSupportSolution's own field names (elements, d_total_xz,
d_total_xy, _kGA_override) already match exactly what
_sweep_plane()/_element_nodal_displacements() expect -- see
RigidSupportSolution in rigid_support.py, which deliberately kept those
names (including the leading underscore on _kGA_override) for this
reason. The override bodies below are therefore pure passthroughs to
super() -- their only job is giving recover_internal_forces() a precise
RigidSupportSolution type hint instead of element_postprocessing.py's
generic Any, mirroring the submodel side's shape even though no
adaptation work is actually needed here.

Torsion lives here (via TorsionSolver, global_solver/torsion.py), not
inside rigid_support.py's solve() -- solve() stays deliberately thin
(bending + axial DOFs only, per its own docstring); torsion is an
independent static analysis, assembled into the final result at the
same point M/V recovery, bearing reactions and section properties are,
which is exactly what this module is for. If this choice turns out
wrong for how you want to use it, moving the TorsionSolver().solve()
call into rigid_support.py itself is a small, localized change -- but
note it would mean RigidSupportFEMSolver importing another class named
"...Solver", which is the situation solvers/README.md's "no solver
imports another solver" rule was written to avoid; today that call
stays outside any solver class, in this orchestration function instead.

Theory dispatch (if beam_theory == "timoshenko": ... elif ==
"euler_bernoulli": ...) is duplicated here rather than shared via a
factory function or a registry/decorator -- same deliberate choice as
submodel_solver/postprocessing.py's postprocessor_for_submodel(); see
that function's docstring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from axisforge.results.fem_results.shaft_results import BearingNodeData, ShaftResults

from axisforge.solvers.machine_elements.shaft.fem_solvers.element_theories.element_postprocessing import (
    EulerBernoulliPostProcessing,
    TimoshenkoPostProcessing,
)
from axisforge.solvers.machine_elements.shaft.fem_solvers.global_solver.torsion import TorsionSolver

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.solvers.machine_elements.shaft.fem_solvers.global_solver.rigid_support import (
        RigidSupportSolution,
    )
    from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import ShaftSystem


def _find_node(x_nodes: list[float], x: float, tol: float = 1e-6) -> int:
    for i, xi in enumerate(x_nodes):
        if abs(xi - x) < tol:
            return i
    raise ValueError(f"Position {x:.4f} mm not found in x_nodes.")


# ===========================================================================
# Theory-specific M/V recovery, specialized to RigidSupportSolution
# ===========================================================================

class GlobalEulerBernoulliPostProcessing(EulerBernoulliPostProcessing):
    """Euler-Bernoulli M/V recovery, specialized to a RigidSupportSolution.
    No view/adapter needed -- see module docstring."""

    def recover_internal_forces(self, solution: "RigidSupportSolution", x_nodes: list[float], n: int):
        return super().recover_internal_forces(solution, x_nodes, n)


class GlobalTimoshenkoPostProcessing(TimoshenkoPostProcessing):
    """Timoshenko counterpart of GlobalEulerBernoulliPostProcessing."""

    def recover_internal_forces(self, solution: "RigidSupportSolution", x_nodes: list[float], n: int):
        return super().recover_internal_forces(solution, x_nodes, n)


def postprocessor_for_global(
    solution: "RigidSupportSolution",
) -> "GlobalEulerBernoulliPostProcessing | GlobalTimoshenkoPostProcessing":
    """Mirrors submodel_solver/postprocessing.py's postprocessor_for_submodel()."""
    beam_theory = solution.elements[0].beam_theory
    if beam_theory == "timoshenko":
        return GlobalTimoshenkoPostProcessing()
    elif beam_theory == "euler_bernoulli":
        return GlobalEulerBernoulliPostProcessing()
    raise ValueError(f"postprocessor_for_global: unknown beam_theory '{beam_theory}'")


# ===========================================================================
# Full-result assembly (was ShaftResultsReader.read())
# ===========================================================================

def build_shaft_result(solution: "RigidSupportSolution", shaft_system: "ShaftSystem") -> ShaftResults:
    """
    The entry point that replaces ShaftResultsReader(solver, shaft_system).read().

    Usage
    -----
        solver   = RigidSupportFEMSolver(settings)
        solution = solver.solve(shaft_system)
        result   = build_shaft_result(solution, shaft_system)
    """
    x_nodes = solution.x_nodes
    n       = len(x_nodes)

    T_total, tau_total, phi_total, torsion_contributions = TorsionSolver().solve(shaft_system, x_nodes)

    postproc = postprocessor_for_global(solution)
    M_xz, M_xy, V_xz, V_xy = postproc.recover_internal_forces(solution, x_nodes, n)

    v_xz, v_xy             = _recover_deflections(solution, n)
    u, theta_xz, theta_xy  = _recover_axial_and_rotation(solution, n)
    d_arr, W_arr, Wt_arr   = _section_properties(shaft_system, x_nodes)

    M_eq    = np.hypot(M_xz, M_xy)
    V_eq    = np.hypot(V_xz, V_xy)
    v_eq    = np.hypot(v_xz, v_xy)
    sigma_b = M_eq / W_arr
    tau_arr = tau_total

    bearing_reactions = _bearing_reactions(solution, shaft_system, x_nodes)
    bearing_nodes     = _bearing_node_data(solution, shaft_system, x_nodes)

    R_xz_arr    = np.array([r["R_xz"]    for r in bearing_reactions])
    R_xy_arr    = np.array([r["R_xy"]    for r in bearing_reactions])
    R_arr       = np.array([r["R"]       for r in bearing_reactions])
    R_axial_arr = np.array([r["R_axial"] for r in bearing_reactions])
    brg_positions = [r["position"] for r in bearing_reactions]

    idx_M   = int(np.argmax(M_eq))
    idx_v   = int(np.argmax(v_eq))
    idx_sb  = int(np.argmax(sigma_b))
    idx_t   = int(np.argmax(np.abs(tau_arr)))
    idx_phi = int(np.argmax(np.abs(phi_total)))

    return ShaftResults(
        name = shaft_system.name,

        x_nodes  = x_nodes,
        elements = solution.elements,

        K                     = solution.K,
        free_dofs             = solution.free_dofs,
        constrained_dofs      = solution.constrained_dofs,
        d_total_xz            = solution.d_total_xz,
        d_total_xy            = solution.d_total_xy,
        f_xz_ext              = solution.f_xz_ext,
        f_xy_ext              = solution.f_xy_ext,
        f_xz_total            = solution.f_xz_total,
        f_xy_total            = solution.f_xy_total,
        f_xz_reaction         = solution.f_xz_reaction,
        f_xy_reaction         = solution.f_xy_reaction,
        T_total               = T_total,
        tau_total             = tau_total,
        phi_total             = phi_total,
        torsion_contributions = torsion_contributions,

        x        = np.array(x_nodes),
        M_xz     = M_xz,    M_xy    = M_xy,    M  = M_eq,
        V_xz     = V_xz,    V_xy    = V_xy,    V  = V_eq,
        v_xz     = v_xz,    v_xy    = v_xy,    v  = v_eq,
        u        = u,
        theta_xz = theta_xz, theta_xy = theta_xy,
        T        = T_total,
        phi      = phi_total,
        d        = d_arr,   W       = W_arr,   Wt = Wt_arr,
        sigma_b  = sigma_b, tau     = tau_arr,
        bearing_positions = brg_positions,
        R_xz     = R_xz_arr, R_xy  = R_xy_arr,
        R        = R_arr,    R_axial = R_axial_arr,
        M_max        = float(M_eq[idx_M]),      x_M_max       = float(x_nodes[idx_M]),
        v_max        = float(v_eq[idx_v]),      x_v_max       = float(x_nodes[idx_v]),
        sigma_b_max  = float(sigma_b[idx_sb]),  x_sigma_b_max = float(x_nodes[idx_sb]),
        tau_max      = float(np.abs(tau_arr[idx_t])), x_tau_max = float(x_nodes[idx_t]),
        phi_max      = float(np.abs(phi_total[idx_phi])), x_phi_max = float(x_nodes[idx_phi]),

        bearing_nodes = bearing_nodes,
    )


# ------------------------------------------------------------------
# Internal helpers -- module-level functions taking (solution,
# shaft_system) explicitly, instead of ShaftResultsReader's
# self._solver/self._sys. Logic unchanged from the previous class
# methods, only de-classed.
# ------------------------------------------------------------------

def _recover_deflections(solution: "RigidSupportSolution", n: int):
    v_xz = np.array([solution.d_total_xz[3 * i + 1] for i in range(n)])
    v_xy = np.array([solution.d_total_xy[3 * i + 1] for i in range(n)])
    return v_xz, v_xy


def _recover_axial_and_rotation(solution: "RigidSupportSolution", n: int):
    """
    Same stride-3 layout (u, v, theta per DOF-triplet) _bearing_node_data()
    below already uses per bearing node -- applied here to every node.
    u is read from d_total_xz only -- same convention documented in
    submodel_solver/postprocessing.py's build_submodel_result() (axial
    DOF shared between planes, only the xz load vector ever receives
    AxialLoad -- reading it from d_total_xy would reintroduce that trap).
    """
    u        = np.array([solution.d_total_xz[3 * i]     for i in range(n)])
    theta_xz = np.array([solution.d_total_xz[3 * i + 2] for i in range(n)])
    theta_xy = np.array([solution.d_total_xy[3 * i + 2] for i in range(n)])
    return u, theta_xz, theta_xy


def _section_properties(shaft_system: "ShaftSystem", x_nodes: list[float]):
    shaft  = shaft_system.shaft
    d_arr  = np.array([shaft.diameter_at(x) for x in x_nodes])
    W_arr  = np.array([shaft.W_at(x)        for x in x_nodes])
    Wt_arr = np.array([shaft.Wt_at(x)       for x in x_nodes])
    return d_arr, W_arr, Wt_arr


def _bearing_reactions(
    solution: "RigidSupportSolution", shaft_system: "ShaftSystem", x_nodes: list[float],
) -> list[dict]:
    reactions = []
    for b in shaft_system.bearings:
        i    = _find_node(x_nodes, b.position)
        R_xz = float(solution.f_xz_reaction[3 * i + 1])
        R_xy = float(solution.f_xy_reaction[3 * i + 1])
        R_ax = float(solution.f_xz_reaction[3 * i]) if b.arrangement == "locating" else 0.0
        reactions.append({
            "label":    b.label or b.designation,
            "position": b.position,
            "R_xz":     R_xz,
            "R_xy":     R_xy,
            "R":        float(np.hypot(R_xz, R_xy)),
            "R_axial":  R_ax,
        })
    return reactions


def _bearing_node_data(
    solution: "RigidSupportSolution", shaft_system: "ShaftSystem", x_nodes: list[float],
) -> list[BearingNodeData]:
    nodes = []

    for b in shaft_system.bearings:
        i = _find_node(x_nodes, b.position)

        u        = float(solution.d_total_xz[3 * i])
        v_xz     = float(solution.d_total_xz[3 * i + 1])
        v_xy     = float(solution.d_total_xy[3 * i + 1])
        theta_xz = float(solution.d_total_xz[3 * i + 2])
        theta_xy = float(solution.d_total_xy[3 * i + 2])

        Fr_xz = float(solution.f_xz_total[3 * i + 1])
        Fr_xy = float(solution.f_xy_total[3 * i + 1])
        Fa    = float(solution.f_xz_total[3 * i]) if b.arrangement == "locating" else 0.0
        M_xz  = float(solution.f_xz_total[3 * i + 2])
        M_xy  = float(solution.f_xy_total[3 * i + 2])

        psi_xz, psi_xy = theta_xz, theta_xy

        nodes.append(BearingNodeData(
            label    = b.label or b.designation,
            position = b.position,
            u        = u,
            v_xz     = v_xz,
            v_xy     = v_xy,
            theta_xz = theta_xz,
            theta_xy = theta_xy,
            Fr_xz    = Fr_xz,
            Fr_xy    = Fr_xy,
            Fr       = float(np.hypot(Fr_xz, Fr_xy)),
            Fa       = Fa,
            M_xz     = M_xz,
            M_xy     = M_xy,
            psi_xz   = psi_xz,
            psi_xy   = psi_xy,
        ))

    return nodes