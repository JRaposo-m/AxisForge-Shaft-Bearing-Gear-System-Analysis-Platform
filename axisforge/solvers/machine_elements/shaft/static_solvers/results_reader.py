"""axisforge/solvers/machine_elements/shaft/static_solvers/results_reader.py"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from axisforge.results.fem_results.shaft_results import BearingNodeData, ShaftResults

from axisforge.solvers.machine_elements.shaft.fem_solvers.element_theories.timoshenko.postprocessing import (
    TimoshenkoPostProcessing,
)
from axisforge.solvers.machine_elements.shaft.fem_solvers.element_theories.euler_bernoulli.postprocessing import (
    EulerBernoulliPostProcessing,
)
from axisforge.solvers.machine_elements.shaft.static_solvers.torsion import TorsionSolver

_recover_internal_forces_timoshenko     = TimoshenkoPostProcessing().recover_internal_forces
_recover_internal_forces_euler_bernoulli = EulerBernoulliPostProcessing().recover_internal_forces
_torsion_solver = TorsionSolver()

if TYPE_CHECKING:
    from axisforge.solvers.machine_elements.shaft.fem_solvers.global_solver.rigid_support import RigidSupportFEMSolver
    from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import ShaftSystem


def _find_node(x_nodes: list[float], x: float, tol: float = 1e-6) -> int:
    for i, xi in enumerate(x_nodes):
        if abs(xi - x) < tol:
            return i
    raise ValueError(f"Position {x:.4f} mm not found in x_nodes.")


class ShaftResultsReader:
    """
    Post-processes a solved RigidSupportFEMSolver into a ShaftResults.
    Does not know libraries exist.

    Torsion is no longer part of RigidSupportFEMSolver.solve() -- it never
    needed the FEM solve at all (pure statics over shaft_system/x_nodes,
    see static_solvers/torsion.py). read() now calls TorsionSolver().solve()
    itself, the same way it already dispatches to the per-theory
    postprocessing module for M/V -- callers no longer pass
    T_total/tau_total/torsion_contributions in.

    Internal-effort recovery (M, V) is delegated to a per-beam-theory
    postprocessing module -- see
    element_theories/timoshenko/postprocessing.py and
    element_theories/euler_bernoulli/postprocessing.py. This reader keeps
    everything else: deflections, axial/rotation readback, section
    properties, bearing reactions, and bearing-node data, plus assembly
    into ShaftResults.
    """

    def __init__(self, solver: "RigidSupportFEMSolver", shaft_system: "ShaftSystem"):
        self._solver = solver
        self._sys    = shaft_system

    def read(self) -> ShaftResults:
        """
        No longer takes T_total/tau_total/torsion_contributions as
        arguments -- torsion is computed right here via TorsionSolver,
        same as M/V go through the per-theory postprocessing module.
        """
        solver  = self._solver
        x_nodes = solver.x_nodes
        n       = len(x_nodes)

        T_total, tau_total, phi_total, torsion_contributions = \
            _torsion_solver.solve(self._sys, x_nodes)

        M_xz, M_xy, V_xz, V_xy = self._recover_internal_forces(x_nodes, n)
        v_xz, v_xy              = self._recover_deflections(n)
        u, theta_xz, theta_xy   = self._recover_axial_and_rotation(n)
        d_arr, W_arr, Wt_arr    = self._section_properties(x_nodes)

        M_eq    = np.hypot(M_xz, M_xy)
        V_eq    = np.hypot(V_xz, V_xy)
        v_eq    = np.hypot(v_xz, v_xy)
        sigma_b = M_eq / W_arr
        tau_arr = tau_total

        bearing_reactions = self._bearing_reactions(x_nodes)
        bearing_nodes     = self._bearing_node_data(x_nodes)

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
            name = self._sys.name,

            x_nodes  = x_nodes,
            elements = solver.elements,

            K                     = solver.K,
            free_dofs             = solver.free_dofs,
            constrained_dofs      = solver.constrained_dofs,
            d_total_xz            = solver.d_total_xz,
            d_total_xy            = solver.d_total_xy,
            f_xz_ext              = solver.f_xz_ext,
            f_xy_ext              = solver.f_xy_ext,
            f_xz_total            = solver.f_xz_total,
            f_xy_total            = solver.f_xy_total,
            f_xz_reaction         = solver.f_xz_reaction,
            f_xy_reaction         = solver.f_xy_reaction,
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

    # -- internal helpers --

    def _recover_internal_forces(self, x_nodes, n):
        """
        Delegates to the per-theory postprocessing module, chosen by
        reading solver.elements[0].beam_theory. Every Elem built for one
        shaft shares the same beam_theory (it's set once, for the whole
        mesh, from the BeamModelSettings passed to Elem.from_mesh()/
        from_x_nodes() -- see elem.py), so the first element is
        representative of the whole solver. If a shaft could ever mix
        theories element-to-element, this would need to dispatch per
        element instead of once for the whole sweep -- not the case
        today as far as I've seen.
        """
        beam_theory = self._solver.elements[0].beam_theory
        if beam_theory == "timoshenko":
            recover = _recover_internal_forces_timoshenko
        elif beam_theory == "euler_bernoulli":
            recover = _recover_internal_forces_euler_bernoulli
        else:
            raise ValueError(f"ShaftResultsReader: unknown beam_theory '{beam_theory}'")
        return recover(self._solver, x_nodes, n)

    def _recover_deflections(self, n: int):
        solver = self._solver
        v_xz   = np.array([solver.d_total_xz[3 * i + 1] for i in range(n)])
        v_xy   = np.array([solver.d_total_xy[3 * i + 1] for i in range(n)])
        return v_xz, v_xy

    def _recover_axial_and_rotation(self, n: int):
        """
        Same stride-3 layout (u, v, theta per DOF-triplet) that
        _bearing_node_data() below already uses per bearing node --
        applied here to every node instead of only the bearing
        indices. u is read from d_total_xz for consistency with
        _bearing_node_data()'s own choice (axial DOF is shared between
        planes in this element formulation, so d_total_xz[3*i] and
        d_total_xy[3*i] agree by construction -- reading either is
        equivalent).
        """
        solver   = self._solver
        u        = np.array([solver.d_total_xz[3 * i]     for i in range(n)])
        theta_xz = np.array([solver.d_total_xz[3 * i + 2] for i in range(n)])
        theta_xy = np.array([solver.d_total_xy[3 * i + 2] for i in range(n)])
        return u, theta_xz, theta_xy

    def _section_properties(self, x_nodes: list[float]):
        shaft  = self._sys.shaft
        d_arr  = np.array([shaft.diameter_at(x) for x in x_nodes])
        W_arr  = np.array([shaft.W_at(x)        for x in x_nodes])
        Wt_arr = np.array([shaft.Wt_at(x)       for x in x_nodes])
        return d_arr, W_arr, Wt_arr

    def _bearing_reactions(self, x_nodes: list[float]) -> list[dict]:
        solver    = self._solver
        reactions = []
        for b in self._sys.bearings:
            i    = _find_node(x_nodes, b.position)
            R_xz = float(solver.f_xz_reaction[3 * i + 1])
            R_xy = float(solver.f_xy_reaction[3 * i + 1])
            R_ax = float(solver.f_xz_reaction[3 * i]) if b.arrangement == "locating" else 0.0
            reactions.append({
                "label":    b.label or b.designation,
                "position": b.position,
                "R_xz":     R_xz,
                "R_xy":     R_xy,
                "R":        float(np.hypot(R_xz, R_xy)),
                "R_axial":  R_ax,
            })
        return reactions

    def _bearing_node_data(self, x_nodes: list[float]) -> list[BearingNodeData]:
        solver = self._solver
        nodes  = []

        for b in self._sys.bearings:
            i = _find_node(x_nodes, b.position)

            u        = float(solver.d_total_xz[3 * i])
            v_xz     = float(solver.d_total_xz[3 * i + 1])
            v_xy     = float(solver.d_total_xy[3 * i + 1])
            theta_xz = float(solver.d_total_xz[3 * i + 2])
            theta_xy = float(solver.d_total_xy[3 * i + 2])

            Fr_xz = float(solver.f_xz_total[3 * i + 1])
            Fr_xy = float(solver.f_xy_total[3 * i + 1])
            Fa    = float(solver.f_xz_total[3 * i]) if b.arrangement == "locating" else 0.0
            M_xz  = float(solver.f_xz_total[3 * i + 2])
            M_xy  = float(solver.f_xy_total[3 * i + 2])

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
