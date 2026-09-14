"""axisforge/solvers/machine_elements/shaft/oneD_analysis/static/results_reader.py"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from axisforge.core.loads import LoadPlane
from axisforge.results.fem_results.shaft_results import BearingNodeData, ShaftResults

if TYPE_CHECKING:
    from axisforge.solvers.machine_elements.shaft.fem_solvers.rigid_support import RigidSupportFEMSolver
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

    Torsion is no longer part of RigidSupportFEMSolver.solve() -- call
    static_solvers/torsion.py::solve_torsion(shaft_system, solver.x_nodes)
    yourself and pass its three return values into read() explicitly.
    This class never calls solve_torsion() itself.
    """

    def __init__(self, solver: "RigidSupportFEMSolver", shaft_system: "ShaftSystem"):
        self._solver = solver
        self._sys    = shaft_system

    def read(self,
              T_total: np.ndarray,
              tau_total: np.ndarray,
              torsion_contributions: list[list[dict]]) -> ShaftResults:
        """
        T_total, tau_total, torsion_contributions: the three return
        values of static_solvers/torsion.py::solve_torsion(shaft_system,
        solver.x_nodes), computed by the caller and handed in here --
        this reader does not compute torsion itself.
        """
        solver  = self._solver
        x_nodes = solver.x_nodes
        n       = len(x_nodes)

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

        idx_M  = int(np.argmax(M_eq))
        idx_v  = int(np.argmax(v_eq))
        idx_sb = int(np.argmax(sigma_b))
        idx_t  = int(np.argmax(np.abs(tau_arr)))

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
            torsion_contributions = torsion_contributions,

            x        = np.array(x_nodes),
            M_xz     = M_xz,    M_xy    = M_xy,    M  = M_eq,
            V_xz     = V_xz,    V_xy    = V_xy,    V  = V_eq,
            v_xz     = v_xz,    v_xy    = v_xy,    v  = v_eq,
            u        = u,
            theta_xz = theta_xz, theta_xy = theta_xy,
            T        = T_total,
            d        = d_arr,   W       = W_arr,   Wt = Wt_arr,
            sigma_b  = sigma_b, tau     = tau_arr,
            bearing_positions = brg_positions,
            R_xz     = R_xz_arr, R_xy  = R_xy_arr,
            R        = R_arr,    R_axial = R_axial_arr,
            M_max        = float(M_eq[idx_M]),      x_M_max       = float(x_nodes[idx_M]),
            v_max        = float(v_eq[idx_v]),      x_v_max       = float(x_nodes[idx_v]),
            sigma_b_max  = float(sigma_b[idx_sb]),  x_sigma_b_max = float(x_nodes[idx_sb]),
            tau_max      = float(np.abs(tau_arr[idx_t])), x_tau_max = float(x_nodes[idx_t]),

            bearing_nodes = bearing_nodes,
        )

    # -- internal helpers --

    def _build_load_function(self, plane: LoadPlane) -> list[dict]:
        segments = []
        for ld in self._sys.distributed_radial_loads:
            if plane == LoadPlane.XY:
                if ld._theta_variable:
                    fn = lambda x, _ld=ld: _ld._q_fn(x) * np.cos(np.radians(_ld._theta_fn(x)))
                else:
                    fn = lambda x, _ld=ld: _ld._q_fn(x) * np.cos(_ld._theta_fn(x))
            else:
                if ld._theta_variable:
                    fn = lambda x, _ld=ld: _ld._q_fn(x) * np.sin(np.radians(_ld._theta_fn(x)))
                else:
                    fn = lambda x, _ld=ld: _ld._q_fn(x) * np.sin(_ld._theta_fn(x))
            segments.append({
                "label":  ld.label,
                "source": ld.source,
                "x_lo":   ld.x_lo,
                "x_hi":   ld.x_hi,
                "fn":     fn,
            })
        return segments

    def _recover_internal_forces(self, x_nodes, n):
        V_xz, M_xz = self._sweep_plane_from_elements(LoadPlane.XZ)
        V_xy, M_xy = self._sweep_plane_from_elements(LoadPlane.XY)
        return M_xz, M_xy, V_xz, V_xy

    def _sweep_plane_from_elements(self, plane: LoadPlane):
        """
        FIX (this pass): the previous version built a 6-entry local dof
        vector (u, v, theta per node) and multiplied it by k_e from
        self._beam.stiffness_element(elem, ...) -- but that k_e was
        always the 4x4 BENDING-ONLY matrix ([v_a, th_a, v_b, th_b]).
        4x4 @ length-6 doesn't even multiply -- this raised immediately
        whenever called. Axial and bending are decoupled (see
        frame_element.py / build_stiffness_matrix.py), so recovering
        V/M never needed the axial dofs at all -- fixed to use the
        bending-only 4-slot local vector, matching what
        elem.stiffness_element() actually returns.

        Also: self._beam (an explicit EulerBernoulliBeam/TimoshenkoBeam
        instance held by the reader, mirroring the builder's old
        StiffnessMatrixBuilder.beam) no longer exists -- Elem dispatches
        its own stiffness_element() by beam_theory now, and already
        knows its own shear_theory (fixed at construction via
        BeamModelSettings), so there is no shear_theory to pass in here
        either. kGA_override is still forwarded explicitly, since it is
        a per-solve override rather than something Elem was built
        with -- elem.stiffness_element() itself ignores it for
        euler_bernoulli elements, same guard used everywhere else.
        """
        solver   = self._solver
        elements = solver.elements
        x_nodes  = solver.x_nodes
        d_total  = solver.d_total_xz if plane == LoadPlane.XZ else solver.d_total_xy

        n = len(x_nodes)
        V = np.zeros(n)
        M = np.zeros(n)

        for elem_idx, elem in enumerate(elements):
            k_e = elem.stiffness_element(kGA_override=solver._kGA_override)  # 4x4, [v_a, th_a, v_b, th_b]
            i, j = elem.idx_node_1, elem.idx_node_2
            dof = [3 * i + 1, 3 * i + 2, 3 * j + 1, 3 * j + 2]
            d_e = d_total[dof]
            f_e = k_e @ d_e

            V1, M1 = -f_e[0], -f_e[1]
            V2, M2 =  f_e[2],  f_e[3]

            if elem_idx == 0:
                V[i] = V1
                M[i] = M1
            V[j] = V2
            M[j] = M2

        return V, M

    @staticmethod
    def _q_total(segments: list[dict], x: float) -> float:
        total = 0.0
        for seg in segments:
            if seg["x_lo"] - 1e-9 <= x <= seg["x_hi"] + 1e-9:
                total += seg["fn"](x)
        return total

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

            psi_xz, psi_xy = self._psi(x_nodes, solver, b)

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

    def _psi(self, x_nodes, solver, bearing):
        x_lo, x_hi = self._sys.bearing_extent(bearing)
        span = x_hi - x_lo

        if span <= 0.0:
            i = _find_node(x_nodes, bearing.position)
            return (float(solver.d_total_xz[3 * i + 2]),
                    float(solver.d_total_xy[3 * i + 2]))

        i_lo = _find_node(x_nodes, x_lo)
        i_hi = _find_node(x_nodes, x_hi)
        psi_xz = float((solver.d_total_xz[3 * i_hi + 1] - solver.d_total_xz[3 * i_lo + 1]) / span)
        psi_xy = float((solver.d_total_xy[3 * i_hi + 1] - solver.d_total_xy[3 * i_lo + 1]) / span)
        return psi_xz, psi_xy