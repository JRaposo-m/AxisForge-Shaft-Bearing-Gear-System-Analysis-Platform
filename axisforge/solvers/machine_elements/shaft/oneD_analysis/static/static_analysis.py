"""
axisforge/solvers/machine_elements/shaft/oneD_analysis/static/static_analysis.py

Read results from the simple_fem_solver and obtain the internal forces 
    post processing - in which it shall take into consideration the stress concentraction factors

Aplly static failure criteria for the shafts

"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from axisforge.core.loads import LoadPlane


if TYPE_CHECKING:
    from axisforge.solvers.machine_elements.shaft.oneD_analysis.FEM_solvers.simple_fem_solver import SimpleFEMSolver
    from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.shaft_system import ShaftSystem


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class ShaftResults:
    """
    All node-aligned arrays extracted from a solved SimpleFEMSolver.

    Arrays  (n = len(x_nodes))
    --------------------------
    x        [mm]     node positions
    M_xz     [N·mm]   bending moment, XZ plane
    M_xy     [N·mm]   bending moment, XY plane
    M        [N·mm]   resultant moment sqrt(M_xz²+M_xy²)
    V_xz     [N]      shear force, XZ plane
    V_xy     [N]      shear force, XY plane
    V        [N]      resultant shear sqrt(V_xz²+V_xy²)
    v_xz     [µm]     lateral deflection, XZ plane
    v_xy     [µm]     lateral deflection, XY plane
    v        [µm]     resultant deflection sqrt(v_xz²+v_xy²)
    T        [N·m]    torsion diagram
    d        [mm]     section diameter at each node
    W        [mm³]    section modulus at each node
    Wt       [mm³]    polar section modulus at each node
    sigma_b  [MPa]    bending stress M/W
    tau      [MPa]    torsional shear stress T/Wt

    Bearing reactions (from FEM reaction vector f = K·d)
    ----------------------------------------------------
    bearing_positions  [mm]   axial position of each bearing
    R_xz               [N]    radial reaction XZ per bearing
    R_xy               [N]    radial reaction XY per bearing
    R                  [N]    resultant radial reaction per bearing
    R_axial            [N]    axial reaction per bearing (locating only)
    """

    name:    str = ""

    x:       np.ndarray = field(default_factory=lambda: np.array([]))

    M_xz:    np.ndarray = field(default_factory=lambda: np.array([]))
    M_xy:    np.ndarray = field(default_factory=lambda: np.array([]))
    M:       np.ndarray = field(default_factory=lambda: np.array([]))

    V_xz:    np.ndarray = field(default_factory=lambda: np.array([]))
    V_xy:    np.ndarray = field(default_factory=lambda: np.array([]))
    V:       np.ndarray = field(default_factory=lambda: np.array([]))

    v_xz:    np.ndarray = field(default_factory=lambda: np.array([]))
    v_xy:    np.ndarray = field(default_factory=lambda: np.array([]))
    v:       np.ndarray = field(default_factory=lambda: np.array([]))

    T:       np.ndarray = field(default_factory=lambda: np.array([]))

    d:       np.ndarray = field(default_factory=lambda: np.array([]))
    W:       np.ndarray = field(default_factory=lambda: np.array([]))
    Wt:      np.ndarray = field(default_factory=lambda: np.array([]))

    sigma_b: np.ndarray = field(default_factory=lambda: np.array([]))
    tau:     np.ndarray = field(default_factory=lambda: np.array([]))

    bearing_positions: list[float] = field(default_factory=list)
    R_xz:    np.ndarray = field(default_factory=lambda: np.array([]))
    R_xy:    np.ndarray = field(default_factory=lambda: np.array([]))
    R:       np.ndarray = field(default_factory=lambda: np.array([]))
    R_axial: np.ndarray = field(default_factory=lambda: np.array([]))

    M_max:         float = 0.0
    x_M_max:       float = 0.0
    v_max:         float = 0.0
    x_v_max:       float = 0.0
    sigma_b_max:   float = 0.0
    x_sigma_b_max: float = 0.0
    tau_max:       float = 0.0
    x_tau_max:     float = 0.0


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

class ShaftResultsReader:
    """
    Reads a solved SimpleFEMSolver into a ShaftResults.

    Usage
    -----
        reader  = ShaftResultsReader(solver, shaft_system)
        results = reader.read()

    The beam element object is taken directly from the solver's builder
    so no second instantiation is needed.
    """

    def __init__(self, solver: "SimpleFEMSolver", shaft_system: "ShaftSystem"):
        self._solver  = solver
        self._sys     = shaft_system
        self._beam    = solver._builder.beam

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def read(self) -> ShaftResults:
        solver  = self._solver
        x_nodes = solver.x_nodes
        n       = len(x_nodes)

        M_xz, M_xy, V_xz, V_xy = self._recover_internal_forces(x_nodes, n)
        v_xz, v_xy              = self._recover_deflections(n)
        d_arr, W_arr, Wt_arr    = self._section_properties(x_nodes)

        M_eq    = np.hypot(M_xz, M_xy)
        V_eq    = np.hypot(V_xz, V_xy)
        v_eq    = np.hypot(v_xz, v_xy)
        sigma_b = M_eq / W_arr
        tau_arr = solver.tau_total          # MPa, from SimpleFEMSolver

        bearing_reactions = self._bearing_reactions(x_nodes)

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
            name    = self._sys.name,
            x       = np.array(x_nodes),
            M_xz    = M_xz,    M_xy    = M_xy,    M       = M_eq,
            V_xz    = V_xz,    V_xy    = V_xy,    V       = V_eq,
            v_xz    = v_xz,    v_xy    = v_xy,    v       = v_eq,
            T       = solver.T_total,
            d       = d_arr,   W       = W_arr,   Wt      = Wt_arr,
            sigma_b = sigma_b, tau     = tau_arr,
            bearing_positions = brg_positions,
            R_xz    = R_xz_arr, R_xy  = R_xy_arr,
            R       = R_arr,    R_axial = R_axial_arr,
            M_max        = float(M_eq[idx_M]),      x_M_max       = float(x_nodes[idx_M]),
            v_max        = float(v_eq[idx_v]),      x_v_max       = float(x_nodes[idx_v]),
            sigma_b_max  = float(sigma_b[idx_sb]),  x_sigma_b_max = float(x_nodes[idx_sb]),
            tau_max      = float(np.abs(tau_arr[idx_t])), x_tau_max = float(x_nodes[idx_t]),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------


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

# ------------------------------------------------------------------
    # Internal forces — recovered directly from element end forces
    # ------------------------------------------------------------------

    def _recover_internal_forces(
        self, x_nodes: list[float], n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Recovers M_xz, M_xy, V_xz, V_xy node arrays directly from element
        end forces: f_e = k_e @ d_e, per element.

        Elements with NO distributed load overlapping them are exact with
        just the two nodal values (V constant, M linear inside).

        Elements that DO overlap a distributed load get an additional
        refined intra-element reconstruction (see _refine_element) since
        the interior of the element can hold the true M_max / sigma_b_max,
        not just the nodes.

        Returns node-aligned M_xz, M_xy, V_xz, V_xy. The refined
        intra-element data is stashed on self._last_refined_xz /
        self._last_refined_xy for read() to consult when computing
        M_max / sigma_b_max (not yet wired in read() — left for you to
        review first, as discussed).
        """
        V_xz, M_xz = self._sweep_plane_from_elements(LoadPlane.XZ)
        V_xy, M_xy = self._sweep_plane_from_elements(LoadPlane.XY)


        return M_xz, M_xy, V_xz, V_xy

    # ------------------------------------------------------------------
    # Per-plane recovery from element end forces (+ distributed refinement)
    # ------------------------------------------------------------------

    def _sweep_plane_from_elements(
        self, plane: "LoadPlane"
    ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        solver   = self._solver
        elements = solver.elements
        x_nodes  = solver.x_nodes
        d_total  = solver.d_total_xz if plane == LoadPlane.XZ else solver.d_total_xy
        segments = self._build_load_function(plane)

        n = len(x_nodes)
        V = np.zeros(n)
        M = np.zeros(n)


        for elem_idx, elem in enumerate(elements):
            k_e = self._beam.stiffness_element(elem)

            dof = [
                3 * elem.idx_node_1,     3 * elem.idx_node_1 + 1, 3 * elem.idx_node_1 + 2,
                3 * elem.idx_node_2,     3 * elem.idx_node_2 + 1, 3 * elem.idx_node_2 + 2,
            ]
            d_e = d_total[dof]
            f_e = k_e @ d_e

            # convenção: f_e = {-P1,-V1,-M1, P2,V2,M2}
            V1, M1 = -f_e[1], -f_e[2]
            V2, M2 =  f_e[4],  f_e[5]

            if elem_idx == 0:
                V[elem.idx_node_1] = V1
                M[elem.idx_node_1] = M1
            V[elem.idx_node_2] = V2
            M[elem.idx_node_2] = M2

        return V, M

    # ------------------------------------------------------------------
    # Distributed-load lookup helpers (shared by _refine_element)
    # ------------------------------------------------------------------

    @staticmethod
    def _q_total(segments: list[dict], x: float) -> float:
        total = 0.0
        for seg in segments:
            if seg["x_lo"] - 1e-9 <= x <= seg["x_hi"] + 1e-9:
                total += seg["fn"](x)
        return total


    def _recover_deflections(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        solver = self._solver
        v_xz   = np.array([solver.d_total_xz[3 * i + 1] for i in range(n)]) * 1e3
        v_xy   = np.array([solver.d_total_xy[3 * i + 1] for i in range(n)]) * 1e3
        return v_xz, v_xy

    def _section_properties(
        self, x_nodes: list[float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shaft  = self._sys.shaft
        d_arr  = np.array([shaft.diameter_at(x) for x in x_nodes])
        W_arr  = np.array([shaft.W_at(x)        for x in x_nodes])
        Wt_arr = np.array([shaft.Wt_at(x)       for x in x_nodes])
        return d_arr, W_arr, Wt_arr

    def _bearing_reactions(
        self, x_nodes: list[float]
    ) -> list[dict]:
        """
        Radial + axial bearing reactions, read directly from the FEM
        reaction vectors f_xz_reaction / f_xy_reaction (f = K·d, minus
        the externally applied load).

        Returns one dict per bearing, in the same order as
        self._sys.bearings (sorted by axial position):

            {"label": str, "position": float,
             "R_xz": float, "R_xy": float, "R": float, "R_axial": float}
        """
        solver = self._solver
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


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _find_node(x_nodes: list[float], x: float, tol: float = 1e-6) -> int:
    for i, xi in enumerate(x_nodes):
        if abs(xi - x) < tol:
            return i
    raise ValueError(f"Position {x:.4f} mm not found in x_nodes.")


class StaticFailure:

    """
    In this class we will analyse the different failure criteria for shafts
    according to it's material and choosen criteria
    """
    def __init__(self):
        pass