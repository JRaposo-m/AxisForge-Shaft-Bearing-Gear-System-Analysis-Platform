"""
solvers/machine_elements/shaft/oneD_analysis/static/shaft_results.py

ShaftResults — data container (dataclass).
ShaftResultsReader — reads a solved SimpleFEMSolver into a ShaftResults.

Units: N, mm, N·mm, MPa, µm throughout.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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

        R_xz_arr, R_xy_arr, R_axial_arr, brg_positions = self._bearing_reactions(x_nodes)
        R_arr = np.hypot(R_xz_arr, R_xy_arr)

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

    def _recover_internal_forces(
        self, x_nodes: list[float], n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Reconstruct V(x) and M(x) by left-to-right integration of the
        external nodal force vector f_ext = K @ d (already available on
        the solver — no re-solve needed).

        f_ext[3*i + 1] is the transverse nodal force at node i (XZ or XY).
        A positive value means an upward external force at that node
        (reaction or applied load). The sign convention for V follows
        beam theory: V steps by +F at each upward force.

            V(x_0) = 0   (free left end — no force before first node)
            V(x_i) = V(x_{i-1}) - F_ext(x_i)
                     ^^ minus because f_ext contains the REACTION on the
                        structure: an upward support reaction pushes the
                        beam up, which reduces the internal shear.

            M(x_i) = M(x_{i-1}) + V(x_{i-1}) * (x_i - x_{i-1})
                     exact between nodes (no distributed load mid-element
                     by construction of Mesh1D — every load has a node).
        """
        solver = self._solver
        x      = np.array(x_nodes)

        def _integrate(f_ext_transverse: np.ndarray):
            V = np.zeros(n)
            M = np.zeros(n)
            for i in range(1, n):
                V[i] = V[i - 1] - f_ext_transverse[3 * i + 1]
                M[i] = M[i - 1] + V[i - 1] * (x[i] - x[i - 1])
            return V, M

        V_xz, M_xz = _integrate(solver.f_xz_ext)
        V_xy, M_xy = _integrate(solver.f_xy_ext)

        return M_xz, M_xy, V_xz, V_xy

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
        solver        = self._solver
        brg_positions = []
        R_xz_list, R_xy_list, R_axial_list = [], [], []

        for b in self._sys.bearings:
            i = _find_node(x_nodes, b.position)
            brg_positions.append(b.position)
            R_xz_list.append(solver.f_xz_ext[3 * i + 1])
            R_xy_list.append(solver.f_xy_ext[3 * i + 1])
            R_axial_list.append(
                solver.f_xz_ext[3 * i] if b.arrangement == "locating" else 0.0
            )

        return (
            np.array(R_xz_list),
            np.array(R_xy_list),
            np.array(R_axial_list),
            brg_positions,
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _find_node(x_nodes: list[float], x: float, tol: float = 1e-6) -> int:
    for i, xi in enumerate(x_nodes):
        if abs(xi - x) < tol:
            return i
    raise ValueError(f"Position {x:.4f} mm not found in x_nodes.")