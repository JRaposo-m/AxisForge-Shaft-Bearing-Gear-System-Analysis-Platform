"""
axisforge/solvers/machine_elements/shaft/oneD_analysis/static/static_analysis.py

Read results from the simple_fem_solver and obtain the internal forces 
    post processing - in which it shall take into consideration the stress concentraction factors

Apply static failure criteria for the shafts

"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

from axisforge.core.loads import LoadPlane


if TYPE_CHECKING:
    from axisforge.solvers.machine_elements.shaft.oneD_analysis.FEM_solvers.simple_fem_solver import SimpleFEMSolver
    from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.shaft_system import ShaftSystem


# ---------------------------------------------------------------------------
# BearingNodeData — raw FEM quantities at each bearing position
# ---------------------------------------------------------------------------

@dataclass
class BearingNodeData:
    """
    Complete FEM nodal state at a bearing position, extracted directly
    from SimpleFEMSolver arrays.

    Consumed by downstream solvers that need raw FEM quantities:
      - ISO_16281_ball_bearing  : Fr_xz, Fr_xy, Fa, psi_xz, psi_xy, u, v_xz, v_xy
      - Newton-Raphson coupling : theta_xz, theta_xy, M_xz, M_xy for BC injection

    psi_xz / psi_xy
    ---------------
    Shaft centreline slope across the bearing seat width [rad].
    If seat width > 0  -> gradient of v across [x_lo, x_hi].
    If seat width == 0 -> nodal theta at bearing position (fallback).
    Computed by ShaftResultsReader during read(); no second FEM pass needed.

    Units
    -----
    Displacements : mm
    Rotations     : rad
    Forces        : N
    Moments       : N·mm
    """
    label:     str    = ""
    position:  float  = 0.0    # [mm]

    # --- displacements ---
    u:         float  = 0.0    # axial [mm]
    v_xz:      float  = 0.0    # transverse XZ [mm]
    v_xy:      float  = 0.0    # transverse XY [mm]
    theta_xz:  float  = 0.0    # bending rotation XZ [rad]
    theta_xy:  float  = 0.0    # bending rotation XY [rad]

    # --- reactions (f = K·d, at constrained DOFs) ---
    Fr_xz:     float  = 0.0    # radial reaction XZ [N]
    Fr_xy:     float  = 0.0    # radial reaction XY [N]
    Fr:        float  = 0.0    # resultant radial [N]
    Fa:        float  = 0.0    # axial reaction [N]   (locating only; 0 for floating)
    M_xz:      float  = 0.0    # moment reaction XZ [N·mm]
    M_xy:      float  = 0.0    # moment reaction XY [N·mm]

    # --- misalignment across seat ---
    psi_xz:    float  = 0.0    # slope XZ [rad]
    psi_xy:    float  = 0.0    # slope XY [rad]


# ---------------------------------------------------------------------------
# ShaftResults — complete FEM solution + post-processed engineering quantities
# ---------------------------------------------------------------------------

@dataclass
class ShaftResults:
    """
    Complete output of SimpleFEMSolver + ShaftResultsReader for one ShaftSystem.

    Organised in four sections:

    1. MESH
       Node positions and element connectivity from Mesh1D / Elem.from_mesh.

    2. FEM SOLUTION — raw arrays
       Global displacement vectors, global force vectors, reaction vectors,
       stiffness matrix, DOF index lists.
       These are the quantities downstream solvers (ISO 16281, NR coupling,
       convergence studies) need without re-running the FEM.

    3. POST-PROCESSED ENGINEERING QUANTITIES
       Internal forces, deflections, section properties, stresses —
       node-aligned arrays derived from the raw FEM solution.

    4. BEARING NODE DATA
       Per-bearing nodal state: displacements, reactions, misalignment.
       Consumed directly by ISO_16281_ball_bearing and NR coupling.

    Arrays  (n = len(x_nodes))
    --------------------------
    x        [mm]     node positions
    M_xz     [N·mm]   bending moment, XZ plane
    M_xy     [N·mm]   bending moment, XY plane
    M        [N·mm]   resultant moment sqrt(M_xz²+M_xy²)
    V_xz     [N]      shear force, XZ plane
    V_xy     [N]      shear force, XY plane
    V        [N]      resultant shear sqrt(V_xz²+V_xy²)
    v_xz     [mm]     lateral deflection, XZ plane
    v_xy     [mm]     lateral deflection, XY plane
    v        [mm]     resultant deflection sqrt(v_xz²+v_xy²)
    T        [N·m]    torsion diagram
    d        [mm]     section diameter at each node
    W        [mm³]    section modulus at each node
    Wt       [mm³]    polar section modulus at each node
    sigma_b  [MPa]    bending stress M/W
    tau      [MPa]    torsional shear stress T/Wt

    Bearing reactions (summary arrays, same order as bearing_nodes)
    ---------------------------------------------------------------
    bearing_positions  [mm]   axial position of each bearing
    R_xz               [N]    radial reaction XZ per bearing
    R_xy               [N]    radial reaction XY per bearing
    R                  [N]    resultant radial reaction per bearing
    R_axial            [N]    axial reaction per bearing (locating only)
    """

    name: str = ""

    # ------------------------------------------------------------------
    # 1. MESH
    # ------------------------------------------------------------------

    x_nodes:   list[float]  = field(default_factory=list)
    """Node positions [mm], len = n."""

    elements:  list         = field(default_factory=list)
    """Elem instances from Elem.from_mesh(). Carries length, E, I, A, v, node indices."""

    # ------------------------------------------------------------------
    # 2. FEM SOLUTION — raw
    # ------------------------------------------------------------------

    K:              np.ndarray = field(default_factory=lambda: np.array([]))
    """Global stiffness matrix K [N/mm or N·mm/rad], shape (3n, 3n)."""

    free_dofs:       list[int] = field(default_factory=list)
    """Indices of unconstrained DOFs in the global system."""

    constrained_dofs: list[int] = field(default_factory=list)
    """Indices of constrained DOFs (bearing transverse + locating axial)."""

    d_total_xz:    np.ndarray = field(default_factory=lambda: np.array([]))
    """Global displacement vector — XZ plane, shape (3n,). [u[mm], v[mm], theta[rad]]"""

    d_total_xy:    np.ndarray = field(default_factory=lambda: np.array([]))
    """Global displacement vector — XY plane, shape (3n,)."""

    f_xz_ext:      np.ndarray = field(default_factory=lambda: np.array([]))
    """Assembled external load vector — XZ plane, shape (3n,) [N, N, N·mm]."""

    f_xy_ext:      np.ndarray = field(default_factory=lambda: np.array([]))
    """Assembled external load vector — XY plane, shape (3n,)."""

    f_xz_total:    np.ndarray = field(default_factory=lambda: np.array([]))
    """Total nodal force vector — XZ plane (K·d), shape (3n,)."""

    f_xy_total:    np.ndarray = field(default_factory=lambda: np.array([]))
    """Total nodal force vector — XY plane (K·d), shape (3n,)."""

    f_xz_reaction: np.ndarray = field(default_factory=lambda: np.array([]))
    """Reaction force vector — XZ plane, shape (3n,). Non-zero only at constrained DOFs."""

    f_xy_reaction: np.ndarray = field(default_factory=lambda: np.array([]))
    """Reaction force vector — XY plane, shape (3n,)."""

    T_total:       np.ndarray = field(default_factory=lambda: np.array([]))
    """Cumulative torsion diagram T(x) [N·m], shape (n,)."""

    tau_total:     np.ndarray = field(default_factory=lambda: np.array([]))
    """Torsional shear stress tau(x) = T(x)/Wt(x) [MPa], shape (n,)."""

    torsion_contributions: list = field(default_factory=list)
    """Per-node torsion contributions: list[list[dict{'label','T','tau'}]]."""

    # ------------------------------------------------------------------
    # 3. POST-PROCESSED ENGINEERING QUANTITIES
    # ------------------------------------------------------------------

    x:       np.ndarray = field(default_factory=lambda: np.array([]))
    """Node positions as array [mm] — same as x_nodes, for plotting convenience."""

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

    # ------------------------------------------------------------------
    # 4. BEARING NODE DATA
    # ------------------------------------------------------------------

    bearing_nodes: list[BearingNodeData] = field(default_factory=list)
    """
    Per-bearing complete FEM nodal state, in the same order as
    ShaftSystem.bearings (sorted by axial position).

    Consumed by ISO_16281_ball_bearing and Newton-Raphson coupling —
    these solvers never need to access the raw SimpleFEMSolver arrays.

    Access by label:
        node = next(n for n in results.bearing_nodes if n.label == "A")
    """


# ---------------------------------------------------------------------------
# SimpleFEMResultsLibrary — registry of ShaftResults keyed by shaft name
# ---------------------------------------------------------------------------

@dataclass
class SimpleFEMResultsLibrary:
    """
    Registry of ShaftResults from the SimpleFEMSolver pipeline.

    Keyed by ShaftSystem.name. All downstream solvers (ISO 16281,
    Newton-Raphson bearing compliance, fatigue, static failure) read
    from this library — they never write to it.

    This library stores ONLY the results of the initial rigid-bearing FEM
    solve (SimpleFEMSolver). It is the canonical source of shaft internal
    forces, deflections, bearing reactions, and section stresses.

    Dependency graph (read-only arrows from this library):

        SimpleFEMSolver
             |
        ShaftResultsReader.read(library)   <- populates this
             |
        SimpleFEMResultsLibrary
             |
             ├── ISO_16281BallBearingSolver
             ├── ShaftPostProcessor
             ├── StaticFailureSolver
             └── FatigueSolver

    Usage
    -----
        library = SimpleFEMResultsLibrary()

        solver = SimpleFEMSolver()
        solver.solve(shaft_system)
        ShaftResultsReader(solver, shaft_system).read(library)

        r = library.get("input_shaft")

    Design constraints
    ------------------
    - Key is ShaftResults.name = ShaftSystem.name; must be non-empty.
    - Re-storing under the same name overwrites (fresh solve replaces stale).
    - No solver logic. No GUI imports. No database calls.
    """

    _store: dict[str, ShaftResults] = field(
        default_factory=dict, init=False, repr=False
    )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def store(self, results: ShaftResults) -> None:
        """
        Add or overwrite results for results.name.

        Called automatically by ShaftResultsReader.read(library).
        Must be called before any downstream solver runs.
        """
        if not results.name:
            raise ValueError(
                "ShaftResults.name must be non-empty to use as library key. "
                "Ensure ShaftSystem.name is set before solving."
            )
        self._store[results.name] = results

    def remove(self, name: str) -> None:
        """Remove results for the given shaft name. No-op if absent."""
        self._store.pop(name, None)

    def clear(self) -> None:
        """Remove all stored results. Use when restarting the full pipeline."""
        self._store.clear()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, name: str) -> ShaftResults:
        """
        Return ShaftResults for the given shaft name.

        Raises KeyError if name is not present.
        """
        try:
            return self._store[name]
        except KeyError:
            raise KeyError(
                f"No SimpleFEM results stored for shaft '{name}'. "
                f"Available: {self.names()}"
            ) from None

    def get_or_none(self, name: str) -> ShaftResults | None:
        """Return ShaftResults for the given shaft name, or None if absent."""
        return self._store.get(name)

    # ------------------------------------------------------------------
    # Iteration / inspection
    # ------------------------------------------------------------------

    def names(self) -> list[str]:
        """Return stored shaft names, in insertion order."""
        return list(self._store)

    def iter(self) -> Iterator[tuple[str, ShaftResults]]:
        """Iterate over (name, ShaftResults) pairs, in insertion order."""
        return iter(self._store.items())

    def all_results(self) -> list[ShaftResults]:
        """Return all stored ShaftResults, in insertion order."""
        return list(self._store.values())

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, name: str) -> bool:
        return name in self._store

    def __repr__(self) -> str:
        return f"SimpleFEMResultsLibrary(shafts={self.names()})"


# ---------------------------------------------------------------------------
# ShaftResultsReader
# ---------------------------------------------------------------------------

class ShaftResultsReader:
    """
    Reads a solved SimpleFEMSolver into a ShaftResults and stores it
    immediately in the SimpleFEMResultsLibrary.

    Usage
    -----
        library = SimpleFEMResultsLibrary()

        reader  = ShaftResultsReader(solver, shaft_system)
        results = reader.read(library)

        # results is also accessible via library.get(shaft_system.name)

    Notes
    -----
    - library is mandatory — read() always stores the result.
    - Re-calling read() on the same shaft_system overwrites the previous
      entry in the library (fresh solve replaces stale data).
    - The beam element object is taken directly from the solver's builder
      so no second instantiation is needed.
    """

    def __init__(self, solver: "SimpleFEMSolver", shaft_system: "ShaftSystem"):
        self._solver  = solver
        self._sys     = shaft_system
        self._beam    = solver._builder.beam

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def read(self, library: SimpleFEMResultsLibrary) -> ShaftResults:
        """
        Post-process the solved SimpleFEMSolver into a ShaftResults and
        store it in the library under shaft_system.name.

        Parameters
        ----------
        library : SimpleFEMResultsLibrary
            Mandatory — ensures the library is always populated after a
            solve, whether in a single analysis or a multi-shaft loop.

        Returns
        -------
        ShaftResults — the same object stored in the library.
        """
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
        tau_arr = solver.tau_total

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

        results = ShaftResults(
            name = self._sys.name,

            # 1. mesh
            x_nodes  = x_nodes,
            elements = solver.elements,

            # 2. FEM solution — raw
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
            T_total               = solver.T_total,
            tau_total             = solver.tau_total,
            torsion_contributions = solver.torsion_contributions,

            # 3. post-processed
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

            # 4. bearing node data
            bearing_nodes = bearing_nodes,
        )

        library.store(results)
        return results

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

    def _recover_internal_forces(
        self, x_nodes: list[float], n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        V_xz, M_xz = self._sweep_plane_from_elements(LoadPlane.XZ)
        V_xy, M_xy = self._sweep_plane_from_elements(LoadPlane.XY)
        return M_xz, M_xy, V_xz, V_xy

    def _sweep_plane_from_elements(
        self, plane: LoadPlane
    ) -> tuple[np.ndarray, np.ndarray]:
        solver   = self._solver
        elements = solver.elements
        x_nodes  = solver.x_nodes
        d_total  = solver.d_total_xz if plane == LoadPlane.XZ else solver.d_total_xy

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

            V1, M1 = -f_e[1], -f_e[2]
            V2, M2 =  f_e[4],  f_e[5]

            if elem_idx == 0:
                V[elem.idx_node_1] = V1
                M[elem.idx_node_1] = M1
            V[elem.idx_node_2] = V2
            M[elem.idx_node_2] = M2

        return V, M

    @staticmethod
    def _q_total(segments: list[dict], x: float) -> float:
        total = 0.0
        for seg in segments:
            if seg["x_lo"] - 1e-9 <= x <= seg["x_hi"] + 1e-9:
                total += seg["fn"](x)
        return total

    def _recover_deflections(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        solver = self._solver
        v_xz   = np.array([solver.d_total_xz[3 * i + 1] for i in range(n)])
        v_xy   = np.array([solver.d_total_xy[3 * i + 1] for i in range(n)])
        return v_xz, v_xy

    def _section_properties(
        self, x_nodes: list[float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        """
        Extract complete FEM nodal state at each bearing position.

        psi_xz / psi_xy are computed from the displacement gradient
        across the bearing seat width (shaft_system.bearing_extent).
        Falls back to nodal theta if seat width == 0.
        """
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

    def _psi(self, x_nodes, solver, bearing) -> tuple[float, float]:
        """
        Shaft centreline slope across the bearing seat [rad].
        Gradient of v across [x_lo, x_hi]; fallback to theta if seat width == 0.
        """
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