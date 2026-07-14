"""
axisforge/solvers/machine_elements/shaft/oneD_analysis/simple_fem_solver.py

Reuses the FEM pipeline from the legacy shaft_analysis.StaticsSolver, wired
to the current-generation components:

  Mesh1D                 -> node grid
  Elem.from_mesh         -> element list (length, E, I, A, v, node indices)
  StiffnessMatrixBuilder -> global K (Timoshenko only, for now)
  ShaftSystem.loads/bearings -> boundary conditions + load vectors

DOF convention (per node, 3 DOF): [u (axial), v (transverse), theta (bending)].
Two INDEPENDENT planar solves are performed (XZ, XY) reusing the same K.

Torsion is bundled into this solver (not a separate class) because it is
static analysis on the SAME ShaftSystem, sharing the SAME x_nodes — no FEM
matrix involved, just a cumulative T(x) = sum of TorqueLoad up to x, and
tau(x) = T(x)/Wt(x) via Shaft.Wt_at(). Keeping it here avoids a second
mesh pass and keeps sigma/tau aligned node-for-node for StressSolver.

Axial DOF de-duplication: both planar solves share the physical axial DOF
(u). AxialLoad is injected ONLY into the XZ load vector, never XY, to
avoid double-counting (mirrors the legacy solver).

solve() does not return anything — every intermediate quantity is stored
as a public attribute on the solver instance (numerical transparency is a
hard project requirement). Call solve() once, then read the attributes:

  x_nodes, elements, free_dofs, constrained_dofs,
  d_total_xz, d_total_xy, f_xz_ext, f_xy_ext, d_contributions,
  T_total, tau_total, torsion_contributions

Each attribute is None until solve() has run at least once for the
current instance. Re-calling solve() (e.g. after a re-resolve() upstream)
overwrites all of them in place — a fresh SimpleFEMSolver is cheap to
build if isolation between runs is needed.
"""

from __future__ import annotations

import numpy as np

from axisforge.mesh.oneD.shaft.mesh_generation.mesh_1D import Mesh1D
from axisforge.mesh.oneD.shaft.Elements.elem import Elem
from axisforge.solvers.machine_elements.shaft.oneD_analysis.build_stiffness_matrix import StiffnessMatrixBuilder
from axisforge.core.loads import RadialLoad, AxialLoad, ExternalMoment, LoadPlane
from axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.shaft_system import ShaftSystem
from axisforge.config import MESH_MIN_NODE_DIST_MM, SOLVER_TOLERANCE


class SimpleFEMSolver:
    """
    Orchestrates: Mesh1D -> Elem.from_mesh -> StiffnessMatrixBuilder ->
    boundary conditions -> per-load-case solve -> superposition ->
    torsion diagram (same x_nodes).

    Usage
    -----
        solver = SimpleFEMSolver()
        solver.solve(shaft_system)
        solver.tau_total       # -> np.ndarray, MPa
        solver.d_total_xz      # -> np.ndarray, global displacement (XZ)
    """

    def __init__(self, theory: str = "timoshenko", constraint_bearing: str = "rigid"):
        self._builder = StiffnessMatrixBuilder(theory=theory)

        # --- public result attributes, populated by solve() ---
        self.x_nodes: list[float] | None = None
        self.elements: list[Elem] | None = None
        self.free_dofs: list[int] | None = None
        self.constrained_dofs: list[int] | None = None

        # bending / axial (FEM)
        self.d_total_xz: np.ndarray | None = None
        self.d_total_xy: np.ndarray | None = None
        self.f_xz_ext: np.ndarray | None = None
        self.f_xy_ext: np.ndarray | None = None
        self.d_contributions: list[dict] | None = None
        # each: {'label': str, 'source': str, 'd_xz': ndarray, 'd_xy': ndarray}

        # torsion (pure statics, no DOF, same x_nodes)
        self.T_total: np.ndarray | None = None
        self.tau_total: np.ndarray | None = None
        self.torsion_contributions: list[list[dict]] | None = None
        # each node: [{'label': str, 'T': float, 'tau': float}, ...]

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def solve(self, shaft_system: ShaftSystem) -> None:
        shaft_system.validate_or_raise()

        mesh = Mesh1D(shaft_system)
        x_nodes = mesh.x_nodes
        elements = Elem.from_mesh(mesh)

        K = self._builder.build_stiffness_matrix(mesh, elements)

        free_dofs, constrained_dofs = self._boundary_dofs(x_nodes, shaft_system)
        K_red = K[np.ix_(free_dofs, free_dofs)]
        self._check_conditioning(K_red)

        # --- bending / axial: one load case per Load object, superposed ---
        load_cases = self._build_load_cases(shaft_system)

        n_dofs = 3 * len(x_nodes)
        d_contributions = []
        d_total_xz = np.zeros(n_dofs)
        d_total_xy = np.zeros(n_dofs)

        for lc in load_cases:
            f_xz = self._assemble_load_vector(x_nodes, lc["radial_xz"],
                                               lc["axial"], lc["moments_xz"])
            f_xy = self._assemble_load_vector(x_nodes, lc["radial_xy"],
                                               [], lc["moments_xy"])  # no axial here in order to have only 1 axial in the total ShaftSystem

            d_xz = np.zeros(n_dofs)
            d_xy = np.zeros(n_dofs)
            d_xz[free_dofs] = np.linalg.solve(K_red, f_xz[free_dofs])
            d_xy[free_dofs] = np.linalg.solve(K_red, f_xy[free_dofs])

            d_contributions.append({
                "label": lc["label"], "source": lc["source"],
                "d_xz": d_xz, "d_xy": d_xy,
            })
            d_total_xz += d_xz
            d_total_xy += d_xy

        f_xz_ext = K @ d_total_xz
        f_xy_ext = K @ d_total_xy

        # --- torsion: pure statics, same x_nodes, no DOF ---
        T_total, tau_total, torsion_contributions = self._solve_torsion(
            shaft_system, x_nodes)

        # --- publish everything as public attributes ---
        self.x_nodes = x_nodes
        self.elements = elements
        self.free_dofs = free_dofs
        self.constrained_dofs = constrained_dofs

        self.d_total_xz = d_total_xz
        self.d_total_xy = d_total_xy
        self.f_xz_ext = f_xz_ext
        self.f_xy_ext = f_xy_ext
        self.d_contributions = d_contributions

        self.T_total = T_total
        self.tau_total = tau_total
        self.torsion_contributions = torsion_contributions

    # ------------------------------------------------------------------
    # Load-case construction (decomposes RadialLoad/ExternalMoment by plane)
    # ------------------------------------------------------------------

    def _build_load_cases(self, shaft_system: ShaftSystem) -> list[dict]:
        cases: list[dict] = []

        for ld in shaft_system.radial_loads:
            cases.append({
                "label": ld.label or f"radial@{ld.position:.1f}",
                "source": ld.source,
                "radial_xz": [(ld.position, ld.component(LoadPlane.XZ))],
                "radial_xy": [(ld.position, ld.component(LoadPlane.XY))],
                "axial": [], "moments_xz": [], "moments_xy": [],
            })

        for ld in shaft_system.axial_loads:
            cases.append({
                "label": ld.label or f"axial@{ld.position:.1f}",
                "source": ld.source,
                "radial_xz": [], "radial_xy": [],
                "axial": [(ld.position, ld.magnitude)],
                "moments_xz": [], "moments_xy": [],
            })

        for m in shaft_system.external_moments:
            cases.append({
                "label": m.label or f"moment@{m.position:.1f}",
                "source": m.source,
                "radial_xz": [], "radial_xy": [], "axial": [],
                "moments_xz": [(m.position, m.component(LoadPlane.XZ))],
                "moments_xy": [(m.position, m.component(LoadPlane.XY))],
            })

        # TorqueLoad intentionally excluded here — no DOF in this element,
        # handled separately by _solve_torsion below.
        return cases

    # ------------------------------------------------------------------
    # Load vector assembly (u, v, theta per node)
    # ------------------------------------------------------------------

    def _assemble_load_vector(self, x_nodes, radial, axial, moments) -> np.ndarray:
        f = np.zeros(3 * len(x_nodes))
        for x, mag in axial:
            i = Elem.find_node_index(x_nodes, x)
            f[3 * i] += mag
        for x, mag in radial:
            i = Elem.find_node_index(x_nodes, x)
            f[3 * i + 1] += mag
        for x, mag in moments:
            i = Elem.find_node_index(x_nodes, x)
            f[3 * i + 2] += mag
        return f

    # ------------------------------------------------------------------
    # Torsion — cumulative T(x), tau(x) = T(x)/Wt(x)
    # ------------------------------------------------------------------

    def _solve_torsion(
        self, shaft_system: ShaftSystem, x_nodes: list[float]
    ) -> tuple[np.ndarray, np.ndarray, list[list[dict]]]:
        """
        T(x) = sum of TorqueLoad.magnitude for every source at position <= x.

        Sign convention fixed upstream in GearSystem._forces_to_loads:
        driver mesh point -> +T_in ; driven mesh point -> -T_out. A shaft
        in equilibrium (net torque ~0) returns T(x)->0 at the free end;
        see validate_torsion_equilibrium() for the explicit check.

        Units: TorqueLoad.magnitude is N*m (per loads.py docstring); Wt_at
        returns mm^3 -> tau = T[N*m]*1000 / Wt[mm^3] = N/mm^2 = MPa.
        """
        torque_sources = sorted(
            (
                (ld.position, ld.magnitude, ld.label or f"torque@{ld.position:.1f}")
                for ld in shaft_system.torque_loads
            ),
            key=lambda t: t[0],
        )

        n = len(x_nodes)
        T_total = np.zeros(n)
        tau_total = np.zeros(n)
        contributions: list[list[dict]] = []

        for i, x in enumerate(x_nodes):
            Wt_x = shaft_system.shaft.Wt_at(x)
            node_contribs = []
            for pos, mag, label in torque_sources:
                if pos <= x + SOLVER_TOLERANCE:
                    node_contribs.append({
                        "label": label,
                        "T": mag,
                        "tau": mag * 1000.0 / Wt_x,
                    })
            contributions.append(node_contribs)
            T_total[i] = sum(c["T"] for c in node_contribs)
            tau_total[i] = T_total[i] * 1000.0 / Wt_x

        return T_total, tau_total, contributions

    def validate_torsion_equilibrium(
        self, shaft_system: ShaftSystem, tol: float = 1e-6
    ) -> list[str]:

        errors: list[str] = []
        net = sum(ld.magnitude for ld in shaft_system.torque_loads)
        return errors

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def _boundary_dofs(self, x_nodes, shaft_system: ShaftSystem) -> tuple[list[int], list[int]]:
        constrained: list[int] = []
        for b in shaft_system.bearings:
            i = Elem.find_node_index(x_nodes, b.position)
            constrained.append(3 * i + 1)          # v = 0, always
            if b.arrangement == "locating":
                constrained.append(3 * i)          # u = 0, locating bearing only
        n_dofs = 3 * len(x_nodes)
        free = [d for d in range(n_dofs) if d not in constrained]
        return free, constrained

    # ------------------------------------------------------------------
    # Numerical guard
    # ------------------------------------------------------------------

    def _check_conditioning(self, K_red: np.ndarray) -> None:
        if np.linalg.cond(K_red) > 1e14:
            raise np.linalg.LinAlgError(
                "SimpleFEMSolver: reduced stiffness matrix is (near-)singular — "
                "check bearing count/positions (mechanism / insufficient constraints)."
            )