"""
axisforge/solvers/machine_elements/shaft/fem_solvers/rigid_bearing.py

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
overwrites all of them in place — a fresh RigidBearingFEMSolver is cheap to
build if isolation between runs is needed.
"""

from __future__ import annotations

import numpy as np
from typing import Callable


from axisforge.mesh.shaft.mesh_generation.mesh_1D import Mesh1D
from axisforge.mesh.shaft.element_type.elem import Elem
from axisforge.solvers.machine_elements.shaft.fem_solvers.build_stiffness_matrix import StiffnessMatrixBuilder
from axisforge.core.loads import RadialLoad, AxialLoad, ExternalMoment, LoadPlane
from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import ShaftSystem
from axisforge.config import MESH_MIN_NODE_DIST_MM, SOLVER_TOLERANCE


class RigidBearingFEMSolver:
    """
    Orchestrates: Mesh1D -> Elem.from_mesh -> StiffnessMatrixBuilder ->
    boundary conditions -> per-load-case solve -> superposition ->
    torsion diagram (same x_nodes).

    Usage
    -----
        solver = RigidBearingFEMSolver()
        solver.solve(shaft_system)
        solver.tau_total       # -> np.ndarray, MPa
        solver.d_total_xz      # -> np.ndarray, global displacement (XZ)
    """

    def __init__(self, theory: str = "timoshenko", 
                 distribute_gear_labels: set[str] | None = None):
        
        """
        distribute_gear_labels : set of gear labels whose mesh loads should be
                             treated as distributed over face width.
                             None or empty -> all gear mesh loads as point loads.
                             e.g. {"pinion", "wheel"} -> only those two distributed.
                             Use {"*"} as a sentinel to distribute ALL gear mesh loads.
        """
        
        self._builder = StiffnessMatrixBuilder(theory=theory)
        self._distribute_all   = distribute_gear_labels == {"*"}
        self._distribute_labels = distribute_gear_labels or set()

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

    def solve(self, 
                shaft_system: ShaftSystem,
                extra_mandatory: list[float] | None = None) -> None:
            shaft_system.validate_or_raise()

            mesh = Mesh1D(shaft_system, extra_mandatory=extra_mandatory or [])
            x_nodes = mesh.x_nodes
            elements = Elem.from_mesh(mesh)

            self.K = self._builder.build_stiffness_matrix(mesh, elements)

            free_dofs, constrained_dofs = self._boundary_dofs(x_nodes, shaft_system)
            K_red = self.K[np.ix_(free_dofs, free_dofs)]
            self._check_conditioning(K_red)

            load_cases = self._build_load_cases(shaft_system)
            n_dofs = 3 * len(x_nodes)

            # --- assemble global load vectors ---
            f_xz_ext = np.zeros(n_dofs)
            f_xy_ext = np.zeros(n_dofs)

            for lc in load_cases:
                f_xz = self._assemble_load_vector(x_nodes, lc["radial_xz"],
                                                lc["axial"], lc["moments_xz"])
                f_xy = self._assemble_load_vector(x_nodes, lc["radial_xy"],
                                                [], lc["moments_xy"])

                if lc.get("distributed_xz"):
                    for d in lc["distributed_xz"]:
                        f_xz += self._assemble_distributed_load_vector(
                            x_nodes, elements, d["x_lo"], d["x_hi"], d["q"],
                            theta_fn=d.get("theta_fn"))

                if lc.get("distributed_xy"):
                    for d in lc["distributed_xy"]:
                        f_xy += self._assemble_distributed_load_vector(
                            x_nodes, elements, d["x_lo"], d["x_hi"], d["q"],
                            theta_fn=d.get("theta_fn"))

                f_xz_ext += f_xz
                f_xy_ext += f_xy

            # --- global solve (single system) ---
            d_total_xz = np.zeros(n_dofs)
            d_total_xy = np.zeros(n_dofs)
            d_total_xz[free_dofs] = np.linalg.solve(K_red, f_xz_ext[free_dofs])
            d_total_xy[free_dofs] = np.linalg.solve(K_red, f_xy_ext[free_dofs])

            f_xz_total = self.K @ d_total_xz
            f_xy_total = self.K @ d_total_xy

            # --- NOTE: per-load-case superposition (kept for reference) ---
            # d_contributions = []
            # d_total_xz = np.zeros(n_dofs)
            # d_total_xy = np.zeros(n_dofs)
            # for lc in load_cases:
            #     f_xz = assemble(lc["radial_xz"], lc["axial"], lc["moments_xz"])
            #     f_xy = assemble(lc["radial_xy"], [], lc["moments_xy"])
            #     d_xz = np.zeros(n_dofs)
            #     d_xy = np.zeros(n_dofs)
            #     d_xz[free_dofs] = solve(K_red, f_xz[free_dofs])
            #     d_xy[free_dofs] = solve(K_red, f_xy[free_dofs])
            #     d_contributions.append({"label": lc["label"], "source": lc["source"],
            #                             "d_xz": d_xz, "d_xy": d_xy})
            #     d_total_xz += d_xz
            #     d_total_xy += d_xy
            # Superposition is numerically equivalent to the global solve above.
            # Restore if per-load traceability in diagrams is needed.

            # --- torsion ---
            T_total, tau_total, torsion_contributions = self._solve_torsion(
                shaft_system, x_nodes)

            # --- publish ---
            self.x_nodes          = x_nodes
            self.elements         = elements
            self.free_dofs        = free_dofs
            self.constrained_dofs = constrained_dofs

            self.f_xz_ext         = f_xz_ext
            self.f_xy_ext         = f_xy_ext

            self.d_total_xz       = d_total_xz
            self.d_total_xy       = d_total_xy
            self.f_xz_total       = f_xz_total
            self.f_xy_total       = f_xy_total

            self.f_xz_reaction    = f_xz_total - f_xz_ext
            self.f_xy_reaction    = f_xy_total - f_xy_ext

            self.d_contributions  = None   # disabled — see NOTE above
            self.T_total          = T_total
            self.tau_total        = tau_total
            self.torsion_contributions = torsion_contributions

    # ------------------------------------------------------------------
    # Load-case construction (decomposes RadialLoad/ExternalMoment by plane)
    # ------------------------------------------------------------------

    def _build_load_cases(self, shaft_system: ShaftSystem) -> list[dict]:
        cases: list[dict] = []


        for ld in shaft_system.radial_loads:
            if self._is_distributed(ld.label, ld.source):
                continue   # gear mesh load — will be handled as distributed below
            cases.append({
                "label": ld.label or f"radial@{ld.position:.1f}",
                "source": ld.source,
                "radial_xz": [(ld.position, ld.component(LoadPlane.XZ))],
                "radial_xy": [(ld.position, ld.component(LoadPlane.XY))],
                "axial": [], "moments_xz": [], "moments_xy": [],
            })

        for ld in shaft_system.distributed_radial_loads:
            theta_fn = ld._theta_fn if ld._theta_variable else None
            cases.append({
                "label": ld.label or f"dist@[{ld.x_lo:.1f},{ld.x_hi:.1f}]",
                "source": ld.source,
                "distributed_xz": [{
                    "x_lo": ld.x_lo, "x_hi": ld.x_hi,
                    "q": lambda x, _ld=ld: _ld.component_intensity(x, LoadPlane.XZ),
                    "theta_fn": theta_fn,
                }],
                "distributed_xy": [{
                    "x_lo": ld.x_lo, "x_hi": ld.x_hi,
                    "q": lambda x, _ld=ld: _ld.component_intensity(x, LoadPlane.XY),
                    "theta_fn": theta_fn
                }],
                "radial_xz": [], "radial_xy": [],
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

    def _assemble_distributed_load_vector(self, x_nodes: list[float],
                                          elements: list[Elem],
                                          x_lo: float, x_hi: float,
                                          q: Callable[[float], float],
                                          theta_fn: Callable[[float], float] | None = None) -> np.ndarray:
        """
        Equivalent nodal force vector for a distributed transverse load q(x) [N/mm]
        over [x_lo, x_hi] via Gauss quadrature on each element.

            f_e = ∫_{-1}^{+1} N^T(zeta) · q(x(zeta)) · J dzeta

        Returns np.ndarray of size 3 * len(x_nodes), transverse DOF only (v).
        """
        beam = self._builder.beam
        f    = np.zeros(3 * len(x_nodes))

        for elem in elements:
            x_a = x_nodes[elem.idx_node_1]
            x_b = x_nodes[elem.idx_node_2]

            if x_b <= x_lo or x_a >= x_hi:
                continue

            # clamp integration to load span
            x_lo_elem = max(x_a, x_lo)
            x_hi_elem = min(x_b, x_hi)

            x_map  = beam.global_to_natural_radial(x_lo_elem, x_hi_elem, elem)
            q_zeta = beam.vetor_global_to_natural(q, x_map)
            J      = beam.jacobian(elem)

            n_gauss          = beam.gauss_order(q, x_lo_elem, x_hi_elem, elem, theta_fn=theta_fn)
            gauss_pts, gauss_wts = beam.gauss_quadrature(n_gauss)

            f_elem = np.zeros(6)
            for xi, w in zip(gauss_pts, gauss_wts):
                N       = beam.shape_functions(xi, elem)
                f_elem[1] += N[1] * q_zeta(xi) * J * w
                f_elem[4] += N[4] * q_zeta(xi) * J * w

            f[3 * elem.idx_node_1 + 1] += f_elem[1]
            f[3 * elem.idx_node_2 + 1] += f_elem[4]

        return f

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
        """
        Check that the net applied torque (sum of every TorqueLoad.magnitude)
        is ~0 -- for a shaft in equilibrium, transmitted torque in must equal
        torque out, so T(x) -> 0 past the last torque source (see the sign
        convention note in _solve_torsion). A non-zero net usually means a
        missing or mis-signed load -- e.g. the driven-side reaction torque
        was never added, or GearSystem._forces_to_loads produced the wrong
        sign for one of the sources.

        Returns
        -------
        list[str]
            Empty if the shaft is in torsional equilibrium; otherwise one
            message describing the imbalance.
        """
        errors: list[str] = []
        net = sum(ld.magnitude for ld in shaft_system.torque_loads)

        if abs(net) > tol:
            errors.append(
                f"Torsional equilibrium violated: net torque = {net:.6f} N*m "
                f"(tol = {tol:.1e}). Check that every TorqueLoad source "
                f"(driver and driven) was added with the correct sign."
            )

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
    # Helper
    # ------------------------------------------------------------------

    def _is_distributed(self, label: str, source: str) -> bool:
        if source != "gear_mesh":
            return False              # user loads sempre pontuais
        if self._distribute_all:
            return True
        gear_label = label.split(":")[0]
        return gear_label in self._distribute_labels

    def _find_node(x_nodes: list[float], x: float, tol: float = 1e-6) -> int:
        for i, xi in enumerate(x_nodes):
            if abs(xi - x) < tol:
                return i
        raise ValueError(f"Position {x:.4f} mm not found in x_nodes.")
    

    # ------------------------------------------------------------------
    # Numerical guard
    # ------------------------------------------------------------------

    def _check_conditioning(self, K_red: np.ndarray) -> None:
        if np.linalg.cond(K_red) > 1e14:
            raise np.linalg.LinAlgError(
                "SimpleFEMSolver: reduced stiffness matrix is (near-)singular — "
                "check bearing count/positions (mechanism / insufficient constraints)."
            )

    # ------------------------------------------------------------------
    # Return values for submodeling
    # ------------------------------------------------------------------

    def return_values(
                    self,
                    x_nodes: list[float],
                    x: list[float]) -> dict[float, dict[str, float]]:
        """
        Extract nodal solution quantities for all nodes within the
        interval [x[0], x[1]].

        Parameters
        ----------
        x_nodes : mesh node positions of the current FEM solution
        x       : [x_lo, x_hi] — interval bounds (must coincide with nodes)

        Returns
        -------
        dict mapping each node position within [x_lo, x_hi] to its quantities:
            {
                x_i: {"v_xz": ..., "theta_xz": ..., "f1_xz": ..., "f2_xz": ...,
                       "v_xy": ..., "theta_xy": ..., "f1_xy": ..., "f2_xy": ...},
                ...
            }

        Notes
        -----
        v     : transverse displacement   [mm]
        theta : rotation                  [rad]
        f1    : nodal transverse force    [N]
        f2    : nodal moment              [N·mm]

        Raises
        ------
        RuntimeError if solve() has not been called yet.
        ValueError   if x does not contain exactly two positions.
        """
        if self.d_total_xz is None:
            raise RuntimeError(
                "return_values() called before solve(). "
                "Call solver.solve(shaft_system) first."
            )

        if len(x) != 2:
            raise ValueError(
                f"x must contain exactly [x_lo, x_hi], got {len(x)} values."
            )

        x_lo, x_hi = x[0], x[1]
        result: dict[float, dict[str, float]] = {}

        for i, xi in enumerate(x_nodes):
            if x_lo - SOLVER_TOLERANCE <= xi <= x_hi + SOLVER_TOLERANCE:
                result[xi] = {
                    "u":            float(self.d_total_xz[3 * i]),
                    "v_xz":         float(self.d_total_xz[3 * i + 1]),
                    "theta_xz":     float(self.d_total_xz[3 * i + 2]),
                    "f_u":          float(self.f_xz_total[3 * i]),
                    "f_v_xz":       float(self.f_xz_total[3 * i + 1]),
                    "f_theta_xz":   float(self.f_xz_total[3 * i + 2]),
                    "v_xy":         float(self.d_total_xy[3 * i + 1]),
                    "theta_xy":     float(self.d_total_xy[3 * i + 2]),
                    "f1_xy":        float(self.f_xy_total[3 * i + 1]),
                    "f2_xy":        float(self.f_xy_total[3 * i + 2]),
                }

        return result
        