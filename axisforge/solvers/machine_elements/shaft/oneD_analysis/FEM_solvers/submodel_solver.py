"""
axisforge/solvers/machine_elements/shaft/oneD_analysis/FEM_solvers/submodel_solver.py

Submodel solver for Richardson GCI convergence study.

Wraps SimpleFEMSolver and restricts metric evaluation to a
subdomain [x_lo, x_hi] — typically the extent of a distributed
radial load. The full shaft_system is solved; only the extraction
of metric values is localised to the subdomain.

Used exclusively by MeshConvergenceStudy — not part of the
production solve pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing import Callable


from axisforge.solvers.machine_elements.shaft.oneD_analysis.FEM_solvers.simple_fem_solver import SimpleFEMSolver
from axisforge.solvers.machine_elements.shaft.oneD_analysis.static.static_analysis import ShaftResultsReader
from axisforge.mesh.oneD.shaft.Elements.elem import Elem
from axisforge.mesh.oneD.shaft.mesh_generation.mesh_1D import Mesh1D
from axisforge.solvers.machine_elements.shaft.oneD_analysis.build_stiffness_matrix import StiffnessMatrixBuilder
from axisforge.config import SOLVER_TOLERANCE
from axisforge.core.loads import LoadPlane


# ===========================================================================
# Data container
# ===========================================================================

@dataclass
class SubmodelResult:
    """Full solution of the submodel within [x_lo, x_hi]."""

    x_lo: float
    x_hi: float
    grade: str
    """Grade string used to generate the mesh (e.g. 'grade_0', 'grade_2')."""

    x_nodes: list[float]
    """Node positions within the subdomain."""

    d_xz: np.ndarray
    """Displacement vector — XZ plane (u, v, theta per node)."""

    d_xy: np.ndarray
    """Displacement vector — XY plane (u, v, theta per node)."""

    lam_xz: np.ndarray
    """Lagrange multipliers — XZ plane (reaction forces at cut nodes)."""

    lam_xy: np.ndarray
    """Lagrange multipliers — XY plane (reaction forces at cut nodes)."""


# ===========================================================================
# Submodel grader
# ===========================================================================

class SubmodelGrader:
    """
    Generates standardised mesh grades for a subdomain [x_lo, x_hi]
    by successive elementwise bisection of the base mesh.

    Grade definition
    ----------------
    grade_0 : base mesh nodes already present in [x_lo, x_hi]
    grade_1 : grade_0 + midpoint of each element
    grade_2 : grade_1 + midpoint of each element  (4x grade_0 elements)
    grade_N : grade_{N-1} bisected elementwise

    The grade string is produced by RichardsonGCI and consumed
    by SubmodelSolver.

    Parameters
    ----------
    x_lo    : lower bound of the subdomain
    x_hi    : upper bound of the subdomain
    x_nodes : full mesh node positions (from Mesh1D.x_nodes)
    """

    def __init__(
        self,
        x_lo: float,
        x_hi: float,
        x_nodes: list[float],
    ):
        self._x_lo = x_lo
        self._x_hi = x_hi
        self._x_nodes = x_nodes

    def get_grade(self, grade: str) -> list[float]:
        """
        Return node positions for the requested grade.

        Parameters
        ----------
        grade : "grade_0" | "grade_1" | ... | "grade_N"

        Returns
        -------
        Sorted list of node positions within [x_lo, x_hi].

        Raises
        ------
        ValueError if grade string is malformed or N is negative.
        """
        n = self._parse_grade(grade)
        nodes = self._base_nodes()
        for _ in range(n):
            nodes = self._bisect_once(nodes)
        return nodes

    def _parse_grade(self, grade: str) -> int:
        """
        Parse grade string into refinement level integer.

        "grade_0" -> 0, "grade_1" -> 1, etc.

        Raises
        ------
        ValueError if string does not match expected format.
        """
        prefix = "grade_"
        if not grade.startswith(prefix):
            raise ValueError(
                f"Invalid grade string: {grade!r}. "
                f"Expected format: 'grade_N' where N >= 0."
            )
        suffix = grade[len(prefix):]
        if not suffix.isdigit():
            raise ValueError(
                f"Invalid grade level: {suffix!r}. "
                f"Expected a non-negative integer after 'grade_'."
            )
        return int(suffix)

    def _base_nodes(self) -> list[float]:
        """
        Extract nodes from x_nodes that fall within [x_lo, x_hi].

        These are grade_0 — the natural mesh nodes in the subdomain,
        one per existing element boundary.
        """
        from axisforge.config import SOLVER_TOLERANCE
        return [
            x for x in self._x_nodes
            if self._x_lo - SOLVER_TOLERANCE <= x <= self._x_hi + SOLVER_TOLERANCE
        ]

    def _bisect_once(self, nodes: list[float]) -> list[float]:
        """
        Insert the midpoint of every interval between consecutive nodes.

        Applied iteratively to produce grade_1, grade_2, ... grade_N.
        """
        result = list(nodes)
        for a, b in zip(nodes, nodes[1:]):
            result.append((a + b) / 2.0)
        return sorted(result)


# ===========================================================================
# Submodel solver
# ===========================================================================

class SubmodelSolver:
    """
    Solves the full shaft_system and extracts metric values
    restricted to the subdomain [x_lo, x_hi].

    Parameters
    ----------
    theory  : FEM theory string forwarded to SimpleFEMSolver
    metric  : quantity to extract — "sigma_b" | "M_max" | "l2_M"
    """
    def __init__(self,
                 x_lo: float,
                 x_hi: float):

        self._x_lo = x_lo
        self._x_hi = x_hi

    def solve(self,
              global_solver: SimpleFEMSolver,   # já resolvido externamente
              shaft_system,
              grade: str) -> SubmodelResult:


        x_lo = self._x_lo
        x_hi = self._x_hi
        # 1. verifica que o global_solver já foi corrido
        if global_solver.d_total_xz is None:
            raise RuntimeError(
                "SubmodelSolver.solve() requires a solved SimpleFEMSolver. "
                "Call global_solver.solve(shaft_system) first."
            )

        # 2. extrai BCs dos nós de corte via return_values
        bc_data = global_solver.return_values(global_solver.x_nodes, [x_lo, x_hi])

        # 3. grade -> candidatos -> Mesh1D com extra_mandatory
        grader     = SubmodelGrader(x_lo, x_hi, global_solver.x_nodes)
        candidates = grader.get_grade(grade)
        mesh       = Mesh1D(shaft_system, extra_mandatory=candidates)
        x_nodes    = mesh.x_nodes
        elements   = Elem.from_mesh(mesh)

        # --- stiffness matrix ---
        builder  = global_solver._builder
        K_sub    = self._build_submodel_stiffness(mesh, elements, builder)

        # --- force vectors ---
        load_cases = self._build_submodel_load_cases(shaft_system)

        n_dofs = 3 * len(x_nodes)
        f_xz   = np.zeros(n_dofs)
        f_xy   = np.zeros(n_dofs)

        for lc in load_cases:
            f_xz += self._assemble_load_vector(
                x_nodes, lc["radial_xz"], lc["axial"], lc["moments_xz"]
            )
            f_xy += self._assemble_load_vector(
                x_nodes, lc["radial_xy"], [], lc["moments_xy"]
            )

            if lc.get("distributed_xz"):
                for d in lc["distributed_xz"]:
                    f_xz += self._assemble_distributed_load_vector(
                        x_nodes, elements, d["x_lo"], d["x_hi"], d["q"],
                        builder, theta_fn=d.get("theta_fn")
                    )

            if lc.get("distributed_xy"):
                for d in lc["distributed_xy"]:
                    f_xy += self._assemble_distributed_load_vector(
                        x_nodes, elements, d["x_lo"], d["x_hi"], d["q"],
                        builder, theta_fn=d.get("theta_fn")
                    )

        # --- constraint matrix and prescribed vectors ---
        C, p_dofs = self._build_constraint_matrix(x_nodes)
        q_xz, q_xy = self._build_prescribed_vectors(x_nodes, bc_data)

        # --- augmented system [K C^T; C 0] ---
        n = n_dofs
        K_aug = np.zeros((2 * n, 2 * n))
        K_aug[:n, :n] = K_sub          # K
        K_aug[:n, n:] = C.T            # C^T
        K_aug[n:, :n] = C              # C
        # K_aug[n:, n:] = 0            # already zero

        # --- augmented RHS ---
        # XZ plane
        rhs_xz = np.zeros(2 * n)
        rhs_xz[:n] = f_xz
        rhs_xz[n:] = q_xz

        # XY plane
        rhs_xy = np.zeros(2 * n)
        rhs_xy[:n] = f_xy
        rhs_xy[n:] = q_xy

        # --- solve ---
        sol_xz = np.linalg.solve(K_aug, rhs_xz)
        sol_xy = np.linalg.solve(K_aug, rhs_xy)

        # extract displacements and Lagrange multipliers
        d_xz = sol_xz[:n]   # displacements
        d_xy = sol_xy[:n]
        lam_xz = sol_xz[n:] # Lagrange multipliers (reaction forces at cut nodes)
        lam_xy = sol_xy[n:]

        return SubmodelResult(
            x_lo=self._x_lo,
            x_hi=self._x_hi,
            grade=grade,
            x_nodes=x_nodes,
            d_xz=d_xz,
            d_xy=d_xy,
            lam_xz=lam_xz,
            lam_xy=lam_xy,
        )

    # ------------------------------------------------------------------
    # Stiffness matrix for the submodel
    # ------------------------------------------------------------------

    def _build_submodel_stiffness(self,
                                  mesh: Mesh1D,
                                  elements: list[Elem],
                                  builder: StiffnessMatrixBuilder) -> np.ndarray:
        """
        Assemble stiffness matrix for elements within [x_lo, x_hi] only.
        Node indices are local to the submodel mesh.
        """
        n_dofs = 3 * mesh.n_nodes
        K = np.zeros((n_dofs, n_dofs))
        beam = builder.beam

        for elem in elements:
            x_a = mesh.x_nodes[elem.idx_node_1]
            x_b = mesh.x_nodes[elem.idx_node_2]

            if x_b <= self._x_lo or x_a >= self._x_hi:
                continue

            K_elem = beam.stiffness_element(elem)
            dofs = [
                3 * elem.idx_node_1,
                3 * elem.idx_node_1 + 1,
                3 * elem.idx_node_1 + 2,
                3 * elem.idx_node_2,
                3 * elem.idx_node_2 + 1,
                3 * elem.idx_node_2 + 2,
            ]
            for i, gi in enumerate(dofs):
                for j, gj in enumerate(dofs):
                    K[gi, gj] += K_elem[i, j]

        return K 

    # ------------------------------------------------------------------
    # Load-case construction (decomposes RadialLoad/ExternalMoment by plane)
    # ------------------------------------------------------------------
  
    def _build_submodel_load_cases(self, shaft_system) -> list[dict]:
        """
        Same as SimpleFEMSolver._build_load_cases but filtered to
        loads within [x_lo, x_hi]. Distributed loads are clamped
        to the subdomain bounds.
        """
        cases: list[dict] = []

        for ld in shaft_system.radial_loads:
            if not (self._x_lo - SOLVER_TOLERANCE
                    <= ld.position
                    <= self._x_hi + SOLVER_TOLERANCE):
                continue
            cases.append({
                "label": ld.label or f"radial@{ld.position:.1f}",
                "source": ld.source,
                "radial_xz": [(ld.position, ld.component(LoadPlane.XZ))],
                "radial_xy": [(ld.position, ld.component(LoadPlane.XY))],
                "axial": [], "moments_xz": [], "moments_xy": [],
            })

        for ld in shaft_system.distributed_radial_loads:
            if ld.x_hi <= self._x_lo or ld.x_lo >= self._x_hi:
                continue
            x_lo_eff = max(ld.x_lo, self._x_lo)
            x_hi_eff = min(ld.x_hi, self._x_hi)
            theta_fn = ld._theta_fn if ld._theta_variable else None
            cases.append({
                "label": ld.label or f"dist@[{x_lo_eff:.1f},{x_hi_eff:.1f}]",
                "source": ld.source,
                "distributed_xz": [{
                    "x_lo": x_lo_eff, "x_hi": x_hi_eff,
                    "q": lambda x, _ld=ld: _ld.component_intensity(x, LoadPlane.XZ),
                    "theta_fn": theta_fn,
                }],
                "distributed_xy": [{
                    "x_lo": x_lo_eff, "x_hi": x_hi_eff,
                    "q": lambda x, _ld=ld: _ld.component_intensity(x, LoadPlane.XY),
                    "theta_fn": theta_fn,
                }],
                "radial_xz": [], "radial_xy": [],
                "axial": [], "moments_xz": [], "moments_xy": [],
            })

        for ld in shaft_system.axial_loads:
            if not (self._x_lo - SOLVER_TOLERANCE
                    <= ld.position
                    <= self._x_hi + SOLVER_TOLERANCE):
                continue
            cases.append({
                "label": ld.label or f"axial@{ld.position:.1f}",
                "source": ld.source,
                "radial_xz": [], "radial_xy": [],
                "axial": [(ld.position, ld.magnitude)],
                "moments_xz": [], "moments_xy": [],
            })

        for m in shaft_system.external_moments:
            if not (self._x_lo - SOLVER_TOLERANCE
                    <= m.position
                    <= self._x_hi + SOLVER_TOLERANCE):
                continue
            cases.append({
                "label": m.label or f"moment@{m.position:.1f}",
                "source": m.source,
                "radial_xz": [], "radial_xy": [], "axial": [],
                "moments_xz": [(m.position, m.component(LoadPlane.XZ))],
                "moments_xy": [(m.position, m.component(LoadPlane.XY))],
            })

        return cases

    # ------------------------------------------------------------------
    # Load vector assembly (u, v, theta per node)
    # ------------------------------------------------------------------

    def _assemble_distributed_load_vector(
        self,
        x_nodes: list[float],
        elements: list[Elem],
        x_lo: float,
        x_hi: float,
        q: Callable[[float], float],
        builder: StiffnessMatrixBuilder,
        theta_fn: Callable[[float], float] | None = None) -> np.ndarray:

        beam = builder.beam
        f    = np.zeros(3 * len(x_nodes))

        for elem in elements:
            x_a = x_nodes[elem.idx_node_1]
            x_b = x_nodes[elem.idx_node_2]

            if x_b <= x_lo or x_a >= x_hi:
                continue

            x_lo_elem = max(x_a, x_lo)
            x_hi_elem = min(x_b, x_hi)

            x_map  = beam.global_to_natural_radial(x_lo_elem, x_hi_elem, elem)
            q_zeta = beam.vetor_global_to_natural(q, x_map)
            J      = beam.jacobian(elem)

            n_gauss = beam.gauss_order(q, x_lo_elem, x_hi_elem, elem, theta_fn=theta_fn)
            gauss_pts, gauss_wts = beam.gauss_quadrature(n_gauss)

            f_elem = np.zeros(6)
            for xi, w in zip(gauss_pts, gauss_wts):
                N = beam.shape_functions(xi, elem)
                f_elem[1] += N[1] * q_zeta(xi) * J * w
                f_elem[4] += N[4] * q_zeta(xi) * J * w

            f[3 * elem.idx_node_1 + 1] += f_elem[1]
            f[3 * elem.idx_node_2 + 1] += f_elem[4]

        return f

    def _assemble_load_vector(
        self,
        x_nodes: list[float],
        radial: list,
        axial: list,
        moments: list) -> np.ndarray:

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
    # Constraints definitions
    # ------------------------------------------------------------------

    def _build_constraint_matrix(
        self,
        x_nodes: list[float],
    ) -> tuple[np.ndarray, list[int]]:
        """
        Build constraint matrix C of shape (n_dofs, n_dofs).

        One row per DOF — identity rows for prescribed DOFs,
        zero rows for free DOFs.

        Returns
        -------
        C      : (n_dofs, n_dofs)
        p_dofs : indices of prescribed DOFs
        """
        n_dofs = 3 * len(x_nodes)

        i_lo = Elem.find_node_index(x_nodes, self._x_lo)
        i_hi = Elem.find_node_index(x_nodes, self._x_hi)

        p_dofs = [
            3 * i_lo,
            3 * i_lo + 1,
            3 * i_lo + 2,
            3 * i_hi,
            3 * i_hi + 1,
            3 * i_hi + 2,
        ]

        C = np.zeros((n_dofs, n_dofs))
        for dof in p_dofs:
            C[dof, dof] = 1.0

        return C, p_dofs


    def _build_prescribed_vectors(
        self,
        x_nodes: list[float],
        bc_data: dict[float, dict[str, float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build prescribed displacement vectors of shape (n_dofs,).
        Zero at free DOFs, prescribed value at constrained DOFs.

        Returns
        -------
        q_xz : (n_dofs,)
        q_xy : (n_dofs,)
        """
        n_dofs = 3 * len(x_nodes)
        q_xz = np.zeros(n_dofs)
        q_xy = np.zeros(n_dofs)

        i_lo = Elem.find_node_index(x_nodes, self._x_lo)
        i_hi = Elem.find_node_index(x_nodes, self._x_hi)

        q_xz[3 * i_lo]     = bc_data[self._x_lo]["u"]
        q_xz[3 * i_lo + 1] = bc_data[self._x_lo]["v_xz"]
        q_xz[3 * i_lo + 2] = bc_data[self._x_lo]["theta_xz"]
        q_xz[3 * i_hi]     = bc_data[self._x_hi]["u"]
        q_xz[3 * i_hi + 1] = bc_data[self._x_hi]["v_xz"]
        q_xz[3 * i_hi + 2] = bc_data[self._x_hi]["theta_xz"]

        q_xy[3 * i_lo]     = bc_data[self._x_lo]["u"]
        q_xy[3 * i_lo + 1] = bc_data[self._x_lo]["v_xy"]
        q_xy[3 * i_lo + 2] = bc_data[self._x_lo]["theta_xy"]
        q_xy[3 * i_hi]     = bc_data[self._x_hi]["u"]
        q_xy[3 * i_hi + 1] = bc_data[self._x_hi]["v_xy"]
        q_xy[3 * i_hi + 2] = bc_data[self._x_hi]["theta_xy"]

        return q_xz, q_xy