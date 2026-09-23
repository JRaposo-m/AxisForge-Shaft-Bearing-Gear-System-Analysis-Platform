"""
axisforge/solvers/machine_elements/shaft/fem_solvers/submodel_solver/lagrange_multipliers.py

Submodel solver for Richardson GCI convergence study.

Restricts a solve to a subdomain [x_lo, x_hi], with the global solution
injected as prescribed displacement BCs at the two cut nodes via
Lagrange multipliers. Used exclusively by MeshConvergenceStudy -- not
part of the production solve pipeline.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

import numpy as np

from axisforge.mesh.shaft.beam_model_settings import BeamModelSettings
from axisforge.mesh.shaft.element_type.elem import Elem
from axisforge.mesh.shaft.mesh_generation.mesh_1D import Mesh1D
from axisforge.mesh.shaft.mesh_generation.mesh_grade import Grader
from axisforge.config import SOLVER_TOLERANCE
from axisforge.core.loads import LoadPlane

from axisforge.solvers.machine_elements.shaft.fem_solvers.assembly.build_stiffness_matrix import (
    StiffnessMatrixBuilder,
)
from axisforge.solvers.machine_elements.shaft.fem_solvers.assembly.load_assembly.point_loads import (
    assemble_point_load_vector,
)
from axisforge.solvers.machine_elements.shaft.fem_solvers.assembly.load_assembly.distributed_loads import (
    assemble_distributed_load_vector,
)
from axisforge.solvers.machine_elements.shaft.fem_solvers.assembly.numerics.gauss_quadrature import (
    QuadratureOrderEstimator,
)
from axisforge.solvers.machine_elements.shaft.fem_solvers.constraints.submodel_extraction import (
    extract_submodel_values,
)
from axisforge.solvers.machine_elements.shaft.fem_solvers.submodel_solver.submodel_postprocessing import (
    build_submodel_result,
)

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.results.fem_results.shaft_results import ShaftResults
    from axisforge.results.fem_results.submodel_results import SubmodelResult


# ===========================================================================
# Data container
# ===========================================================================

@dataclass
class SubmodelSolution:
    """Raw solve output for the submodel within [x_lo, x_hi] -- pre-postprocessing.
    See SubmodelResult in results/fem_results/submodel_results.py for the
    postprocessed, engineering-quantity shape built FROM this one, via
    submodel_solver/postprocessing.py."""

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

    elements: list[Elem] = field(default_factory=list)
    """Local element list for this grade's subdomain mesh (same objects
    used to build K_sub). Required by any post-hoc postprocessing call
    (TimoshenkoPostProcessing.bending_moment()/shear_force()) that needs
    element geometry/rigidity plus a reconstructable a_e -- d_xz/d_xy
    alone are not enough for that."""

    kGA_override: float | None = None
    """Carried over unchanged from the kGA_override this SubmodelSolution
    was solved with -- published (not just consumed internally) so that
    submodel_solver/postprocessing.py can hand it to
    TimoshenkoPostProcessing.shear_force() the same way the global
    pipeline does, without the caller having to remember and re-thread
    the original solve()'s argument by hand."""


# ===========================================================================
# Submodel solver
# ===========================================================================

class _SubdomainMesh(Mesh1D):
    """
    Synthetic Mesh1D for the subdomain [x_lo, x_hi]. Bypasses
    _create_mesh -- uses the already-filtered nodes directly. Used
    exclusively by SubmodelSolver.
    """
    def __init__(self, shaft_system, x_nodes_sub: list[float]):
        super().__init__(shaft_system)
        self._x_nodes = x_nodes_sub


class SubmodelSolver:
    """
    Solves a subdomain [x_lo, x_hi] of shaft_system, with the global
    solution (given as a ShaftResults) injected as prescribed
    displacement BCs at the two cut nodes, and returns the postprocessed
    SubmodelResult directly -- solve() does both steps itself (see
    module docstring's second CHANGED note for why this solver, unlike
    RigidSupportFEMSolver, folds postprocessing into the same call
    instead of leaving it to a separate caller).
    """
    def __init__(self, x_lo: float, x_hi: float):
        self._x_lo = x_lo
        self._x_hi = x_hi

    def solve(self,
              shaft_results: "ShaftResults",
              shaft_system,
              settings: BeamModelSettings,
              grade: str,
              *,
              kGA_override: float | None = None) -> "SubmodelResult":

        x_lo = self._x_lo
        x_hi = self._x_hi

        # 1. shaft_results must already be a solved result, not a fresh/empty one
        if shaft_results.d_total_xz is None or len(shaft_results.d_total_xz) == 0:
            raise RuntimeError(
                "SubmodelSolver.solve() requires an already-solved ShaftResults "
                "(shaft_results.d_total_xz is empty/None). Run RigidSupportFEMSolver "
                "+ ShaftResultsReader (or equivalent) first, and pass the result here "
                "-- SubmodelSolver never solves the global model itself."
            )

        # 2. BC data at the cut nodes, via the free extraction function --
        #    not a solver method (see module docstring)
        bc_data = extract_submodel_values(
            shaft_results.x_nodes, [x_lo, x_hi],
            shaft_results.d_total_xz, shaft_results.d_total_xy,
            shaft_results.f_xz_total, shaft_results.f_xy_total,
        )

        # 3. grade -> candidate nodes within the subdomain
        grader     = Grader(x_lo, x_hi, shaft_results.x_nodes)
        candidates = grader.get_grade(grade)

        # 4. global mesh with the extra candidates (ensures they land on the mesh)
        mesh_full = Mesh1D(shaft_system, extra_mandatory=candidates)

        # 5. filter to the subdomain's own nodes
        x_nodes_sub = [
            x for x in mesh_full.x_nodes
            if x_lo - SOLVER_TOLERANCE <= x <= x_hi + SOLVER_TOLERANCE
        ]

        x_nodes_sub = sorted(set(x_nodes_sub))
        load_cases = self._build_submodel_load_cases(shaft_system)

        # 6. synthetic subdomain mesh + its own elements (settings passed
        #    explicitly now -- no more global_solver._settings reach-through)
        mesh_sub = _SubdomainMesh(shaft_system, x_nodes_sub)
        x_nodes  = mesh_sub.x_nodes
        elements = Elem.from_x_nodes(x_nodes_sub, shaft_system, settings)

        # K_sub via the real builder -- no manual element loop, no
        # out-of-range skip needed (elements are already submodel-only)
        K_sub = StiffnessMatrixBuilder(mesh_sub, elements, frame=True).build(
            kGA_override=kGA_override,
        )

        # --- force vectors, via the real assembly primitives ---
        # load_cases already built above -- reused here as-is, not rebuilt.
        estimator  = QuadratureOrderEstimator(settings.beam_theory)

        n_dofs = 3 * len(x_nodes)
        f_xz   = np.zeros(n_dofs)
        f_xy   = np.zeros(n_dofs)

        for lc in load_cases:
            f_xz += assemble_point_load_vector(
                x_nodes, lc["radial_xz"], lc["axial"], lc["moments_xz"],
            )
            f_xy += assemble_point_load_vector(
                x_nodes, lc["radial_xy"], [], lc["moments_xy"],
            )

            if lc.get("distributed_xz"):
                f_xz += assemble_distributed_load_vector(
                    x_nodes, elements, lc["distributed_xz"], estimator,
                )
            if lc.get("distributed_xy"):
                f_xy += assemble_distributed_load_vector(
                    x_nodes, elements, lc["distributed_xy"], estimator,
                )

        # --- constraint matrix and prescribed vectors ---
        C, p_dofs = self._build_constraint_matrix(x_nodes)
        q_xz, q_xy = self._build_prescribed_vectors(x_nodes, bc_data)

        n   = n_dofs
        n_c = C.shape[0]

        K_aug = np.zeros((n + n_c, n + n_c))
        K_aug[:n, :n] = K_sub
        K_aug[:n, n:] = C.T
        K_aug[n:, :n] = C

        rhs_xz = np.zeros(n + n_c)
        rhs_xz[:n] = f_xz
        rhs_xz[n:] = q_xz

        rhs_xy = np.zeros(n + n_c)
        rhs_xy[:n] = f_xy
        rhs_xy[n:] = q_xy

        sol_xz = np.linalg.solve(K_aug, rhs_xz)
        sol_xy = np.linalg.solve(K_aug, rhs_xy)

        d_xz   = sol_xz[:n]
        d_xy   = sol_xy[:n]
        lam_xz = sol_xz[n:]
        lam_xy = sol_xy[n:]

        solution = SubmodelSolution(
            x_lo=self._x_lo,
            x_hi=self._x_hi,
            grade=grade,
            x_nodes=x_nodes,
            d_xz=d_xz,
            d_xy=d_xy,
            lam_xz=lam_xz,
            lam_xy=lam_xy,
            elements=elements,
            kGA_override=kGA_override,
        )

        # solve() returns the postprocessed result directly -- see the
        # module docstring's second CHANGED note. SubmodelSolution stays
        # a local, unreturned intermediate.
        return build_submodel_result(solution)

    # ------------------------------------------------------------------
    # Load-case construction (submodel-specific: filters + clamps to
    # [x_lo, x_hi]; NOT a duplicate of build_load_cases() -- see module
    # docstring) -- unchanged from the previous draft.
    # ------------------------------------------------------------------

    def _build_submodel_load_cases(self, shaft_system) -> list[dict]:
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
    # Constraints (Lagrange multipliers at the two cut nodes) --
    # unchanged from the previous draft.
    # ------------------------------------------------------------------

    def _build_constraint_matrix(
        self,
        x_nodes: list[float],
    ) -> tuple[np.ndarray, list[int]]:

        n_dofs = 3 * len(x_nodes)

        i_lo = Elem.find_node_index(x_nodes, self._x_lo)
        i_hi = Elem.find_node_index(x_nodes, self._x_hi)

        p_dofs = [
            3 * i_lo,      3 * i_lo + 1,  3 * i_lo + 2,
            3 * i_hi,      3 * i_hi + 1,  3 * i_hi + 2,
        ]

        n_c = len(p_dofs)
        C = np.zeros((n_c, n_dofs))
        for row, dof in enumerate(p_dofs):
            C[row, dof] = 1.0

        return C, p_dofs

    def _build_prescribed_vectors(
        self,
        x_nodes: list[float],
        bc_data: dict,
    ) -> tuple[np.ndarray, np.ndarray]:

        q_xz = np.array([
            bc_data[self._x_lo]["u"],
            bc_data[self._x_lo]["v_xz"],
            bc_data[self._x_lo]["theta_xz"],
            bc_data[self._x_hi]["u"],
            bc_data[self._x_hi]["v_xz"],
            bc_data[self._x_hi]["theta_xz"],
        ])

        q_xy = np.array([
            bc_data[self._x_lo]["u"],
            bc_data[self._x_lo]["v_xy"],
            bc_data[self._x_lo]["theta_xy"],
            bc_data[self._x_hi]["u"],
            bc_data[self._x_hi]["v_xy"],
            bc_data[self._x_hi]["theta_xy"],
        ])

        return q_xz, q_xy