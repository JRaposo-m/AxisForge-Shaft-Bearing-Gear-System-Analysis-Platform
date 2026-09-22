"""
axisforge/solvers/machine_elements/shaft/fem_solvers/global_solver/rigid_support.py

RigidSupportFEMSolver -- models each bearing location as a rigid point
SUPPORT boundary condition. Knows nothing about rolling-bearing physics
(ISO 281 / ISO/TS 16281, handled elsewhere in AxisForge). "Rigid" is
this solver's identity, not one of several modes -- a compliant-bearing
solver is a sibling module, not a flag on this one.

MOVED (this pass): into global_solver/, sibling to submodel_solver/ --
same reasoning discussed for lagrange_multipliers.py: this and
SubmodelSolver are the same kind of thing (a solver that turns a
ShaftSystem/BeamModelSettings into raw DOFs), so they live at the same
folder depth now.

BREAKING CHANGE (this pass): solve() used to return None and publish
every intermediate quantity (x_nodes, elements, d_total_xz, ...) as
public attributes on the solver instance itself -- re-calling solve()
overwrote them in place. It now returns a ShaftResults directly --
same shift already made for SubmodelSolver
(submodel_solver/lagrange_multipliers.py), and now made consistent here
too: solve() builds the raw RigidSupportSolution internally, then calls
global_solver/postprocessing.py's build_shaft_result() on it (M/V
recovery, torsion, bearing reactions, section properties) and returns
the finished ShaftResults. RigidSupportSolution stays as a local,
unreturned intermediate, same role SubmodelSolution plays in
lagrange_multipliers.py. Every existing caller doing
    solver.solve(shaft_system)
    solver.d_total_xz
needs updating to
    result = solver.solve(shaft_system)
    result.d_total_xz
This solver is more likely than SubmodelSolver to already have callers
elsewhere in the codebase (validation/fem_studies) -- check before
relying on this file. This is also a bigger behavioural change than the
submodel case: solve() now always runs TorsionSolver too, so every call
pays for torsion even if a caller only wanted the bending/axial DOFs
(e.g. an optimization loop varying bearing stiffness) -- see the
"no solver imports another solver" note in postprocessing.py's
docstring if that cost turns out to matter later.

RigidSupportSolution's field names (elements, d_total_xz, d_total_xy,
_kGA_override) are chosen to match exactly what
element_theories/element_postprocessing.py's ElementTheoryPostProcessor
expects -- deliberately, so global_solver/postprocessing.py's
GlobalEulerBernoulliPostProcessing/GlobalTimoshenkoPostProcessing need
no view/adapter object, unlike submodel_solver's SubmodelSolution (which
chose shorter, locally-nicer names -- d_xz/d_xy -- and pays for that
with _SubmodelSolverView). The leading underscore on _kGA_override is
unusual for a public dataclass field; kept anyway for that exact-name
match, since it mirrors the private attribute name
RigidSupportFEMSolver already used internally for the same value.

Orchestration only: solve() calls, in order,
  Mesh1D -> Elem.from_mesh (BeamModelSettings-driven)
  -> StiffnessMatrixBuilder (frame=True, fixed -- see note below)
  -> boundary_dofs (constraints/boundary_conditions.py)
  -> load_cases.py -> assembly/load_assembly/point_loads.py +
     assembly/load_assembly/distributed_loads.py
  -> linear solve (still beam-theory-agnostic)
  -> returns RigidSupportSolution.

Torsion is NOT part of this solver -- it lives in global_solver/torsion.py
as a separate static analysis on the same ShaftSystem/x_nodes, called
from global_solver/postprocessing.py's build_shaft_result() (not from
solve() itself) -- see that module's docstring for why it lives there
and not here. This solver only produces bending + axial results.

Why frame=True is fixed, not a caller choice: this solver accepts
AxialLoad and injects it into the global force vector, so the axial
DOFs MUST be wired into K for the system to be solvable at all --
frame=False would leave every axial DOF disconnected (see
build_stiffness_matrix.py's own docstring on that). Since there is
only one correct choice for what this solver does, making it a
constructor argument would just be one more way to misconfigure it, so
it is a fixed identity property instead -- same reasoning as "rigid"
in the class name.

FLAGGED, not verified in this pass -- these calls are carried forward
UNCHANGED in shape from the previous skeleton because their source
files were not available to check against BeamModelSettings/frame/the
new distributed_loads.py API:
  - assemble_point_load_vector()   (assembly/load_assembly/point_loads.py)
  - extract_submodel_values()      (constraints/submodel_extraction.py)
Confirm each of these still matches this call shape before relying on
this file -- I have not seen their current content in this
conversation. check_conditioning() (numerics/numerical_guards.py) HAS
been confirmed -- takes only K_red, no tolerance argument, fixed
1e14 condition-number threshold.

DOF convention (per node, 3 DOF): [u (axial), v (transverse), theta
(bending)]. Two INDEPENDENT planar solves are performed (XZ, XY),
reusing the same K. Axial DOF is shared between planes -- AxialLoad is
injected only into the XZ load vector, never XY, to avoid double-
counting it.

DROPPED (this pass): the previous version stored an unused
`self._builder = builder` on the instance, left over with no reader
anywhere -- removed, since RigidSupportSolution has no use for the
builder object itself, only its output (K).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from axisforge.mesh.shaft.beam_model_settings import BeamModelSettings
from axisforge.mesh.shaft.mesh_generation.mesh_1D import Mesh1D
from axisforge.mesh.shaft.element_type.elem import Elem
from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import ShaftSystem

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
from axisforge.solvers.machine_elements.shaft.fem_solvers.assembly.numerics.numerical_guards import (
    check_conditioning,
)
from axisforge.solvers.machine_elements.shaft.fem_solvers.assembly.load_assembly.vector_external_forces import (
    build_load_cases,
)
from axisforge.solvers.machine_elements.shaft.fem_solvers.constraints.boundary_conditions import (
    boundary_dofs,
)
from axisforge.solvers.machine_elements.shaft.fem_solvers.constraints.submodel_extraction import (
    extract_submodel_values,
)
from axisforge.solvers.machine_elements.shaft.fem_solvers.global_solver.global_postprocessing import (
    build_shaft_result,
)

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.results.fem_results.shaft_results import ShaftResults


@dataclass
class RigidSupportSolution:
    """
    Raw solve output for the whole shaft -- pre-postprocessing. Mirrors
    SubmodelSolution (submodel_solver/lagrange_multipliers.py): data,
    not behaviour. See module docstring for why the field names match
    element_postprocessing.py exactly.
    """

    x_nodes: list[float]
    elements: list[Elem]

    K: np.ndarray
    free_dofs: list[int]
    constrained_dofs: list[int]

    d_total_xz: np.ndarray
    d_total_xy: np.ndarray
    f_xz_ext: np.ndarray
    f_xy_ext: np.ndarray
    f_xz_total: np.ndarray
    f_xy_total: np.ndarray
    f_xz_reaction: np.ndarray
    f_xy_reaction: np.ndarray

    _kGA_override: float | None = None

    def return_values(self, x: list[float]) -> dict[float, dict[str, float]]:
        """
        Same job RigidSupportFEMSolver.return_values() always did --
        kept as a method here (not a free function) to preserve the
        existing call shape for whoever already calls
        solver.return_values(x); the underlying logic
        (extract_submodel_values) is unchanged, only where it's called
        from moved (solver -> solution).
        """
        return extract_submodel_values(
            self.x_nodes, x,
            self.d_total_xz, self.d_total_xy,
            self.f_xz_total, self.f_xy_total,
        )


class RigidSupportFEMSolver:
    """
    Usage
    -----
        settings = BeamModelSettings(beam_theory="timoshenko",
                                      shear_theory="cowper",
                                      integration_method="exact")
        solver = RigidSupportFEMSolver(settings)
        result = solver.solve(shaft_system)
        result.d_total_xz      # -> np.ndarray, global displacement (XZ)
        result.f_xz_total      # -> np.ndarray, global force (incl. reactions)
        result.M_xz, result.T, result.phi   # -> already postprocessed, torsion included

    solve() does both steps itself (raw solve + postprocessing) and
    returns the finished ShaftResults -- see module docstring's
    BREAKING CHANGE note for why, and for the cost this carries (torsion
    + full postprocessing runs on every call now, unconditionally).
    """

    # This solver always assembles full frame elements (axial + bending)
    # -- see module docstring for why this is fixed, not a constructor
    # argument.
    _FRAME = True

    def __init__(self, settings: BeamModelSettings,
                 distribute_gear_labels: set[str] | None = None,
                 kGA_override: float | None = None):
        """
        settings : BeamModelSettings -- decides beam_theory (and, for
                Timoshenko, shear_theory/integration_method) for every
                Elem this solver builds. No default on purpose: the
                caller must decide explicitly (see
                beam_model_settings.py).
        distribute_gear_labels : set of gear labels whose mesh loads
                should be treated as distributed over face width. None
                or empty -> all gear mesh loads as point loads. e.g.
                {"pinion", "wheel"} -> only those two distributed. Use
                {"*"} as a sentinel to distribute ALL gear mesh loads.
        kGA_override : if given, REPLACES the transverse shear
                stiffness K*G*A used for every Timoshenko element in
                this solve -- forwarded to
                StiffnessMatrixBuilder.build() (which assembles K).
                shear_theory is ignored for every element once this is
                set. Has no effect when settings.beam_theory ==
                "euler_bernoulli" -- Euler-Bernoulli has no shear term
                to override. None (default): normal path, unchanged
                behaviour.

                Use this to plug in a value read directly from
                Abaqus's own *Preprint, model=YES section-properties
                printout (e.g. "K*G(23)*A"/"K*G(13)*A" for a *Beam
                Section) to test whether matching Abaqus's ACTUAL
                transverse shear stiffness closes a deflection gap
                against Abaqus.
        """
        self._settings = settings
        self._distribute_all    = distribute_gear_labels == {"*"}
        self._distribute_labels = distribute_gear_labels or set()
        self._kGA_override = kGA_override

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def solve(self,
              shaft_system: ShaftSystem,
              extra_mandatory: list[float] | None = None) -> "ShaftResults":
        shaft_system.validate_or_raise()

        mesh = Mesh1D(shaft_system, extra_mandatory=extra_mandatory or [])
        x_nodes = mesh.x_nodes
        elements = Elem.from_mesh(mesh, self._settings)

        builder = StiffnessMatrixBuilder(mesh, elements, frame=self._FRAME)
        K = builder.build(kGA_override=self._kGA_override)

        free_dofs, constrained_dofs = boundary_dofs(x_nodes, shaft_system)
        K_red = K[np.ix_(free_dofs, free_dofs)]
        check_conditioning(K_red)

        load_cases = build_load_cases(
            shaft_system,
            distribute_all=self._distribute_all,
            distribute_labels=self._distribute_labels,
        )
        n_dofs = 3 * len(x_nodes)
        estimator = QuadratureOrderEstimator(self._settings.beam_theory)

        # --- assemble global load vectors ---
        f_xz_ext = np.zeros(n_dofs)
        f_xy_ext = np.zeros(n_dofs)

        for lc in load_cases:
            f_xz = assemble_point_load_vector(x_nodes, lc["radial_xz"],
                                               lc["axial"], lc["moments_xz"])
            f_xy = assemble_point_load_vector(x_nodes, lc["radial_xy"],
                                               [], lc["moments_xy"])

            if lc.get("distributed_xz"):
                f_xz += assemble_distributed_load_vector(
                    x_nodes, elements, lc["distributed_xz"], estimator,
                )
            if lc.get("distributed_xy"):
                f_xy += assemble_distributed_load_vector(
                    x_nodes, elements, lc["distributed_xy"], estimator,
                )

            f_xz_ext += f_xz
            f_xy_ext += f_xy

        # --- global solve (single system) ---
        d_total_xz = np.zeros(n_dofs)
        d_total_xy = np.zeros(n_dofs)
        d_total_xz[free_dofs] = np.linalg.solve(K_red, f_xz_ext[free_dofs])
        d_total_xy[free_dofs] = np.linalg.solve(K_red, f_xy_ext[free_dofs])

        f_xz_total = K @ d_total_xz
        f_xy_total = K @ d_total_xy

        solution = RigidSupportSolution(
            x_nodes=x_nodes,
            elements=elements,
            K=K,
            free_dofs=free_dofs,
            constrained_dofs=constrained_dofs,
            d_total_xz=d_total_xz,
            d_total_xy=d_total_xy,
            f_xz_ext=f_xz_ext,
            f_xy_ext=f_xy_ext,
            f_xz_total=f_xz_total,
            f_xy_total=f_xy_total,
            f_xz_reaction=f_xz_total - f_xz_ext,
            f_xy_reaction=f_xy_total - f_xy_ext,
            _kGA_override=self._kGA_override,
        )

        # solve() returns the postprocessed result directly -- see the
        # module docstring's BREAKING CHANGE note. RigidSupportSolution
        # stays a local, unreturned intermediate, same role
        # SubmodelSolution plays in lagrange_multipliers.py.
        return build_shaft_result(solution, shaft_system)