"""Static FEM solver of the whole shaft with every bearing modelled as a rigid point support
(constrained DOFs eliminated from the system). It does not model rolling-bearing physics
(ISO 281, ISO/TS 16281 are handled elsewhere); a compliant-bearing solver would be a sibling
module, not an option of this one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from axisforge.mesh.shaft.beam_model_settings import BeamModelSettings
from axisforge.mesh.shaft.mesh_generation.mesh_1D import Mesh1D
from axisforge.mesh.shaft.element_type.elem import Elem
from axisforge.core.mechanical_system.shaft_system import ShaftSystem

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
    """Raw output of the global shaft solve, before post-processing (data only, no behaviour).

    The field names ``elements``, ``d_total_xz``, ``d_total_xy`` and ``_kGA_override`` are the ones
    expected by ``ElementTheoryPostProcessor``, so the global post-processors need no adapter. The
    leading underscore of ``_kGA_override`` mirrors the private attribute of the solver.

    Parameters
    ----------
    x_nodes : list of float
        Node positions along the shaft axis [mm].
    elements : list of Elem
        Beam elements of the mesh.
    K : ndarray, shape (3n, 3n)
        Global stiffness matrix [N/mm, N, N mm/rad], DOF order ``[u, v, theta]`` per node.
    free_dofs, constrained_dofs : list of int
        Global DOF indices kept in / eliminated from the solve.
    d_total_xz, d_total_xy : ndarray, shape (3n,)
        Nodal displacements of the XZ and XY bending planes [mm, mm, rad] per ``[u, v, theta]``.
    f_xz_ext, f_xy_ext : ndarray, shape (3n,)
        Applied nodal loads (point loads plus consistent distributed loads) [N, N, N mm].
    f_xz_total, f_xy_total : ndarray, shape (3n,)
        ``K d`` for each plane [N, N, N mm]: applied load plus support reaction.
    f_xz_reaction, f_xy_reaction : ndarray, shape (3n,)
        ``K d - f_ext`` for each plane [N, N, N mm]; non-zero at constrained DOFs only (up to
        round-off).
    _kGA_override : float or None
        Shear rigidity k G A used in the solve [N], or ``None`` for the element values.

    Notes
    -----
    The axial DOF ``u`` is read from the XZ solution only: ``AxialLoad`` enters the XZ load vector
    and never the XY one.
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
        """Nodal displacements and forces of the nodes inside ``[x[0], x[1]]``.

        Parameters
        ----------
        x : list of float
            ``[x_lo, x_hi]``, interval bounds [mm].

        Returns
        -------
        dict
            See ``extract_submodel_values`` (keys, units and sign note).
        """
        return extract_submodel_values(
            self.x_nodes, x,
            self.d_total_xz, self.d_total_xy,
            self.f_xz_total, self.f_xy_total,
        )


class RigidSupportFEMSolver:
    """Static finite-element solution of a shaft on rigid bearing supports, bending in two planes plus
    axial force, followed by post-processing and torsion.

    Frame elements (axial plus bending) are always assembled, because ``AxialLoad`` is injected into
    the global force vector and the axial DOFs must therefore be wired into K. The two bending planes
    (XZ and XY) are solved separately with the same reduced matrix; the axial load goes to the XZ
    vector only.

    Parameters
    ----------
    settings : BeamModelSettings
        Beam theory (and, for Timoshenko, shear theory and integration method) of every element.
    distribute_gear_labels : set of str or None, optional
        Gear labels whose gear-mesh point loads (``RadialLoad`` with ``source="gear_mesh"``) are
        LEFT OUT of the load vector; ``{"*"}`` leaves out all of them. The solver does not create
        the distributed load: a ``DistributedRadialLoad`` must already be in ``shaft_system`` (gear
        system resolved with ``distribute=True``), and such loads are always assembled as
        distributed, whatever this argument. With a label flagged and no distributed counterpart,
        the radial gear force is missing from the solve. ``None`` or empty: every ``RadialLoad`` is
        assembled as a point load.
    kGA_override : float or None, optional
        Transverse shear rigidity k G A [N] that replaces the value of every Timoshenko element
        (``shear_theory`` is then ignored); no effect for Euler-Bernoulli. Intended for plugging in a
        value read from an Abaqus section-properties printout (``*Preprint, model=YES``). ``None``:
        element values.

    Notes
    -----
    Boundary conditions are imposed by elimination (``K_red = K[free, free]``, no penalty); the
    reactions are ``K d - f_ext`` (Onate2013_vol2, 1.3.4, Eq. (1.22), Examples 1.3-1.4). Bearings are
    rigid supports (see ``boundary_dofs``); bearing stiffness is not used by this solver.

    Examples
    --------
    >>> settings = BeamModelSettings(beam_theory="timoshenko", shear_theory="cowper",
    ...                              integration_method="exact")  # doctest: +SKIP
    >>> result = RigidSupportFEMSolver(settings).solve(shaft_system)  # doctest: +SKIP
    >>> result.d_total_xz  # doctest: +SKIP
    """

    # This solver always assembles full frame elements (axial + bending)
    # -- fixed, not a constructor argument (see the class docstring).
    _FRAME = True

    def __init__(self, settings: BeamModelSettings,
                 distribute_gear_labels: set[str] | None = None,
                 kGA_override: float | None = None):
        """Store the settings and the gear-load distribution flags (see the class docstring).
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
        """Solve the shaft for all loads and return the post-processed results.

        Steps: mesh, elements, global K, boundary conditions (elimination), conditioning check, load
        vectors (point loads and consistent distributed loads, per plane), solution of the XZ and XY
        planes, ``K d`` and reactions, then ``build_shaft_result`` (moment and shear force, bearing data,
        section properties and torsion).

        Parameters
        ----------
        shaft_system : ShaftSystem
            Shaft geometry, bearings and loads (lengths [mm], forces [N], moments [N mm]); validated
            first.
        extra_mandatory : list of float or None, optional
            Extra positions [mm] that must be mesh nodes.

        Returns
        -------
        ShaftResults
            Displacements [mm, rad], resultants, reactions, bearing-node data, stresses and twist; units
            in ``ShaftResults``.

        Raises
        ------
        numpy.linalg.LinAlgError
            If the reduced stiffness matrix is (near-)singular (``check_conditioning``).
        ValueError
            From ``boundary_dofs`` (missing axial fixed point or fewer than two radial supports) and from
            node lookup of loads and bearings; also if ``shaft_system`` is not valid
            (``validate_or_raise``).
        KeyError
            If a section material is not registered.

        Warnings
        --------
        ``TorsionSolver`` runs on every call, even if only bending and axial results are wanted.
        """
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

        # RigidSupportSolution is a local intermediate; build_shaft_result
        # turns it into the ShaftResults that solve() returns.
        return build_shaft_result(solution, shaft_system)