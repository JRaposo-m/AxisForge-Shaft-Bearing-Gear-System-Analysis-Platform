# axisforge.solvers

Numerical solvers of AxisForge: the static finite-element solution of a shaft on rigid bearing
supports (bending in two planes, axial force, torsion, sub-model), stress concentration at the
shoulders and keyways, the ISO/TS 16281 internal load distribution of rolling bearings with the
basic reference rating life, and the Grid Convergence Index of a mesh refinement. A solver reads
objects from `core`, `mesh` and `config` and returns the containers of `results`. The package does
not hold a library of results, does not print and does not plot; reports belong to `outputs`
and studies to the repositories that use AxisForge. There is no gear solver here: gear geometry
is in `core`, and `machine_elements/gears/` holds two placeholders that compute nothing.

## Contents

```
solvers/
    machine_elements/
        shaft/
            utils.py                       Marin factors, shoulder Kt, Neuber constant, notch sensitivity
            fem_solvers/                   rollup of the shaft FEM solvers (explicit __all__)
                assembly/
                    build_stiffness_matrix.py     StiffnessMatrixBuilder
                    load_assembly/                point loads, distributed loads, load cases
                    numerics/                     Gauss quadrature, conditioning check
                constraints/
                    boundary_conditions.py        free and constrained DOFs for rigid supports
                    submodel_extraction.py        values of the nodes of a sub-domain
                element_theories/
                    element_postprocessing.py     M, Q, strains per beam theory
                global_solver/
                    rigid_support.py              RigidSupportFEMSolver, RigidSupportSolution
                    torsion.py                    TorsionSolver
                    global_postprocessing.py      build_shaft_result (ShaftResults)
                submodel_solver/
                    lagrange_multipliers.py       SubmodelSolver (cut nodes by Lagrange multipliers)
                    submodel_postprocessing.py    build_submodel_result (SubmodelResult)
            static_solvers/
                postprocessing.py          ShaftPostProcessor, StressConcentration
                static_failure.py          empty
        gears/                             placeholders, not reviewed: every function raises NotImplementedError
            iso6336.py                     contracts of the ISO 6336 load-capacity calculation
            thermal.py                     contracts of the ISO/TR 14179-2 power loss and heat balance
        bearings/load_distribution/iso_16281/
            dispatch.py                    solver registry, selection by family capability
            numerics.py                    run_root, SciPy root-finder wrapper
            validation.py                  check_bearing_ready
            contact_solver.py              ball and roller solvers, single-row and multi-row
            contact_postprocessing.py      stiffness, rolling-element loads, rating life
    mesh/
        convergence_solver.py              Richardson extrapolation, GCI, MeshConvergenceStudy
```

Tests are in `tests/` folders next to the code (`shaft/fem_solvers/tests`, `iso_16281/tests`, `gears/tests`, `mesh/tests`). `solvers/mesh` is the mesh-convergence solver;
mesh generation and elements are in `axisforge.mesh`.

## Entry points

`axisforge.solvers` re-exports every name below eagerly, except `ShaftPostProcessor` and the
functions of `shaft.utils`, which are imported from their module. Arguments, units and raised
errors are in the docstrings; see the class docstrings for the attributes of each result.

| Task | Call | Returns |
|---|---|---|
| Shaft on rigid supports | `RigidSupportFEMSolver(settings, distribute_gear_labels=None, kGA_override=None).solve(shaft_system, extra_mandatory=None)` | `ShaftResults` (nothing is kept on the solver) |
| Torsion | `TorsionSolver().solve(shaft_system, x_nodes)`; `.validate_equilibrium(shaft_system, tol=1e-6)` | `(T, tau, phi, contributions)`; list of messages |
| Sub-model of an interval | `SubmodelSolver(x_lo, x_hi).solve(shaft_results, shaft_system, settings, grade, *, kGA_override=None)` | `SubmodelResult` |
| Raw global solution, no post-processing | `RigidSupportSolution`; `build_shaft_result(solution, shaft_system)`; `postprocessor_for_global(solution)` | dataclass; `ShaftResults`; post-processor |
| Stiffness matrix | `StiffnessMatrixBuilder(mesh, elements, frame)`; `.add_nodal_stiffness(x, K_local, dof_slots, node_tol=None)`; `.build(*, kGA_override=None)` | matrix |
| Stress concentration | `static_solvers.postprocessing.ShaftPostProcessor(shaft_system, results).process()` | `PostProcessedResults` |
| Fatigue and notch factors | `shaft.utils`: `ka_surface_finish`, `kb_size`, `kc_load`, `kd_temperature`, `ke_reliability`, `kt_shoulder_bending`, `kt_shoulder_torsion`, `neuber_constant_sqrt_a`, `kf_from_kt` | float |
| Bearing load distribution | `ISO16281BallSolver` or `ISO16281RollerSolver` `(tol=SOLVER_TOLERANCE, psi_input=False, postprocess=False).solve(shaft_system, bearings, shaft_results, psi_override=None)` | `dict[label, BearingAnalysisResult]` |
| Solver of a bearing | `resolve_solver_cls(bearing, *, label="")` | solver class; `SolverDispatchError` if none matches |
| Readiness of a bearing | `check_bearing_ready(bearing, label, capability)` | None, or RuntimeError |
| Stiffness and rating life | `bearing_stiffness`, `ball_basic_reference_rating_life`, `roller_basic_reference_rating_life`, `combine_row_L10r`, `stress_riser_factor` | see the docstrings |
| Mesh convergence | `MeshConvergenceStudy(requests, gci_threshold=0.01, safety_factor=1.25, min_p=None, max_p=None)` with `.new_record`, `.add_level`, `.finalize_unconverged`; `build_mesh_refinement_result(shaft_name, per_load)` | `ConvergenceRecord`; `MeshRefinementResult` |
| Richardson and GCI of one triplet | `RichardsonGCI`; `register_eval_form`, `resolve_eval_form` | see the docstrings |

`RigidSupportFEMSolver.solve` gives the `ShaftResults` of the global model, torsion included
(through `build_shaft_result`). `SubmodelSolver.solve` re-solves an interval on a finer mesh and
`MeshConvergenceStudy.add_level` collects the levels. The bearing solvers read the bearing nodes
of `ShaftResults`. `ShaftResults`, `SubmodelResult`, `BearingAnalysisResult`, `ConvergenceRecord`
and `MeshRefinementResult` are defined in `axisforge.results`.

## Conventions

* Layer rule: solvers import `axisforge.core`, `axisforge.mesh`, `axisforge.results` and
  `axisforge.config`. The rule of the package is that no solver imports another solver's
  internals.
* Variants register themselves: contact solvers with `@register_contact_solver(capability=...)`,
  mesh evaluation forms with `@register_eval_form(name, needs_point=...)`. The post-processor of
  a beam theory is chosen from `elements[0].beam_theory`.
* Units: length mm, force N, moment N mm, stiffness N/mm, stress MPa, angles rad. Torque is in N m
  in `TorsionSolver` and in `ShaftResults.T`.
* Frame (D-0010): `x` is the shaft axis; the plane suffix `_xz` is the horizontal plane (along Z)
  and `_xy` the vertical plane (along Y). Three DOFs per node, in the order `[u, v, theta]`.
  The two planes use the same reduced matrix; the axial load enters the XZ vector only.
* The bearing load direction is `phi_Fr = arctan2(Fr_xy, Fr_xz)` [rad], from +Z toward +Y; it
  equals `pi/2 - theta` of the load angle of the root README.
* Signs: `M = +E I d2v/dx2` and `dM/dx = -Q`; the torque sign is fixed upstream by the gear
  system, and the solvers accumulate it from left to right.
* Boundary conditions are imposed by elimination (no penalty); reactions are `K d - f_ext`.
  Bearings are rigid supports in the shaft solver; bearing stiffness is not used there. Frame
  elements (axial and bending) are always assembled.
* A failed bearing solve is a result, not an exception: `n_iter` (the number of residual
  evaluations, `nfev`), `residual` and `ok` are stored; the consumer checks `ok`.
* The multi-row bearing solvers are not part of ISO/TS 16281, which prints no multi-row model.
  `postprocess=True` with a multi-row solver raises `NotImplementedError`.
* The stresses of `ShaftResults` are nominal. The sources of the stress-concentration and Marin
  factors are named in the docstrings of `static_solvers/postprocessing.py` and `shaft/utils.py`.
* `solvers`, `machine_elements`, `shaft`, `shaft/fem_solvers`, `bearings`, `iso_16281` and `mesh`
  re-export their public names eagerly with an explicit `__all__`.

## Standards

Library keys (D-0023). Clauses and equations are in the docstrings.

| Key | Used for |
|---|---|
| `ISO_TS_16281_2008` | Internal load distribution of ball and roller bearings, rating life |
| `ISO_TS_16281_2008_Cor1_2009` | Corrigendum to the above; Eq. (19)-(20) as corrected, cited in the ball solver post-processing (`contact_solver.py`) |
| `Onate2013_vol2` | Beam elements, load assembly, quadrature, torsion, post-processing |
| `Harris2007_vol1`, `Harris2007_vol2` | Rating-life product law and Weibull slopes (vol. 1); bearing load distribution and beam sign (vol. 2) |
| `Zaretsky2016` | Combination of row lives (`combine_row_L10r`), Eq. (49a) and (C11) |
| `Hernot2000` | Secant stiffness matrix of an angular-contact ball bearing |
| `Cao2004` | Bearing as a two-node element (cited in `boundary_dofs`, `StiffnessMatrixBuilder`) |
