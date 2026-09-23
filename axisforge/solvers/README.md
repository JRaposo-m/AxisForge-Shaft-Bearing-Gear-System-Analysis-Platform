# axisforge/solvers — Module Reference

Solvers consume the domain objects in [`core/README.md`](../core/README.md) and the mesh objects in [`mesh/README.md`](../mesh/README.md), and publish their output into the containers defined in [`results/README.md`](../results/README.md). No solver imports another solver's internals — only the shared result containers.

← back to [project root](../../README.md)

---

## Table of Contents

- [Scope](#scope)
- [Status](#status)
- [Position in the architecture](#position-in-the-architecture)
- [Structure](#structure)
- [Import surface](#import-surface)
- [Conventions](#conventions)
- [Shaft FEM](#shaft-fem)
- [Torsion](#torsion)
- [Shaft results and post-processing](#shaft-results-and-post-processing)
- [Bearings — ISO/TS 16281](#bearings--isots-16281)
- [Gears](#gears)
- [Mesh convergence](#mesh-convergence)
- [Design contracts](#design-contracts)
- [Extending this package](#extending-this-package)
- [Standards referenced](#standards-referenced)

---

## Scope

| In scope | Out of scope |
|---|---|
| Numerical procedures: assembly, root solves, quadrature, extrapolation | The geometry and catalogue data they operate on (`core/`) |
| Recovery of engineering quantities from a raw solution | The shape those quantities are stored in (`results/`) |
| Dispatch of a bearing to the solver its family supports | Collecting many results into a run-level library (`fixtures/`) |
| Post-processing that requires a standard's formula | The standard's capacity formulae themselves (owned by the bearing family) |

A solver is a computation. It does not know how many shafts a study has, it does not keep a registry, and it does not print.

---

## Status

| Group | Module | Contents | Status |
|---|---|---|---|
| `shaft/fem_solvers/global_solver/` | `rigid_support.py` | `RigidSupportSolution`, `RigidSupportFEMSolver` | Implemented — constructed from a `BeamModelSettings` (see [`mesh/README.md`](../mesh/README.md)) instead of a bare theory string. **Moved** from `fem_solvers/rigid_support.py` (its old location) into `fem_solvers/global_solver/`, alongside torsion and the global-solve post-processing it is paired with. |
| `shaft/fem_solvers/global_solver/` | `torsion.py` | `TorsionSolver` | Implemented. **Moved** from `static_solvers/torsion.py` into `fem_solvers/global_solver/` — grouped with the global (non-submodel) solve rather than with the other static post-processors. |
| `shaft/fem_solvers/global_solver/` | `global_postprocessing.py` | `GlobalEulerBernoulliPostProcessing`, `GlobalTimoshenkoPostProcessing`, `postprocessor_for_global`, `build_shaft_result` | Implemented — **replaces** the `ShaftResultsReader` class entirely. `build_shaft_result(...)` is a function, dispatching to the matching per-theory post-processor via `postprocessor_for_global(...)`, in the same registry-dispatch style used elsewhere (`mesh`'s `_FORMULATION_REGISTRY`, `core`'s `_FAMILY_REGISTRY`) rather than a single reader class. See [Shaft results and post-processing](#shaft-results-and-post-processing). |
| `shaft/fem_solvers/assembly/` | `build_stiffness_matrix.py` | `StiffnessMatrixBuilder` | Implemented — theory-agnostic, `frame` is a required constructor argument (see [Shaft FEM](#shaft-fem)) |
| `shaft/fem_solvers/assembly/load_assembly/` | `point_loads.py`, `distributed_loads.py`, `vector_external_forces.py` | Point-load and distributed-load vector assembly | Present |
| `shaft/fem_solvers/assembly/numerics/` | `gauss_quadrature.py`, `numerical_guards.py` | `QuadratureOrderEstimator`, `check_conditioning`, plus `gauss_points_weights`, `q_degree`, `theta_degree` | Implemented |
| `shaft/fem_solvers/constraints/` | `boundary_conditions.py`, `submodel_extraction.py` | `boundary_dofs`, `boundary_dofs_explicit`, `extract_submodel_values` | Implemented |
| `shaft/fem_solvers/element_theories/` | `element_postprocessing.py` | `ElementTheoryPostProcessor` (base), `EulerBernoulliPostProcessing`, `TimoshenkoPostProcessing` | Implemented — one module, not the previous `timoshenko/`/`euler_bernoulli/` subfolder split |
| `shaft/fem_solvers/submodel_solver/` | `lagrange_multipliers.py` | `SubmodelSolution`, `SubmodelSolver` | Implemented — **new since the previous pass of this document.** The submodel solve is now a full Lagrange-multiplier solver, not just `extract_submodel_values()` feeding a reduced re-solve. Not yet described in narrative form in this document — flagged rather than guessed at; read the module directly before relying on its contract. |
| `shaft/fem_solvers/submodel_solver/` | `submodel_postprocessing.py` | `SubmodelEulerBernoulliPostProcessing`, `SubmodelTimoshenkoPostProcessing`, `postprocessor_for_submodel`, `build_submodel_result` | Implemented — mirrors the `global_solver` postprocessing pattern, builds a `SubmodelResult` (see [`results/README.md`](../results/README.md)). |
| `shaft/static_solvers/` | `postprocessing.py` | `ShaftPostProcessor`, `PostProcessedResults`, `StressConcentration` | Implemented |
| `shaft/static_solvers/` | `static_failure.py` | static failure criteria | Reserved — module exists, defines nothing |

> **Open item — `static_solvers/__init__.py` is broken.** It still lazily points `BearingNodeData`, `ShaftResults`, `SimpleFEMResultsLibrary` and `ShaftResultsReader` at a `.static_analysis` submodule that no longer exists in `static_solvers/` (that logic now lives in `fem_solvers/global_solver/global_postprocessing.py`, as functions, not as a `ShaftResultsReader` class). Any code that still does `from axisforge...static_solvers import ShaftResultsReader` will raise `ModuleNotFoundError` at that lazy `__getattr__`, not at package-import time — a fairly easy failure to miss until it is actually called. Flagged here rather than fixed, since fixing it is a code change, not a documentation one.
| `bearings/load_distribution/iso_16281/` | `dispatch.py`, `numerics.py`, `validation.py` | Solver resolution, root-solve wrapper, guards | Implemented |
| `bearings/load_distribution/iso_16281/` | `contact_solver.py` | `ISO16281BallSolver`, `ISO16281RollerSolver`, `ISO16281MultiRowBallSolverSharedDisplacement`, `ISO16281MultiRowRollerSolverSharedDisplacement` — single-row **and** multi-row solvers, one contact-type-per-class, in one module | Implemented — see the [Bearings](#bearings--isots-16281) section below for the shape of what `solve()` returns |
| `bearings/load_distribution/iso_16281/` | `contact_postprocessing.py` | Bearing stiffness (`ContactBearingStiffness`), basic reference rating life (`Ball/RollerBasicReferenceRatingLife`, `Ball/RollerDynamicEquivalentReferenceLoad`), stress riser factor | Implemented |
| `gears/` | `geometry.py`, `utils.py` | `GearSolver` and stateless helpers | Implemented |
| `gears/SpurHelicalGears/LoadCapacity_solver/` | `load_capacity.py` | ISO 6336 load capacity | Reserved — module exists, defines nothing |
| `mesh/` | `convergence_solver.py` | `MeshConvergenceStudy`, `RichardsonGCI`, `MeshRefinementResult`, `ConvergenceRecord` | Implemented |

> **Superseded structure.** An earlier pass of this package split bearing solving by row count first (`single_row/iso_16281/{ball_bearing,roller_bearing}/` versus `multi_row/thrust_bearings/iso_16281/{ball_bearings,roller_bearings}/`), each branch keeping its own local `results.py` under a name shared with, but structurally different from, the copies in `results/`. That split is gone: there is now one `bearings/load_distribution/iso_16281/` package, `contact_solver.py` covers single- and multi-row for both contact types, and every result is a `results/bearings/load_distribution/load_distribution_results.py` class — no solver package defines its own result shape any more. See [`results/README.md`](../results/README.md) for the container side of this.

---

## Position in the architecture

```
   config      core/      mesh/          domain and constants
      └───────────┼──────────┘
                  ▼
              solvers/                   this package
                  │  writes
                  ▼
              results/                   containers
                  │  reads
                  ▼
             fixtures/                   studies, libraries, reports
```

Two boundaries are strict and worth stating plainly:

- **Downward.** A solver may read `core/`, `mesh/`, `config` and `results/`. It never imports `fixtures/`.
- **Sideways.** A solver never imports another solver. The ball and roller stacks each own a full vertical slice — solve, post-process — and meet only at the dispatcher and at the result containers they share.

Run-level registries are not here. Results libraries and any run-level bearing orchestration live in `fixtures/studies/`: collecting results across a whole gearbox is orchestration, and orchestration is a study concern.

---

## Structure

```
solvers/
├── machine_elements/
│   ├── shaft/
│   │   ├── fem_solvers/
│   │   │   ├── rigid_support.py              RigidSupportFEMSolver — orchestrates the bending + axial solve
│   │   │   ├── assembly/
│   │   │   │   ├── build_stiffness_matrix.py         StiffnessMatrixBuilder (frame required)
│   │   │   │   ├── load_assembly/
│   │   │   │   │   ├── point_loads.py                assemble_point_load_vector
│   │   │   │   │   ├── distributed_loads.py           assemble_distributed_load_vector
│   │   │   │   │   └── vector_external_forces.py      build_load_cases
│   │   │   │   └── numerics/
│   │   │   │       ├── gauss_quadrature.py             QuadratureOrderEstimator
│   │   │   │       └── numerical_guards.py             check_conditioning
│   │   │   ├── constraints/
│   │   │   │   ├── boundary_conditions.py              boundary_dofs, boundary_dofs_explicit
│   │   │   │   └── submodel_extraction.py              extract_submodel_values
│   │   │   ├── element_theories/
│   │   │   │   └── element_postprocessing.py           ElementTheoryPostProcessor, Euler/TimoshenkoPostProcessing
│   │   │   ├── global_solver/
│   │   │   │   ├── rigid_support.py                    RigidSupportSolution, RigidSupportFEMSolver
│   │   │   │   ├── torsion.py                          TorsionSolver
│   │   │   │   └── global_postprocessing.py            build_shaft_result, postprocessor_for_global,
│   │   │   │                                           GlobalEulerBernoulliPostProcessing, GlobalTimoshenkoPostProcessing
│   │   │   └── submodel_solver/
│   │   │       ├── lagrange_multipliers.py             SubmodelSolution, SubmodelSolver
│   │   │       └── submodel_postprocessing.py          build_submodel_result, postprocessor_for_submodel,
│   │   │                                               Submodel{EulerBernoulli,Timoshenko}PostProcessing
│   │   └── static_solvers/
│   │       ├── postprocessing.py                       ShaftPostProcessor, StressConcentration
│   │       └── static_failure.py                       Reserved
│   ├── bearings/load_distribution/
│   │   └── iso_16281/
│   │       ├── dispatch.py                   Capability-based solver resolution
│   │       ├── numerics.py                   Root-solve wrapper
│   │       ├── validation.py                 Readiness and arrangement guards
│   │       ├── contact_solver.py             ISO16281BallSolver, ISO16281RollerSolver,
│   │       │                                 ISO16281MultiRowBallSolverSharedDisplacement,
│   │       │                                 ISO16281MultiRowRollerSolverSharedDisplacement
│   │       ├── contact_postprocessing.py     ContactBearingStiffness, Ball/RollerBasicReferenceRatingLife,
│   │       │                                 Ball/RollerDynamicEquivalentReferenceLoad, stress_riser_factor
│   │       └── tests/
│   └── gears/
│       ├── geometry.py                       GearSolver
│       ├── utils.py                          Involute helpers, rack constants
│       └── SpurHelicalGears/LoadCapacity_solver/load_capacity.py    Reserved
└── mesh/
    └── convergence_solver.py                 MeshConvergenceStudy, RichardsonGCI
```

The bearing tree is now split by **standard, then by contact type** — row count is a runtime property of a `BearingResult` (`n_rows >= 1`), not a package boundary. The shaft tree is split by **what it produces**: `fem_solvers/` for the bending + axial FEM solve and its assembly internals, `static_solvers/` for everything that post-processes a solved state without its own DOFs (torsion, stress recovery, the reserved static-failure module).

---

## Import surface

| Import from | Names |
|---|---|
| `...shaft.fem_solvers` (rollup) | Everything below, re-exported eagerly — see the package's own `__init__.py` for the authoritative list |
| `...shaft.fem_solvers.global_solver.rigid_support` | `RigidSupportSolution`, `RigidSupportFEMSolver` |
| `...shaft.fem_solvers.global_solver.torsion` | `TorsionSolver` |
| `...shaft.fem_solvers.global_solver.global_postprocessing` | `GlobalEulerBernoulliPostProcessing`, `GlobalTimoshenkoPostProcessing`, `postprocessor_for_global`, `build_shaft_result` |
| `...shaft.fem_solvers.submodel_solver.lagrange_multipliers` | `SubmodelSolution`, `SubmodelSolver` |
| `...shaft.fem_solvers.submodel_solver.submodel_postprocessing` | `SubmodelEulerBernoulliPostProcessing`, `SubmodelTimoshenkoPostProcessing`, `postprocessor_for_submodel`, `build_submodel_result` |
| `...shaft.fem_solvers.element_theories.element_postprocessing` | `ElementTheoryPostProcessor`, `EulerBernoulliPostProcessing`, `TimoshenkoPostProcessing` |
| `...shaft.fem_solvers.assembly.build_stiffness_matrix` | `StiffnessMatrixBuilder` |
| `...shaft.fem_solvers.assembly.load_assembly.point_loads` | `assemble_point_load_vector` |
| `...shaft.fem_solvers.assembly.load_assembly.distributed_loads` | `element_distributed_force_vector`, `assemble_distributed_load_vector` |
| `...shaft.fem_solvers.assembly.numerics.numerical_guards` | `check_conditioning` |
| `...shaft.fem_solvers.assembly.numerics.gauss_quadrature` | `gauss_points_weights`, `q_degree`, `theta_degree`, `QuadratureOrderEstimator` |
| `...shaft.fem_solvers.constraints.boundary_conditions` | `boundary_dofs`, `boundary_dofs_explicit` |
| `...shaft.fem_solvers.constraints.submodel_extraction` | `extract_submodel_values` |
| `...shaft.static_solvers.postprocessing` | `ShaftPostProcessor`, `PostProcessedResults`, `StressConcentration` — **not** `...static_solvers` itself, whose own `__init__.py` is broken (see the open item under [Status](#status)) |
| `...bearings.load_distribution.iso_16281.dispatch` | `resolve_solver_cls`, `resolve_solver_cls_for_attrs`, `register_contact_solver`, `SolverDispatchError` |
| `...bearings.load_distribution.iso_16281.contact_solver` | `ISO16281BallSolver`, `ISO16281RollerSolver`, `ISO16281MultiRowBallSolverSharedDisplacement`, `ISO16281MultiRowRollerSolverSharedDisplacement` |
| `...bearings.load_distribution.iso_16281.contact_postprocessing` | `ContactBearingStiffness`, `BallBasicReferenceRatingLife`, `RollerBasicReferenceRatingLife`, `BallDynamicEquivalentReferenceLoad`, `RollerDynamicEquivalentReferenceLoad`, `stress_riser_factor` |
| `...gears.geometry` | `GearSolver` |
| `axisforge.solvers.mesh.convergence_solver` | `MeshConvergenceStudy`, `RichardsonGCI`, `MeshRefinementResult`, `ConvergenceRecord` |

Ball and roller solvers are imported explicitly and never flattened into one namespace: point and line contact produce different result types, and flattening would hide which contact model a name belongs to.

---

## Conventions

| Symbol | Meaning |
|---|---|
| `x` | Axial coordinate, increasing left → right |
| `XZ` | Horizontal plane — tangential gear force `Wt` |
| `XY` | Vertical plane — radial gear force `Wr`, opposed to gravity |
| Torsion | Accumulates left → right; positive counter-clockwise viewed from `+x` |

Units, as stored: length and displacement in **mm**, force in **N**, bending moment in **N·mm**, torque in **N·m**, twist angle `phi` in **rad**, stress in **MPa**, stiffness in **N/mm**, angles in **rad**. The torque/bending-moment unit discontinuity is real and documented rather than normalised — see [`results/README.md`](../results/README.md#units).

Degrees of freedom are three per node: axial displacement, transverse displacement, rotation. Each bending plane is solved independently against the same stiffness matrix and superposed afterwards.

---

## Shaft FEM

### `RigidSupportFEMSolver`

`fem_solvers/global_solver/rigid_support.py` (moved here from `fem_solvers/rigid_support.py` — it now sits with `torsion.py` and `global_postprocessing.py`, the three modules that together produce one shaft's global-scale result). Models each bearing location as a rigid point **support** boundary condition — it knows nothing about rolling-bearing physics (ISO 281 / ISO/TS 16281, handled elsewhere). "Rigid" is this solver's identity, not one of several modes; a compliant-bearing solver would be a sibling module. The solve also now publishes a `RigidSupportSolution` object (new since the previous pass of this document) — **not yet described here in detail**; read the module directly for its exact field list before relying on it in place of the individual published attributes below, which remain accurate.

```python
settings = BeamModelSettings(beam_theory="timoshenko", shear_theory="cowper", integration_method="exact")
solver = RigidSupportFEMSolver(settings)
solver.solve(shaft_system)
solver.d_total_xz      # global displacement (XZ)
solver.f_xz_total       # global force, including reactions
```

Orchestration only. `solve()` calls, in order: `Mesh1D` → `Elem.from_mesh()` (driven by the `BeamModelSettings` passed at construction — see [`mesh/README.md`](../mesh/README.md) for how the beam formulation is resolved) → `StiffnessMatrixBuilder` (`frame=True`, fixed — see below) → `boundary_dofs` → the point- and distributed-load assemblers → a linear solve (still beam-theory-agnostic) → publishes public attributes.

| Published attribute | Contents |
|---|---|
| `x_nodes`, `elements` | The mesh actually solved. |
| `free_dofs`, `constrained_dofs` | Boundary condition partition. |
| `d_total_xz`, `d_total_xy` | Superposed displacement vectors per plane. |
| `f_xz_ext`, `f_xy_ext` | External force vectors per plane. |
| `f_xz_total`, `f_xy_total` | Total nodal force vectors, including reactions. |

Each attribute is `None` until `solve()` has run at least once, and re-calling `solve()` overwrites all of them in place — build a fresh instance if isolation between runs is needed.

**Why `frame=True` is fixed, not a caller choice.** This solver accepts `AxialLoad` and injects it into the global force vector, so the axial DOFs must be wired into `K` for the system to be solvable at all; `frame=False` would leave every axial DOF disconnected. Since there is only one correct choice for what this solver does, it is a fixed identity property (`_FRAME = True`), not a constructor argument.

**Torsion is not produced here.** `T` and `tau` come from `TorsionSolver`, called separately — see [Torsion](#torsion).

### `StiffnessMatrixBuilder`

`fem_solvers/assembly/build_stiffness_matrix.py`. Assembles the global stiffness matrix from element contributions, plus optional nodal stiffness injected at chosen positions (bearings, or any other spring). Theory-agnostic — each `Elem` already knows its own beam theory (see [`mesh/README.md`](../mesh/README.md) for the `BeamFormulation` registry that resolves it) and dispatches its own `stiffness_element()`; this builder never branches on theory itself.

```python
builder = StiffnessMatrixBuilder(mesh, elements, frame=True)
builder.add_nodal_stiffness(x=120.0, K_local=K_bearing, dof_slots=[1, 2])
K = builder.build()
```

| Member | Purpose |
|---|---|
| `__init__(mesh, elements, frame)` | `frame` is required, no default. `frame=True` scatters a combined 6×6 block per element (4×4 bending + 2×2 axial, local order `[u_a, v_a, th_a, u_b, v_b, th_b]`); `frame=False` scatters only the 4×4 bending block and leaves every axial DOF at zero — unconstrained unless the caller removes it separately, which this builder does not check. |
| `add_nodal_stiffness(x, K_local, dof_slots)` | Adds an arbitrary local stiffness block at the node nearest `x`, at the named DOF slots (`0`=axial, `1`=transverse, `2`=rotation). Generic — it does not know what a bearing is; the caller supplies the numbers. |
| `build(*, kGA_override=None)` | The assembled global matrix; `kGA_override` is forwarded only to elements whose formulation uses it. |

### The submodel solve

`constraints/submodel_extraction.py` (`extract_submodel_values`) restricts the solution to a subdomain, injecting the global solution at the cut nodes.

> **Grown substantially since the previous pass of this document.** `fem_solvers/submodel_solver/` is no longer just `extract_submodel_values()` feeding a reduced re-solve — it is now a dedicated package: `lagrange_multipliers.py` defines `SubmodelSolver`/`SubmodelSolution` (a Lagrange-multiplier formulation for enforcing the extracted boundary values on the cut submodel), and `submodel_postprocessing.py` mirrors the global-solver postprocessing pattern (`postprocessor_for_submodel`, `build_submodel_result`, one `Submodel{EulerBernoulli,Timoshenko}PostProcessing` class per theory) to produce a `SubmodelResult` (see [`results/README.md`](../results/README.md)). This is a real, and non-trivial, addition to the solver's numerical method — the previous submodel treatment did not use Lagrange multipliers. **Not documented in narrative detail here** — this section states what exists and where, not yet how the formulation works; read `lagrange_multipliers.py` directly before relying on its contract for a new submodel study.

---

## Torsion

### `TorsionSolver`

`shaft/fem_solvers/global_solver/torsion.py` (moved from `static_solvers/torsion.py` — grouped with `rigid_support.py` and `global_postprocessing.py` as the three modules producing one shaft's global result, rather than filed under "static" post-processing). Pure statics over the same `x_nodes` the bending/axial solve used — no DOF, no stiffness matrix, no dependency on beam theory. `T(x)` is the cumulative sum of `TorqueLoad` up to `x`; `tau(x) = T(x) / Wt(x)` via `Shaft.Wt_at()`; `phi(x)` is the twist angle, integrated node-to-node from `d(phi)/dx = T(x) / (G·J(x))`, referenced to `phi = 0` at the first node, using `Shaft.J_at()`.

```python
T_total, tau_total, phi_total, contributions = TorsionSolver().solve(shaft_system, x_nodes)
```

| Member | Purpose |
|---|---|
| `solve` | Returns `(T_total, tau_total, phi_total, contributions)`, each a length-`n` array aligned with `x_nodes`. |
| `validate_equilibrium` | Torsion residual as a list of error strings. |

This is a class, not a pair of module-level functions, and it is **not called from `RigidSupportFEMSolver.solve()`** — it is invoked independently by whoever assembles a full result set (`build_shaft_result()`, see below).

---

## Shaft results and post-processing

### `build_shaft_result()`

`fem_solvers/global_solver/global_postprocessing.py`. **Replaces the `ShaftResultsReader` class entirely** — the previous pass of this document described a `ShaftResultsReader.read()` method; that class no longer exists. Post-processing a solved `RigidSupportFEMSolver` (plus a separately-run `TorsionSolver`) into a `ShaftResults` is now done through module-level functions:

| Member | Purpose |
|---|---|
| `build_shaft_result(...)` | Assembles the populated `ShaftResults` — see [`results/README.md`](../results/README.md) for the container's full field list. |
| `postprocessor_for_global(...)` | Resolves the correct per-theory post-processor (`GlobalEulerBernoulliPostProcessing` / `GlobalTimoshenkoPostProcessing`), the same registry-dispatch shape used by `mesh`'s `_FORMULATION_REGISTRY` and `core`'s bearing family registry, rather than a reader class branching internally. |
| `GlobalEulerBernoulliPostProcessing`, `GlobalTimoshenkoPostProcessing` | Internal-effort (`M`, `V`) recovery per beam theory — the class pair that the previous pass of this document called `EulerBernoulliPostProcessing`/`TimoshenkoPostProcessing` under `element_theories/`; those two names are still defined in `element_theories/element_postprocessing.py` as well (`ElementTheoryPostProcessor` base), and are distinct from these `Global*` classes. Confirm which pair a given caller actually uses before assuming they are interchangeable — **not yet reconciled in this document.** |

**Not re-verified in this pass:** the exact call signature of `build_shaft_result(...)`, and whether it still dispatches bending/shear recovery by reading `beam_theory` off `solver.elements[0]` the way the previous `ShaftResultsReader._recover_internal_forces()` did. Read `global_postprocessing.py` directly before depending on either.

### Stress concentration

`ShaftPostProcessor` (in `static_solvers/postprocessing.py`) enriches a `ShaftResults` with stress concentration factors at shoulders and keyways, producing corrected stress arrays for the fatigue and failure solvers. It performs no FEM and no equilibrium — it consumes an already-solved result.

| Class | Purpose |
|---|---|
| `StressConcentration` | One feature: position `x`, kind (`shoulder`, `keyway`, `groove`, `press_fit`), `Kt_bending`/`Kt_torsion`, `Kf_bending`/`Kf_torsion`, `q_bending`/`q_torsion`, the geometry `r`, `D`, `d` that produced them, and a free-text `note`. |
| `PostProcessedResults` | The original `ShaftResults` by composition, plus the fatigue factor arrays, the corrected bending and shear stress arrays, the feature list, and the located maxima. |
| `ShaftPostProcessor` | `process()` walks the shaft's features and returns a `PostProcessedResults`. |

Shoulder factors come from the Peterson curve fits in Shigley's tables; notch sensitivity interpolates the Neuber constant against ultimate strength, referenced to the shear yield for torsion; the ultimate strength at each position is read from the material database, with a conservative fallback.

The stresses in `ShaftResults` are **nominal**. Notch effects live only in `PostProcessedResults`. A governing location that coincides with a shoulder or keyway is the signal to run the post-processor before drawing a conclusion.

### `shaft/utils.py`

Stateless helpers, reusable by the fatigue solvers: Marin factors (surface finish, size, load type, temperature, reliability) and notch factors (theoretical shoulder `Kt` in bending and torsion, the Neuber material constant, notch sensitivity, and the fatigue factor from `Kt` and `q`).

---

## Bearings — ISO/TS 16281

One package, `bearings/load_distribution/iso_16281/`, covers point contact and line contact, single-row and multi-row alike. There is no longer a row-count split at the package level: `contact_solver.py` defines `ISO16281BallSolver`, `ISO16281RollerSolver` (single-row) and `ISO16281MultiRowBallSolverSharedDisplacement`, `ISO16281MultiRowRollerSolverSharedDisplacement` (multi-row) side by side, and every one of them returns the same shape.

### Dispatch

A bearing is matched to a solver by the capabilities its family declares and the geometry attributes it actually carries. Each contact solver registers itself at import time.

| Name | Purpose |
|---|---|
| `resolve_solver_cls` | The solver whose contract an assembled bearing satisfies. Ambiguity resolves to the most specific match. |
| `resolve_solver_cls_for_attrs` | The same match from a bare attribute set — how the rows of a multi-row bearing, which are plain dicts, are resolved. |
| `register_contact_solver` | The registration decorator, applied by each solver class. |
| `SolverDispatchError` | Raised, naming the offending bearing label, when nothing matches. |

### Ball bearings — point contact

**`ISO16281BallSolver`** — one class covers deep groove, angular contact, and each row of a multi-row thrust ball bearing, since the nominal contact angle is read off the bearing. Prescribed-tilt formulation: per raceway, a two-unknown root solve for radial and axial approach in the plane of the resultant radial force.

> **Open item.** The thrust-vs-radial axial/radial treatment note (calculations for a thrust bearing must use the axial approach, not the radial one) is written into the module docstrings of `contact_postprocessing.py`/`contact_solver.py` and into the multi-row combination path, but the equivalent explicit note for the **single-row thrust ball** case (`bearing.duty == "thrust"` handled by `ISO16281BallSolver` itself) has not yet been added at that call site — flagged here rather than silently assumed correct.

### `ISO16281MultiRowBallSolverSharedDisplacement`

Multi-row is a genuinely different solve, not a mode of the single-row one: axially co-located rows share one rigid ring, so the ring displacement(s) are found by a *single* root-find and every row's reaction is read off that one converged state — there is no per-row local displacement and no outer fixed-point loop over rows. `solve()` builds each row's `LoadDistributionResult`, then wraps them with `BallBearingResult.multirow(rows, **shared)`.

### Roller bearings — line contact

**`ISO16281RollerSolver`** — the lamina-model solver for radial cylindrical roller bearings: zero nominal contact angle, no axial capacity (`Fa=0.0` is always passed to the result, never read off the FEM node), each roller sliced into at least thirty laminae and corrected for its logarithmic profile.

### `ISO16281MultiRowRollerSolverSharedDisplacement`

Same shared-displacement treatment as the ball case. No multi-row roller family exists in `core/` yet, so this path has not been exercised on a real case.

### What `solve()` returns

Every entry point returns `dict[label, BearingAnalysisResult]` — never a bare `BearingResult`. `BearingAnalysisResult.load_distribution` is always populated; `BearingAnalysisResult.basic_life` is populated when `solve()` is called with `postprocess=True` and is `None` otherwise. See [`results/README.md`](../results/README.md#bearingsbearing_analysis_resultpy) for the container itself.

**Post-processing (`_attach_postprocessing`)**, per solver class, computes bearing stiffness via `contact_postprocessing.bearing_stiffness()` and attaches it with `result.with_stiffness(stiffness)` (never mutation — results are frozen), and builds the `Ball/RollerBasicReferenceRatingLife` rows plus the bearing's `Pref` object to assemble a `BasicReferenceRatingLifeResult`.

> **Open items, flagged in the source.**
> - Bearing stiffness can evaluate to `float('inf')` when the projected displacement is ~0 — acknowledged incorrect, deliberately not yet fixed.
> - An unloaded row in `ball_row_reference_rating_life`/`roller_row_reference_rating_life` should have infinite life and drop out of the combination; today it raises `ValueError` instead.
> - `n_iter` on `BearingResult` is, in the current implementation, actually `nfev` (root-solver function evaluations), not an iteration count — the name may need to change.

---

## Gears

**`GearSolver`** — cylindrical gear geometry and mesh forces, following ISO 21771 and the MAAG conventions. All methods are pure functions with no internal state. `SpurHelicalGears/LoadCapacity_solver/load_capacity.py` remains reserved.

---

## Mesh convergence

`solvers/mesh/convergence_solver.py`. Grid Convergence Index by Richardson extrapolation on the resultant transverse displacement, across three or more refinement levels.

| Class | Purpose |
|---|---|
| `RichardsonGCI` | For one triplet of coarse, medium and fine solutions: the refinement ratios, the observed order of convergence, the relative errors, the extrapolated value, the two convergence indices and their pass flags. |
| `ConvergenceRecord` | The refinement history for one interval. |
| `MeshRefinementResult` | One record per interval, plus `all_extra_nodes`. |
| `MeshConvergenceStudy` | The orchestrator (`run`, `intervals_from_shaft_system`). |

---

## Design contracts

- **A solver computes; it does not collect.** No solver owns a registry keyed by shaft or bearing label. That is the study layer's job.
- **No solver imports another solver.** Ball and roller, single-row and multi-row: each is a vertical slice within `contact_solver.py`. They meet at the dispatcher and at the result containers, never at each other's internals.
- **No hidden state.** Every intermediate quantity is a public attribute. A solver exposes its full working for inspection.
- **One instance per subject.** A solver's published attributes describe the last thing it solved. Re-solving overwrites them.
- **Beam theory is a construction-time input, never inferred.** `RigidSupportFEMSolver` takes a `BeamModelSettings`; nothing downstream guesses which theory produced a result except by reading `beam_theory` explicitly off the elements.
- **Capacity belongs to the element.** A solver never carries its own copy of a standard's capacity formula; it calls the family through the bearing.
- **A result belongs in `results/`, never in the solver package.** No solver module defines its own `results.py` any more — the earlier split where the multi-row bearing branch kept a local, structurally-different result under the same class names as `results/` has been retired.
- **A failed solve is still a result.** Convergence metadata (`n_iter`, `residual`, `ok`) is stored, never raised away. Checking it is the consumer's responsibility.
- **Flag, do not silently fix.** A suspected discrepancy is documented in place — the bearing-stiffness `inf` case, the unloaded-row life gap, and the `n_iter`/`nfev` naming question are stated here rather than silently guessed at or patched over.

---

## Extending this package

**Adding a bearing family.** Nothing in this package changes. Declare the family's `CAPABILITIES` and `REQUIRED_FOR` in `core/`; dispatch resolves it from those declarations.

**Adding a beam theory.** Add the formulation to `mesh/shaft/element_type/elem.py`'s registry (see [`mesh/README.md`](../mesh/README.md#extending-this-package)) — this package's solvers do not need to change, since `StiffnessMatrixBuilder` and the postprocessing dispatch are already theory-agnostic at the assembly level.

**Adding a solver.** Place it by the analysis it performs, mirroring the tree under `results/`. Give it its own result container in `results/` — never a local `results.py` in the solver package. Carry `n_iter`, `residual` and `ok` if it iterates. Import no other solver.

**Adding post-processing.** It belongs beside the solver whose output it consumes, not in a shared module: the exponents and contact laws differ per branch, and a shared implementation would need a branch on contact type inside it.

---

## Standards referenced

| Standard | Applies to |
|---|---|
| ISO/TS 16281 | Rolling bearing internal load distribution — §4 point contact, §5 line contact, §5.3.4 per-lamina equivalent load, basic reference rating life |
| ISO 281 | Dynamic load ratings, basic rating life, multi-row reduction factors |
| ISO 1281-1 | Explanatory notes on ISO 281 |
| ISO 21771 | Cylindrical gear geometry and mesh force resolution |
| ISO 53 | Standard basic rack tooth profile |
| ISO 6336-1 | Application and dynamic factors — data layer present, solver reserved |
