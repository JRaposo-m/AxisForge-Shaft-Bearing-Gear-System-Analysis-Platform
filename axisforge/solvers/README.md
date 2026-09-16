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
- [Bearings — ISO/TS 16281, single row](#bearings--isots-16281-single-row)
- [Bearings — ISO/TS 16281, multi-row thrust](#bearings--isots-16281-multi-row-thrust)
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
| `shaft/fem_solvers/` | `rigid_support.py` | `RigidSupportFEMSolver` | Implemented — **renamed from `RigidBearingFEMSolver`/`rigid_bearing.py`**, now constructed from a `BeamModelSettings` (see [`mesh/README.md`](../mesh/README.md#beam-model-settings)) instead of a bare theory string |
| `shaft/fem_solvers/assembly/` | `build_stiffness_matrix.py` | `StiffnessMatrixBuilder` | Implemented — moved under `assembly/`; theory-agnostic, `frame` is now a required constructor argument (see [Shaft FEM](#shaft-fem)) |
| `shaft/fem_solvers/assembly/load_assembly/` | `point_loads.py`, `distributed_loads.py`, `vector_external_forces.py` | Point-load and distributed-load vector assembly | Present — `assemble_point_load_vector()` and `extract_submodel_values()`'s call shape are carried forward **unverified against the new API** by `rigid_support.py`'s own docstring; confirm before relying on them |
| `shaft/fem_solvers/assembly/numerics/` | `gauss_quadrature.py`, `numerical_guards.py` | `QuadratureOrderEstimator`, `check_conditioning` | Implemented — `check_conditioning` confirmed: takes only `K_red`, fixed 1e14 condition-number threshold, no tolerance argument |
| `shaft/fem_solvers/constraints/` | `boundary_conditions.py`, `submodel_extraction.py` | `boundary_dofs`, `extract_submodel_values` | Present — submodel extraction not re-verified against the new module layout (see above) |
| `shaft/static_solvers/` | `torsion.py` | `TorsionSolver` | Implemented, but **see the open item under [Torsion](#torsion)** — depends on a `Shaft.J_at()` accessor not confirmed to exist |
| `shaft/static_solvers/` | `results_reader.py` | `ShaftResultsReader` | Implemented — moved from `shaft/static/`, now also assembles `u`, `theta_xz`, `theta_xy`, `phi`/`phi_total`/`phi_max` into `ShaftResults` (see [`results/README.md`](../results/README.md)) |
| `shaft/fem_solvers/element_theories/timoshenko/` | `postprocessing.py` | `TimoshenkoPostProcessing` | Implemented |
| `shaft/fem_solvers/element_theories/euler_bernoulli/` | `postprocessing.py` | `EulerBernoulliPostProcessing` | Implemented |
| `shaft/static_solvers/` | `postprocessing.py` | `ShaftPostProcessor`, `PostProcessedResults`, `StressConcentration` | Implemented — moved from `shaft/static/` |
| `shaft/static_solvers/` | `static_failure.py` | static failure criteria | Reserved — module exists, defines nothing (moved from `shaft/static/`, still empty) |
| `bearings/.../single_row/iso_16281/` | `dispatch.py`, `numerics.py`, `validation.py` | Solver resolution, root-solve wrapper, guards | Implemented |
| `bearings/.../single_row/iso_16281/ball_bearing/` | `solver.py`, `postprocessing.py` | `ISO16281BallSolver` and its post-processing | Implemented |
| `bearings/.../single_row/iso_16281/roller_bearing/` | `solver.py`, `postprocessing.py` | `ISO16281RollerSolver` and its post-processing | Implemented |
| `bearings/.../multi_row/thrust_bearings/iso_16281/ball_bearings/` | `multirow_solver.py`, `postprocessing.py`, `results.py` | `ISO16281MultiRowBallSolverSharedDisplacement` | Implemented |
| `bearings/.../multi_row/thrust_bearings/iso_16281/roller_bearings/` | `multirow_solver.py`, `postprocessing.py`, `results.py` | `ISO16281MultiRowRollerSolverSharedDisplacement` | Implemented, not exercised — no multi-row roller family exists in `core/` |
| `gears/` | `geometry.py`, `utils.py` | `GearSolver` and stateless helpers | Implemented |
| `gears/SpurHelicalGears/LoadCapacity_solver/` | `load_capacity.py` | ISO 6336 load capacity | Reserved — module exists, defines nothing |
| `mesh/` | `convergence_solver.py` | `MeshConvergenceStudy`, `RichardsonGCI`, `MeshRefinementResult`, `ConvergenceRecord` | Implemented — **renamed from `mesh_convergence_study.py`**, same classes |

> **Open item — the old solver name is still called from `fixtures/`.** `fixtures/studies/shafts/convergence_studies/convergence_study.py` still does `from axisforge.solvers.machine_elements.shaft.fem_solvers.rigid_support import RigidBearingFEMSolver` and constructs it as `RigidBearingFEMSolver(theory=theory, distribute_gear_labels=distribute_gear_labels)`. `rigid_support.py` today defines `RigidSupportFEMSolver`, taking a `BeamModelSettings` instance, not a bare `theory` string — so as written, that import fails, and the mesh-convergence fixture path (`run_convergence`) cannot run until it is updated to match the renamed class and constructor. `fixtures/studies/shafts/fem_studies/fem_simple.py` (the ordinary resolution path) already uses the current name and signature correctly.

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

Run-level registries are not here. `RigidSupportFEMResultsLibrary`, `BearingResultsLibrary` and `RollingBearingSolver` all live in `fixtures/studies/`: collecting results across a whole gearbox is orchestration, and orchestration is a study concern.

> **Open item.** The multi-row branch keeps its own `results.py` in each of `ball_bearings/` and `roller_bearings/`, redefining `BallBearingResult`, `RollerBearingResult` and their per-row types under the same names as the copies in `results/`, with a different predicate (`is_multirow` here, `is_single` there). These belong in `results/bearings/load_distribution/multi_row/`. Until they move, a consumer that can receive a result from either branch cannot rely on either predicate.

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
│   │   │   └── constraints/
│   │   │       ├── boundary_conditions.py              boundary_dofs
│   │   │       └── submodel_extraction.py              extract_submodel_values
│   │   ├── fem_solvers/element_theories/
│   │   │   ├── timoshenko/postprocessing.py            TimoshenkoPostProcessing
│   │   │   └── euler_bernoulli/postprocessing.py       EulerBernoulliPostProcessing
│   │   └── static_solvers/
│   │       ├── torsion.py                              TorsionSolver — separate from the bending/axial solve
│   │       ├── results_reader.py                       ShaftResultsReader
│   │       ├── postprocessing.py                       ShaftPostProcessor, StressConcentration
│   │       └── static_failure.py                       Reserved
│   ├── bearings/load_distribution/
│   │   ├── single_row/iso_16281/
│   │   │   ├── dispatch.py                   Capability-based solver resolution
│   │   │   ├── numerics.py                   Root-solve wrapper
│   │   │   ├── validation.py                 Readiness and arrangement guards
│   │   │   ├── ball_bearing/{solver,postprocessing}.py
│   │   │   └── roller_bearing/{solver,postprocessing}.py
│   │   └── multi_row/thrust_bearings/iso_16281/
│   │       ├── numerics.py  validation.py
│   │       ├── ball_bearings/{multirow_solver,postprocessing,results}.py
│   │       └── roller_bearings/{multirow_solver,postprocessing,results}.py
│   └── gears/
│       ├── geometry.py                       GearSolver
│       ├── utils.py                          Involute helpers, rack constants
│       └── SpurHelicalGears/LoadCapacity_solver/load_capacity.py    Reserved
└── mesh/
    └── convergence_solver.py                 MeshConvergenceStudy, RichardsonGCI (renamed from mesh_convergence_study.py)
```

The bearing tree is split by **row count first**, then by standard, then by contact type. The shaft tree is now split by **what it produces**: `fem_solvers/` for the bending + axial FEM solve and its assembly internals, `static_solvers/` for everything that post-processes a solved state without its own DOFs (torsion, stress recovery, the reserved static-failure module).

---

## Import surface

| Import from | Names |
|---|---|
| `...shaft.fem_solvers.rigid_support` | `RigidSupportFEMSolver` |
| `...shaft.fem_solvers.assembly.build_stiffness_matrix` | `StiffnessMatrixBuilder` |
| `...shaft.fem_solvers.assembly.load_assembly.point_loads` | `assemble_point_load_vector` |
| `...shaft.fem_solvers.assembly.load_assembly.distributed_loads` | `assemble_distributed_load_vector` |
| `...shaft.fem_solvers.assembly.numerics.numerical_guards` | `check_conditioning` |
| `...shaft.fem_solvers.constraints.boundary_conditions` | `boundary_dofs` |
| `...shaft.fem_solvers.constraints.submodel_extraction` | `extract_submodel_values` |
| `...shaft.static_solvers.torsion` | `TorsionSolver` |
| `...shaft.static_solvers.results_reader` | `ShaftResultsReader` |
| `...shaft.static_solvers.postprocessing` | `ShaftPostProcessor`, `PostProcessedResults`, `StressConcentration` |
| `...bearings.load_distribution.single_row.iso_16281.dispatch` | `resolve_solver_cls`, `resolve_solver_cls_for_attrs`, `register_contact_solver`, `SolverDispatchError` |
| `...single_row.iso_16281.ball_bearing.solver` | `ISO16281BallSolver` |
| `...single_row.iso_16281.roller_bearing.solver` | `ISO16281RollerSolver` |
| `...multi_row.thrust_bearings.iso_16281.ball_bearings.multirow_solver` | `ISO16281MultiRowBallSolverSharedDisplacement` |
| `...multi_row.thrust_bearings.iso_16281.roller_bearings.multirow_solver` | `ISO16281MultiRowRollerSolverSharedDisplacement` |
| `...gears.geometry` | `GearSolver` |
| `axisforge.solvers.mesh.convergence_solver` | `MeshConvergenceStudy`, `RichardsonGCI`, `MeshRefinementResult`, `ConvergenceRecord` |

Ball and roller packages are imported explicitly and never flattened into one namespace: point and line contact produce different result types, and flattening would hide which contact model a name belongs to.

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

`fem_solvers/rigid_support.py`. Models each bearing location as a rigid point **support** boundary condition — it knows nothing about rolling-bearing physics (ISO 281 / ISO/TS 16281, handled elsewhere). "Rigid" is this solver's identity, not one of several modes; a compliant-bearing solver would be a sibling module.

```python
settings = BeamModelSettings(beam_theory="timoshenko", shear_theory="cowper", integration_method="exact")
solver = RigidSupportFEMSolver(settings)
solver.solve(shaft_system)
solver.d_total_xz      # global displacement (XZ)
solver.f_xz_total       # global force, including reactions
```

Orchestration only. `solve()` calls, in order: `Mesh1D` → `Elem.from_mesh()` (driven by the `BeamModelSettings` passed at construction) → `StiffnessMatrixBuilder` (`frame=True`, fixed — see below) → `boundary_dofs` → the point- and distributed-load assemblers → a linear solve (still beam-theory-agnostic) → publishes public attributes.

| Published attribute | Contents |
|---|---|
| `x_nodes`, `elements` | The mesh actually solved. |
| `free_dofs`, `constrained_dofs` | Boundary condition partition. |
| `d_total_xz`, `d_total_xy` | Superposed displacement vectors per plane. |
| `f_xz_ext`, `f_xy_ext` | External force vectors per plane. |
| `f_xz_total`, `f_xy_total` | Total nodal force vectors, including reactions. |

Each attribute is `None` until `solve()` has run at least once, and re-calling `solve()` overwrites all of them in place — build a fresh instance if isolation between runs is needed (same "one solver instance per shaft" rule as before).

**Why `frame=True` is fixed, not a caller choice.** This solver accepts `AxialLoad` and injects it into the global force vector, so the axial DOFs must be wired into `K` for the system to be solvable at all; `frame=False` would leave every axial DOF disconnected. Since there is only one correct choice for what this solver does, it is a fixed identity property (`_FRAME = True`), not a constructor argument.

**Torsion is not produced here.** `T` and `tau` come from `TorsionSolver`, called separately — see [Torsion](#torsion).

### `StiffnessMatrixBuilder`

`fem_solvers/assembly/build_stiffness_matrix.py`. Assembles the global stiffness matrix from element contributions, plus optional nodal stiffness injected at chosen positions (bearings, or any other spring). Fully theory-agnostic — each `Elem` already knows its own `beam_theory` and dispatches its own `stiffness_element()`; this builder never branches on theory itself.

```python
builder = StiffnessMatrixBuilder(mesh, elements, frame=True)
builder.add_nodal_stiffness(x=120.0, K_local=K_bearing, dof_slots=[1, 2])
K = builder.build()
```

| Member | Purpose |
|---|---|
| `__init__(mesh, elements, frame)` | `frame` is required, no default. `frame=True` scatters a combined 6×6 block per element (4×4 bending + 2×2 axial, local order `[u_a, v_a, th_a, u_b, v_b, th_b]`); `frame=False` scatters only the 4×4 bending block and leaves every axial DOF at zero — unconstrained unless the caller removes it separately, which this builder does not check. |
| `add_nodal_stiffness(x, K_local, dof_slots)` | Adds an arbitrary local stiffness block at the node nearest `x`, at the named DOF slots (`0`=axial, `1`=transverse, `2`=rotation). Generic — it does not know what a bearing is; the caller supplies the numbers. |
| `build(*, kGA_override=None)` | The assembled global matrix; `kGA_override` is forwarded only to Timoshenko elements, silently ignored for Euler-Bernoulli ones. |

### The submodel solve

`constraints/submodel_extraction.py` (`extract_submodel_values`) restricts the solution to a subdomain, injecting the global solution at the cut nodes. **Not re-verified against the current module layout** — `rigid_support.py`'s own docstring flags this call as carried forward unchanged in shape from the previous skeleton, without the source having been checked against the new `BeamModelSettings`/`frame` API. Confirm its call contract before relying on it for a new submodel study.

---

## Torsion

### `TorsionSolver`

`shaft/static_solvers/torsion.py`. Pure statics over the same `x_nodes` the bending/axial solve used — no DOF, no stiffness matrix, no dependency on beam theory. `T(x)` is the cumulative sum of `TorqueLoad` up to `x`; `tau(x) = T(x) / Wt(x)` via `Shaft.Wt_at()`; `phi(x)` is the twist angle, integrated node-to-node from `d(phi)/dx = T(x) / (G·J(x))`, referenced to `phi = 0` at the first node.

```python
T_total, tau_total, phi_total, contributions = TorsionSolver().solve(shaft_system, x_nodes)
```

| Member | Purpose |
|---|---|
| `solve` | Returns `(T_total, tau_total, phi_total, contributions)`, each a length-`n` array aligned with `x_nodes`. |
| `validate_equilibrium` | Torsion residual as a list of error strings — renamed from the previous module-level `validate_torsion_equilibrium()`. |

This is a **class now**, not the previous module-level `solve_torsion()`/`validate_torsion_equilibrium()` functions, and it is **not called from `RigidSupportFEMSolver.solve()`** — it is invoked independently by whoever assembles a full result set (`ShaftResultsReader.read()`).

> **Open item — `Shaft.J_at()` is not confirmed to exist.** `_twist_angle()` calls `shaft_system.shaft.J_at(x)`, a polar-second-moment-of-area accessor alongside `diameter_at()`/`W_at()`/`Wt_at()`. This is flagged directly in the module's own docstring rather than assumed: if `Wt_at()` is already `J / r_outer` for both solid and hollow sections, `J_at(x)` could be `Wt_at(x) * diameter_at(x) / 2`, but that relationship has not been confirmed against `Shaft`'s actual source. Until it is, `TorsionSolver.solve()` may raise `AttributeError` on `J_at`.

---

## Shaft results and post-processing

### `ShaftResultsReader`

`static_solvers/results_reader.py`. Post-processes a solved `RigidSupportFEMSolver` (plus a separately-run `TorsionSolver`) into a `ShaftResults`. It recovers internal forces, deflections, section properties, bearing reactions and per-bearing node data in one pass.

| Member | Purpose |
|---|---|
| `read()` | Returns a populated `ShaftResults` — see [`results/README.md`](../results/README.md) for the full field list, including the fields new since the last pass (`u`, `theta_xz`, `theta_xy`, `phi`/`phi_total`/`phi_max`/`x_phi_max`). |

**Bending/shear recovery now dispatches by theory.** `_recover_internal_forces()` reads `beam_theory` off `solver.elements[0]` (every `Elem` built for one shaft shares the same theory, since it comes from one `BeamModelSettings`) and delegates to `TimoshenkoPostProcessing` or `EulerBernoulliPostProcessing` accordingly. This dispatch **is wired** in `results_reader.py` today — the `TODO(owner)` comments still present inside both `element_theories/timoshenko/postprocessing.py` and `element_theories/euler_bernoulli/postprocessing.py`, which describe this dispatch as not yet decided, are stale relative to `results_reader.py`'s own current code.

### `TimoshenkoPostProcessing` / `EulerBernoulliPostProcessing`

`fem_solvers/element_theories/{timoshenko,euler_bernoulli}/postprocessing.py`. Internal-effort (`M`, `V`) recovery, split out of `ShaftResultsReader` — one class per beam theory, same method shape on both (`recover_element_displacements`, `bending_moment`, `shear_force`, `recover_internal_forces`, `_sweep_plane`).

| Class | `M(x)` | `V(x)` |
|---|---|---|
| `EulerBernoulliPostProcessing` | `M = E·I · B_b(zeta) @ a^(e)` | Constant within an element (third derivative of a cubic shape function) |
| `TimoshenkoPostProcessing` | Same form, `B_b` dispatched via `elem.bending_strain_matrix()` | `Q = D_s · B_s(zeta) @ a^(e)`, genuinely a function of `x` when `integration_method="exact"` |

Both read `a^(e)` — the element's local 4-DOF nodal displacement vector `[v_a, th_a, v_b, th_b]` — straight from the plane's global displacement vector; axial is decoupled and not part of `a^(e)` here.

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

## Bearings — ISO/TS 16281, single row

*(unchanged from the previous pass — no evidence found of edits to this branch)*

### Dispatch

A bearing is matched to a solver by the capabilities its family declares and the geometry attributes it actually carries. Each contact solver registers itself at import time.

| Name | Purpose |
|---|---|
| `resolve_solver_cls` | The single-row solver whose contract an assembled bearing satisfies. Ambiguity resolves to the most specific match. |
| `resolve_solver_cls_for_attrs` | The same match from a bare attribute set — how the rows of a multi-row bearing, which are plain dicts, are resolved. |
| `register_contact_solver` | The registration decorator, applied by the two single-row solvers. |
| `SolverDispatchError` | Raised, naming the offending bearing label, when nothing matches. |

### Ball bearings — point contact

**`ISO16281BallSolver`** — one class covers deep groove, angular contact, and each row of a multi-row thrust ball bearing, since the nominal contact angle is read off the bearing. Prescribed-tilt formulation: per raceway, a two-unknown root solve for radial and axial approach in the plane of the resultant radial force.

### Roller bearings — line contact

**`ISO16281RollerSolver`** — the lamina-model solver for radial cylindrical roller bearings: zero nominal contact angle, no axial capacity, each roller sliced into at least thirty laminae and corrected for its logarithmic profile.

---

## Bearings — ISO/TS 16281, multi-row thrust

*(unchanged from the previous pass)*

Multi-row is a genuinely different solve, not a mode of the single-row one: axially stacked rows share one rigid ring, so the ring displacement is a single unknown and each row's reaction is summed against it. `ISO16281MultiRowBallSolverSharedDisplacement` and its roller equivalent implement this; the roller side has no multi-row roller family in `core/` to run against, so it has not been exercised on a real case.

---

## Gears

*(unchanged from the previous pass)*

**`GearSolver`** — cylindrical gear geometry and mesh forces, following ISO 21771 and the MAAG conventions. All methods are pure functions with no internal state. `SpurHelicalGears/LoadCapacity_solver/load_capacity.py` remains reserved.

---

## Mesh convergence

`solvers/mesh/convergence_solver.py` — **renamed from `mesh_convergence_study.py`**, same classes. Grid Convergence Index by Richardson extrapolation on the resultant transverse displacement, across three or more refinement levels.

| Class | Purpose |
|---|---|
| `RichardsonGCI` | For one triplet of coarse, medium and fine solutions: the refinement ratios, the observed order of convergence, the relative errors, the extrapolated value, the two convergence indices and their pass flags. |
| `ConvergenceRecord` | The refinement history for one interval. |
| `MeshRefinementResult` | One record per interval, plus `all_extra_nodes`. Its `print_report()` method was removed on its move into `results/` — see the `fixtures/studies/shafts/convergence_studies/` note under [`fixtures/README.md`](../fixtures/README.md), which has not been rebuilt yet. |
| `MeshConvergenceStudy` | The orchestrator (`run`, `intervals_from_shaft_system`). |

This solver-level module is now driven, at the fixture layer, by `fixtures/studies/shafts/convergence_studies/convergence_study.py` (`run_convergence`) — but see the open item under [Status](#status): that fixture still calls the pre-rename solver API and does not currently import successfully.

---

## Design contracts

- **A solver computes; it does not collect.** No solver owns a registry keyed by shaft or bearing label. That is the study layer's job.
- **No solver imports another solver.** Ball and roller, single-row and multi-row, shaft and bearing: each is a vertical slice. They meet at the dispatcher and at the result containers.
- **No hidden state.** Every intermediate quantity is a public attribute. A solver exposes its full working for inspection.
- **One instance per subject.** A solver's published attributes describe the last thing it solved. Re-solving overwrites them.
- **Beam theory is a construction-time input, never inferred.** `RigidSupportFEMSolver` takes a `BeamModelSettings`; nothing downstream guesses which theory produced a result except by reading `beam_theory` explicitly off the elements.
- **Capacity belongs to the element.** A solver never carries its own copy of a standard's capacity formula; it calls the family through the bearing.
- **A failed solve is still a result.** Convergence metadata (`n_iter`, `residual`, `ok`) is stored, never raised away. Checking it is the consumer's responsibility.
- **Flag, do not silently fix.** A suspected discrepancy is documented in place — the `Shaft.J_at()` gap in `TorsionSolver` and the pre-rename call in `convergence_study.py` are stated here rather than silently guessed at or patched over.

---

## Extending this package

**Adding a bearing family.** Nothing in this package changes. Declare the family's `CAPABILITIES` and `REQUIRED_FOR` in `core/`; dispatch resolves it from those declarations.

**Adding a beam theory.** Add the element module under `mesh/shaft/element_type/` (see [`mesh/README.md`](../mesh/README.md#extending-this-package)) and extend `Elem`'s own dispatch — this package's solvers do not need to change, since `StiffnessMatrixBuilder` and the postprocessing split are already theory-agnostic at the assembly level and theory-specific only in `element_theories/`.

**Adding a solver.** Place it by the analysis it performs, mirroring the tree under `results/`. Give it its own result container in `results/` — never a local `results.py` in the solver package. Carry `n_iter`, `residual` and `ok` if it iterates. Import no other solver.

**Adding post-processing.** It belongs beside the solver whose output it consumes, not in a shared module: the exponents and contact laws differ per branch, and a shared implementation would need a branch on contact type inside it.

---

## Standards referenced

| Standard | Applies to |
|---|---|
| ISO/TS 16281 | Rolling bearing internal load distribution — §4 point contact, §5 line contact, §5.3.4 per-lamina equivalent load |
| ISO 281 | Dynamic load ratings, basic rating life, multi-row reduction factors |
| ISO 1281-1 | Explanatory notes on ISO 281 |
| ISO 21771 | Cylindrical gear geometry and mesh force resolution |
| ISO 53 | Standard basic rack tooth profile |
| ISO 6336-1 | Application and dynamic factors — data layer present, solver reserved |