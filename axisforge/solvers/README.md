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
| `shaft/fem_solvers/` | `build_stiffness_matrix.py` | `StiffnessMatrixBuilder` | Implemented |
| `shaft/fem_solvers/` | `rigid_bearing.py` | `RigidBearingFEMSolver` | Implemented |
| `shaft/fem_solvers/sub_models/` | `lagrange_multipliers.py` | `SubmodelResult` and the submodel solve | Implemented |
| `shaft/static/` | `results_reader.py` | `ShaftResultsReader` | Implemented |
| `shaft/static/` | `postprocessing.py` | `ShaftPostProcessor`, `PostProcessedResults`, `StressConcentration` | Implemented |
| `shaft/static/` | `static_failure.py` | static failure criteria | Reserved — module exists, defines nothing |
| `shaft/` | `utils.py` | Marin factors, Peterson Kt, Neuber q, Kf | Implemented |
| `bearings/.../single_row/iso_16281/` | `dispatch.py`, `numerics.py`, `validation.py` | Solver resolution, root-solve wrapper, guards | Implemented |
| `bearings/.../single_row/iso_16281/ball_bearing/` | `solver.py`, `postprocessing.py` | `ISO16281BallSolver` and its post-processing | Implemented |
| `bearings/.../single_row/iso_16281/roller_bearing/` | `solver.py`, `postprocessing.py` | `ISO16281RollerSolver` and its post-processing | Implemented |
| `bearings/.../multi_row/thrust_bearings/iso_16281/ball_bearings/` | `multirow_solver.py`, `postprocessing.py`, `results.py` | `ISO16281MultiRowBallSolverSharedDisplacement` | Implemented |
| `bearings/.../multi_row/thrust_bearings/iso_16281/roller_bearings/` | `multirow_solver.py`, `postprocessing.py`, `results.py` | `ISO16281MultiRowRollerSolverSharedDisplacement` | Implemented, not exercised — no multi-row roller family exists in `core/` |
| `gears/` | `geometry.py`, `utils.py` | `GearSolver` and stateless helpers | Implemented |
| `gears/SpurHelicalGears/LoadCapacity_solver/` | `load_capacity.py` | ISO 6336 load capacity | Reserved — module exists, defines nothing |
| `mesh/` | `mesh_convergence_study.py` | `MeshConvergenceStudy`, `RichardsonGCI`, `MeshRefinementResult`, `ConvergenceRecord` | Implemented |

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

Run-level registries are not here. `RigidBearingFEMResultsLibrary`, `BearingResultsLibrary` and `RollingBearingSolver` all live in `fixtures/studies/`: collecting results across a whole gearbox is orchestration, and orchestration is a study concern.

> **Open item.** The multi-row branch keeps its own `results.py` in each of `ball_bearings/` and `roller_bearings/`, redefining `BallBearingResult`, `RollerBearingResult` and their per-row types under the same names as the copies in `results/`, with a different predicate (`is_multirow` here, `is_single` there). These belong in `results/bearings/load_distribution/multi_row/`. Until they move, a consumer that can receive a result from either branch cannot rely on either predicate. Separately, `roller_bearings/multirow_solver.py` imports `ball_bearings/results` rather than its own — either a wrong import or undeclared reuse.

---

## Structure

```
solvers/
├── machine_elements/
│   ├── shaft/
│   │   ├── fem_solvers/
│   │   │   ├── build_stiffness_matrix.py     Global stiffness assembly
│   │   │   ├── rigid_bearing.py              The full shaft solve
│   │   │   └── sub_models/
│   │   │       └── lagrange_multipliers.py   Subdomain solve with prescribed cut nodes
│   │   ├── static/
│   │   │   ├── results_reader.py             Raw solution → ShaftResults
│   │   │   ├── postprocessing.py             Stress concentration
│   │   │   └── static_failure.py             Reserved
│   │   └── utils.py                          Marin, Peterson, Neuber helpers
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
    └── mesh_convergence_study.py
```

The bearing tree is split by **row count first**, then by standard, then by contact type. That ordering is deliberate: single-row and multi-row are different solve *problems* — one ring displacement versus a load split across co-located rows — whereas ball and roller are different *contact laws* within each. Splitting the other way round would put two unrelated formulations in the same package.

---

## Import surface

| Import from | Names |
|---|---|
| `...shaft.fem_solvers.build_stiffness_matrix` | `StiffnessMatrixBuilder` |
| `...shaft.fem_solvers.rigid_bearing` | `RigidBearingFEMSolver` |
| `...shaft.fem_solvers.sub_models.lagrange_multipliers` | `SubmodelResult` |
| `...shaft.static.results_reader` | `ShaftResultsReader` |
| `...shaft.static.postprocessing` | `ShaftPostProcessor`, `PostProcessedResults`, `StressConcentration` |
| `...shaft.utils` | Marin factors, Peterson Kt, Neuber notch sensitivity, Kf |
| `...bearings.load_distribution.single_row.iso_16281.dispatch` | `resolve_solver_cls`, `resolve_solver_cls_for_attrs`, `register_contact_solver`, `SolverDispatchError` |
| `...single_row.iso_16281.ball_bearing.solver` | `ISO16281BallSolver` |
| `...single_row.iso_16281.ball_bearing.postprocessing` | `BallBearingStiffness`, `DynamicEquivalentRollingElementLoad`, `BasicReferenceRatingLife`, `DynamicEquivalentReferenceLoad`, `Q_j`, `phi_j_global`, `contact_distribution`, `bearing_stiffness`, `combine_row_L10r`, `basic_reference_rating_life`, `debug_radial_capacity` |
| `...single_row.iso_16281.roller_bearing.solver` | `ISO16281RollerSolver` |
| `...single_row.iso_16281.roller_bearing.postprocessing` | `RollerBearingStiffness`, `LaminaDynamicEquivalentLoad`, `BasicReferenceRatingLife`, `DynamicEquivalentReferenceLoad`, `Q_j`, `phi_j_global`, `contact_distribution`, `lamina_distribution`, `bearing_stiffness`, `stress_riser_factor`, `debug_radial_capacity` |
| `...multi_row.thrust_bearings.iso_16281.ball_bearings.multirow_solver` | `ISO16281MultiRowBallSolverSharedDisplacement` |
| `...multi_row.thrust_bearings.iso_16281.roller_bearings.multirow_solver` | `ISO16281MultiRowRollerSolverSharedDisplacement` |
| `...gears.geometry` | `GearSolver` |
| `axisforge.solvers.mesh` | `MeshConvergenceStudy`, `RichardsonGCI`, `MeshRefinementResult`, `ConvergenceRecord` |

Ball and roller packages are imported explicitly and never flattened into one namespace: point and line contact produce different result types, and flattening would hide which contact model a name belongs to. Note that `Q_j`, `phi_j_global`, `contact_distribution`, `bearing_stiffness` and `BasicReferenceRatingLife` are each defined **four times**, once per branch and contact type, with the exponents and contact law appropriate to that branch. They are not interchangeable; import from the module matching the bearing in hand.

---

## Conventions

| Symbol | Meaning |
|---|---|
| `x` | Axial coordinate, increasing left → right |
| `XZ` | Horizontal plane — tangential gear force `Wt` |
| `XY` | Vertical plane — radial gear force `Wr`, opposed to gravity |
| Torsion | Accumulates left → right; positive counter-clockwise viewed from `+x` |

Units, as stored: length and displacement in **mm**, force in **N**, bending moment in **N·mm**, torque in **N·m**, stress in **MPa**, stiffness in **N/mm**, angles in **rad**. The torque/bending-moment unit discontinuity is real and documented rather than normalised — see [`results/README.md`](../results/README.md#units).

Degrees of freedom are three per node: axial displacement, transverse displacement, rotation. Each bending plane is solved independently against the same stiffness matrix and superposed afterwards.

---

## Shaft FEM

### `StiffnessMatrixBuilder`

Assembles the global stiffness matrix from element contributions. The beam theory is selected by name from an extensible registry; Timoshenko with selective integration is wired today, and `euler_bernoulli` is the reserved second entry.

| Member | Purpose |
|---|---|
| `build_stiffness_matrix(mesh, elements)` | The assembled global matrix. |

### `RigidBearingFEMSolver`

Orchestrates the full shaft solve: node grid, element list, global stiffness, boundary conditions, two independent planar solves sharing the same stiffness matrix, superposition, and the torsion diagram on the same nodes. Bearings are treated as rigid supports — hence the name; a compliant-bearing solver would be a sibling module, not a mode of this one.

Every intermediate quantity is stored as a public attribute.

| Member | Purpose |
|---|---|
| `solve` | Runs the whole pipeline for one shaft system, optionally with extra mandatory nodes for local refinement. |
| `return_values` | Nodal solution quantities within an interval — the hand-off to the submodel solver. |
| `validate_torsion_equilibrium` | Torsion residual as a list of error strings. |

| Published attribute | Contents |
|---|---|
| `x_nodes`, `elements` | The mesh actually solved. |
| `free_dofs`, `constrained_dofs` | Boundary condition partition. |
| `d_total_xz`, `d_total_xy` | Superposed displacement vectors per plane. |
| `f_xz_ext`, `f_xy_ext` | External force vectors per plane. |
| `f_xz_reaction`, `f_xy_reaction` | Reaction vectors — non-zero only at constrained DOFs. |
| `f_xz_total`, `f_xy_total` | Total nodal force vectors. |
| `d_contributions` | Per-load-case displacement contributions. |
| `T_total`, `tau_total` | Torsion diagram and shear stress on the same nodes. |
| `torsion_contributions` | Per-source torsion contributions, per node. |

Construction options: the beam theory, the bearing constraint model, and the set of gear labels whose mesh loads are treated as distributed over the face width rather than as point loads.

Distributed loads are integrated by Gauss quadrature whose order is chosen adaptively from the estimated polynomial degree of the shape functions, the load intensity and the direction function — the caller never picks a quadrature rule.

**One solver instance per shaft.** A second `solve()` overwrites every published attribute. `fixtures/studies/shafts/fem_simple.solve_system()` creates a fresh solver per shaft for this reason.

### The submodel solve

`sub_models/lagrange_multipliers.py` restricts the solution to a subdomain, injecting the global solution at the cut nodes through Lagrange multipliers. Distributed loads are clamped to the subdomain bounds. It is what the mesh convergence study refines against.

`SubmodelResult` carries the subdomain bounds `x_lo`/`x_hi`, the grade, the local node list, the displacement vectors per plane, and the cut-node reaction multipliers.

---

## Shaft results and post-processing

### `ShaftResultsReader`

Post-processes a solved `RigidBearingFEMSolver` into a `ShaftResults`. It recovers internal forces, deflections, section properties, bearing reactions and per-bearing node data in one pass.

| Member | Purpose |
|---|---|
| `read()` | Returns a populated `ShaftResults`. |

The reader **does not know libraries exist**. It returns the container; storing it under a shaft name is the study layer's job. This is a change from the earlier design, where the reader wrote directly into a registry.

Two quantities produced here are easy to conflate and are kept deliberately separate, computed from different arrays:

| Quantity | Source | Meaning |
|---|---|---|
| Bearing **reactions** (`R_xz`, `R_xy`, `R`, `R_axial`) | the reaction vector | Constraint forces, non-zero only at constrained DOFs |
| Bearing node **loads** (`Fr_xz`, `Fr_xy`, `Fr`, `Fa`) | the total nodal force vector | Total force at that node |

At a bearing node with no other load applied at exactly that position they are numerically close, but they are not the same computed quantity and are not guaranteed identical. Report them separately; do not present one as a check on the other.

**Seat misalignment.** `_psi` computes the shaft centreline slope across the bearing seat width — the gradient of transverse displacement between the seat bounds, falling back to the nodal rotation for a zero-width seat. No second FEM pass is needed. This is the prescribed tilt the ISO/TS 16281 solvers consume, and it is what makes the shaft and bearing analyses one coupled calculation rather than two independent ones.

### Stress concentration

`ShaftPostProcessor` enriches a `ShaftResults` with stress concentration factors at shoulders and keyways, producing corrected stress arrays for the fatigue and failure solvers. It performs no FEM and no equilibrium — it consumes an already-solved result.

| Class | Purpose |
|---|---|
| `StressConcentration` | One feature: position `x`, kind (`shoulder`, `keyway`, `groove`, `press_fit`), `Kt_bending`/`Kt_torsion`, `Kf_bending`/`Kf_torsion`, `q_bending`/`q_torsion`, the geometry `r`, `D`, `d` that produced them, and a free-text `note`. |
| `PostProcessedResults` | The original `ShaftResults` by composition, plus the fatigue factor arrays, the corrected bending and shear stress arrays, the feature list, and the located maxima. |
| `ShaftPostProcessor` | `process()` walks the shaft's features and returns a `PostProcessedResults`. |

Shoulder factors come from the Peterson curve fits in Shigley's tables; notch sensitivity interpolates the Neuber constant against ultimate strength, referenced to the shear yield for torsion; the ultimate strength at each position is read from the material database, with a conservative fallback.

The stresses in `ShaftResults` are **nominal**. Notch effects live only in `PostProcessedResults`. A governing location that coincides with a shoulder or keyway is the signal to run the post-processor before drawing a conclusion.

### `shaft/utils.py`

Stateless helpers, reusable by the fatigue solvers.

| Group | Functions |
|---|---|
| Marin factors | Surface finish, size, load type, temperature, reliability (from a percentage or from a standard normal deviate), and the combined corrected endurance limit, which returns every factor alongside the result. |
| Notch factors | Theoretical shoulder Kt in bending and torsion, the Neuber material constant, notch sensitivity, and the fatigue factor from Kt and q. |

The load-type factor is per load mode; for combined bending and torsion the caller must not apply it to both terms.

---

## Bearings — ISO/TS 16281, single row

### Dispatch

A bearing is matched to a solver by the capabilities its family declares and the geometry attributes it actually carries. Each contact solver registers itself at import time.

| Name | Purpose |
|---|---|
| `resolve_solver_cls` | The single-row solver whose contract an assembled bearing satisfies. Ambiguity resolves to the most specific match. |
| `resolve_solver_cls_for_attrs` | The same match from a bare attribute set — how the rows of a multi-row bearing, which are plain dicts, are resolved. |
| `register_contact_solver` | The registration decorator, applied by the two single-row solvers. |
| `SolverDispatchError` | Raised, naming the offending bearing label, when nothing matches. |

Adding a bearing family therefore requires no edit to any dispatch table.

### Shared utilities

| Module | Contents |
|---|---|
| `numerics.py` | Root solve wrapper: a primary algorithm with a fallback, returning the solution, evaluation count, residual norm and success flag. |
| `validation.py` | Readiness guards — raises, naming the missing attribute, if a bearing has not been fully assembled for the requested solve; warns when a bearing declared floating is carrying axial load. |

Both are duplicated in the multi-row branch rather than shared across it. That is consistent with the vertical-slice rule, at the cost of two copies to keep in step.

### Ball bearings — point contact

**`ISO16281BallSolver`** — one class covers deep groove, angular contact, and each row of a multi-row thrust ball bearing, since the nominal contact angle is read off the bearing. Prescribed-tilt formulation: per raceway, a two-unknown root solve for radial and axial approach in the plane of the resultant radial force.

| Member | Purpose |
|---|---|
| `solve` | Solves every point-contact bearing in the given set against the shaft results library. |
| `solve_contact` | The two-equation root solve for one raceway — a whole single-row bearing, or one row of a multi-row one. |
| `elements` | Per-element elastic deflection and effective contact angle for a given approach and tilt. |
| `minimum_axial_load` | The smallest axial preload at which the contact closes, by bracketing on the axial approach. |
| `debug_radial_capacity` | Prints per-element capacities for manual cross-check, sourced from the same family call production uses. |

Initial approach values trust a non-negligible FEM hint and fall back to a Hertz-scale estimate otherwise.

**Post-processing.** Every per-element function returns one entry per row.

| Function or class | Purpose |
|---|---|
| `Q_j` | Per-element contact force from the converged deflections, `Q_j = c_p · delta_j^1.5`. |
| `phi_j_global` | Element angular positions in the global frame, wrapped to `[0, 2π)`. |
| `contact_distribution` | Angle and contact force pairs, in the global or local frame. |
| `BallBearingStiffness`, `bearing_stiffness` | Secant stiffness decomposed onto the two bending planes and the axis, with a regime label distinguishing `no_load`, `engaged` and `closing_clearance`. |
| `DynamicEquivalentRollingElementLoad` | Dynamic equivalent load per rolling element for the inner and outer raceway, with the exponent chosen by which ring rotates relative to the load. |
| `BasicReferenceRatingLife`, `combine_row_L10r`, `basic_reference_rating_life` | Per-row lives and the combined bearing life. |
| `DynamicEquivalentReferenceLoad` | The reference load corresponding to a computed life, radial and axial. |

Per-element contact forces are **not stored** on the result — they are derived on demand from the converged deflections, so there is one source of truth.

### Roller bearings — line contact

**`ISO16281RollerSolver`** — the lamina-model solver for radial cylindrical roller bearings: zero nominal contact angle, no axial capacity, each roller sliced into at least thirty laminae and corrected for its logarithmic profile.

| Member | Purpose |
|---|---|
| `solve` | Solves every line-contact bearing in the given set. Validates the lamina count and the cached profile length. |
| `solve_contact` | The one-unknown root solve on radial approach. |
| `elements` | Per-roller and per-lamina deflection and load for a given approach and tilt. |
| `debug_radial_capacity` | Prints whole-roller and per-lamina capacities for manual cross-check. |

Only the radial approach is solved. The tilt is an input taken from the FEM shaft slope, and moment equilibrium is evaluated afterwards as a diagnostic rather than as a solve constraint. The reference roller profile is read from the bearing, where it was cached at assembly — the solver never computes it.

**Post-processing.** Same per-row list convention as the ball side.

| Function or class | Purpose |
|---|---|
| `Q_j` | Total force per roller, summed over its laminae. |
| `phi_j_global`, `contact_distribution` | As the ball side. |
| `lamina_distribution` | Position and load along a single roller — the pressure-profile view, and where edge loading becomes visible. |
| `RollerBearingStiffness`, `bearing_stiffness` | Secant stiffness in the two bending planes. Axial stiffness is reported as unloaded, which is the physically correct statement for a radial roller bearing. |
| `stress_riser_factor` | Approximate edge-stress concentration along the roller. Its validity conditions — moderate load, small misalignment, logarithmic profile — are **not** checked. |
| `LaminaDynamicEquivalentLoad` | Dynamic equivalent load per lamina, both raceways, ISO/TS 16281 §5.3.4 eq. (61)–(64). Compare against the family's per-lamina capacities `q_ci`/`q_ce`, never against the whole-roller `Q_ci`/`Q_ce`. |
| `BasicReferenceRatingLife`, `combine_row_L10r`, `basic_reference_rating_life`, `DynamicEquivalentReferenceLoad` | As the ball side, with the line-contact exponents. |

Combining reference rating lives into a **modified** rating life is not implemented on either side.

---

## Bearings — ISO/TS 16281, multi-row thrust

Multi-row is a genuinely different solve, not a mode of the single-row one: axially stacked rows share one rigid ring, so the ring displacement is a single unknown and each row's reaction is summed against it.

| Class | Purpose |
|---|---|
| `ISO16281MultiRowBallSolverSharedDisplacement` | One root solve on the shared displacement of the rigid ring; each row contributes its own Hertzian reaction. Reaches individual rows through `ISO16281BallSolver.elements`. |
| `ISO16281MultiRowRollerSolverSharedDisplacement` | The line-contact equivalent, mirroring the ball side. |

A second multi-row formulation based on load-split fractions was tried and is not registered: with negligible axial load and rows of differing stiffness its Jacobian degenerates and it fails to converge. The shared-displacement formulation is the registered one for that reason.

**What "multi-row" does not mean.** For a *radial* bearing — deep groove ball or cylindrical roller — a second row is a plain capacity-rating multiplier on one raceway (ISO 281:2007 Table 1), solved as a single `delta_r` with no rows list and no load split. That case is handled by the single-row solver and the family's own `i`, not here. This package is for **thrust** duty only, where the rows genuinely share a compatibility solve.

The roller multi-row solver has no multi-row roller family in `core/` to run against, so it has not been exercised on a real case.

---

## Gears

**`GearSolver`** — cylindrical gear geometry and mesh forces, following ISO 21771 and the MAAG conventions. All methods are pure functions with no internal state.

| Member | Purpose |
|---|---|
| `compute_geometry` | Full pair geometry with optional profile shift: working centre distance from an imposed value or from the involute equation, transverse and base quantities, tip, root and working diameters, base pitch, and the three contact ratios. |
| `compute_forces` | Tangential, radial, axial, base-tangential and normal forces from the pinion torque, plus the output torque and ratio. |
| `to_gear_element` | Assembles a `GearElement` for injection into the system pipeline, referenced to the working pitch circle so the torque consistency check holds. |

**`gears/utils.py`** — stateless building blocks: the involute function, the working pressure angle and centre distance from the involute equation, transverse and overlap contact ratios, the minimum tooth count to avoid undercut, input validation, and the standard rack constants.

**`SpurHelicalGears/LoadCapacity_solver/load_capacity.py`** is reserved: the module exists and defines nothing yet. Its data layer — application factor `K_A` and dynamic factor `K_v` by Method B and Method C — is already in [`axisforge/database/`](../core/README.md#database), though those two modules currently import a `core` package path that no longer exists and need repairing before they can be consumed.

---

## Mesh convergence

Grid Convergence Index by Richardson extrapolation on the resultant transverse displacement, across three or more refinement levels. It drives repeated submodel solves, which is why it lives with the solvers rather than with the mesh primitives it refines.

| Class | Purpose |
|---|---|
| `RichardsonGCI` | For one triplet of coarse, medium and fine solutions: the refinement ratios, the observed order of convergence, the relative errors, the extrapolated value, the two convergence indices and their pass flags. |
| `ConvergenceRecord` | The refinement history for one interval: levels attempted, metric history, index history, the converged flag, and the final node set. |
| `MeshRefinementResult` | One record per interval, plus `all_extra_nodes` (the union of every final node set) and `print_report`. |
| `MeshConvergenceStudy` | The orchestrator. |

| `MeshConvergenceStudy` member | Purpose |
|---|---|
| `run` | Refines each requested interval until the convergence index falls below the threshold or the level cap is reached. |
| `intervals_from_shaft_system` | Derives the intervals worth refining from the shaft's elements, and reports what was skipped. |

Bearing extents are excluded automatically: displacement there is a prescribed boundary condition, so refining it gains nothing. The default index threshold is 1% with a safety factor of 1.25 and a cap of eight levels. The convergence metric is the mean absolute transverse displacement per plane and its resultant over physically meaningful evaluation points inside the interval.

`all_extra_nodes` is the set to lock into production runs as extra mandatory nodes.

---

## Design contracts

- **A solver computes; it does not collect.** No solver owns a registry keyed by shaft or bearing label. That is the study layer's job.
- **No solver imports another solver.** Ball and roller, single-row and multi-row, shaft and bearing: each is a vertical slice. They meet at the dispatcher and at the result containers.
- **No hidden state.** Every intermediate quantity is a public attribute. A solver exposes its full working for inspection.
- **One instance per subject.** A solver's published attributes describe the last thing it solved. Re-solving overwrites them.
- **Capacity belongs to the element.** A solver never carries its own copy of a standard's capacity formula; it calls the family through the bearing.
- **A failed solve is still a result.** Convergence metadata (`n_iter`, `residual`, `ok`) is stored, never raised away. Checking it is the consumer's responsibility.
- **Derive, do not duplicate.** Quantities recoverable from the converged solution — per-element contact force, angular positions — are computed on demand, not stored.
- **Flag, do not silently fix.** A suspected discrepancy against a standard is documented in place. `stress_riser_factor` not checking its own validity conditions is stated here rather than silently guarded.

---

## Extending this package

**Adding a bearing family.** Nothing in this package changes. Declare the family's `CAPABILITIES` and `REQUIRED_FOR` in `core/`; dispatch resolves it from those declarations. If the family needs a contact law neither solver implements, that is a new solver module under the matching branch, registered with `register_contact_solver`.

**Adding a beam theory.** Register it in `StiffnessMatrixBuilder`'s theory registry and add the element module under `mesh/shaft/element_type/`. `euler_bernoulli` is the reserved slot.

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