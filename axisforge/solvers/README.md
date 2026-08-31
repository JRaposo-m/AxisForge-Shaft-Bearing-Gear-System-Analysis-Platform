# axisforge/solvers — Module Reference

Solvers consume the domain objects in [`core/README.md`](../core/README.md) and the mesh objects in [`mesh/README.md`](../mesh/README.md), and publish results as explicit, GUI-independent containers. No solver imports another solver's internals — only the shared result containers.

← back to [project root](../../README.md)

---

## Table of Contents

- [Import surface](#import-surface)
- [Shaft FEM](#shaft-fem)
- [Shaft results and post-processing](#shaft-results-and-post-processing)
- [Bearings — ISO/TS 16281](#bearings--isots-16281)
- [Bearings — rating life](#bearings--rating-life)
- [Gears](#gears)
- [Mesh convergence](#mesh-convergence)
- [Lubrication](#lubrication)

---

## Import surface

| Import from | Names |
|---|---|
| `...shaft.oneD_analysis.build_stiffness_matrix` | `StiffnessMatrixBuilder` |
| `...shaft.oneD_analysis.FEM_solvers` | `SimpleFEMSolver`, `SubmodelSolver`, `SubmodelResult` |
| `...shaft.oneD_analysis.static` | `ShaftResults`, `ShaftResultsReader`, `SimpleFEMResultsLibrary`, `BearingNodeData` |
| `...shaft.oneD_analysis.static.shaft_post_processor` | `ShaftPostProcessor`, `PostProcessedResults`, `StressConcentration` |
| `...shaft.utils` | Marin factors, Peterson Kt, Neuber notch sensitivity, Kf |
| `...bearings.ISO_16281` | `RollingBearingSolver`, `BearingResultsLibrary`, `warn_if_floating_loaded`, `resolve_solver_cls`, `resolve_solver_cls_for_attrs`, `SolverDispatchError` |
| `...bearings.ISO_16281.Ball_Bearing` | `ISO16281BallSolver`, `ISO16281MultiRowBallSolverSharedDisplacement`, `BallBearingResult`, `BallLoadDistributionResult`, `BallLoadDistributionLibrary`, `BallBearingStiffness`, `DynamicEquivalentRollingElementLoad`, `Q_j`, `phi_j_global`, `contact_distribution`, `bearing_stiffness`, `debug_radial_capacity` |
| `...bearings.ISO_16281.Roller_Bearing` | `ISO16281RollerSolver`, `RollerBearingResult`, `RollerLoadDistributionResult`, `RollerLoadDistributionLibrary`, `RollerBearingStiffness`, `LaminaDynamicEquivalentLoad`, `Q_j`, `phi_j_global`, `contact_distribution`, `lamina_distribution`, `bearing_stiffness`, `stress_riser_factor`, `debug_radial_capacity` |
| `...bearings.life` | `BearingLifeSolver` |
| `...gears.geometry` | `GearSolver` |
| `axisforge.solvers.mesh` | `MeshConvergenceStudy`, `RichardsonGCI`, `MeshRefinementResult`, `ConvergenceRecord` |

`ISO_16281` exports only the orchestration layer. `Ball_Bearing` and `Roller_Bearing` are imported explicitly: point and line contact produce different result types, and flattening them into one namespace would hide which contact model a name belongs to.

---

## Shaft FEM

### `StiffnessMatrixBuilder`

Assembles the global stiffness matrix from element contributions, with three degrees of freedom per node (axial displacement, transverse displacement, rotation). The beam theory is selected by name from an extensible registry; Timoshenko with selective integration is wired today.

### `SimpleFEMSolver`

Orchestrates the full shaft solve: node grid, element list, global stiffness, boundary conditions, two independent planar solves sharing the same stiffness matrix, superposition, and the torsion diagram on the same nodes. Every intermediate quantity is stored as a public attribute.

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
| `d_contributions` | Per-load-case displacement contributions. |
| `T_total`, `tau_total` | Torsion diagram and shear stress on the same nodes. |
| `torsion_contributions` | Per-source torsion contributions, per node. |

Construction options: the beam theory, the bearing constraint model, and the set of gear labels whose mesh loads are treated as distributed over the face width rather than as point loads.

Distributed loads are integrated by Gauss quadrature whose order is chosen adaptively from the estimated polynomial degree of the shape functions, the load intensity and the direction function — the caller never picks a quadrature rule.

One solver instance per shaft: a second `solve()` overwrites the published attributes.

### `SubmodelSolver` and `SubmodelResult`

Wraps `SimpleFEMSolver` and restricts the solution to a subdomain, injecting the global solution at the cut nodes through Lagrange multipliers. Distributed loads are clamped to the subdomain bounds. Used by the mesh convergence study.

| Member | Purpose |
|---|---|
| `solve` | Refines the subdomain to a requested grade, rebuilds the local stiffness and load vectors, and solves with the cut-node constraints. |

`SubmodelResult` carries the subdomain bounds, the grade, the local node list, the displacement vectors per plane, and the cut-node reaction multipliers.

---

## Shaft results and post-processing

### Result containers

| Class | Purpose |
|---|---|
| `ShaftResults` | The complete output for one shaft: mesh, raw FEM solution, torsion, post-processed engineering quantities, and per-bearing node data. |
| `BearingNodeData` | The complete FEM nodal state at one bearing position. |
| `SimpleFEMResultsLibrary` | Registry of `ShaftResults` keyed by shaft name — the canonical source every downstream solver reads from. |
| `ShaftResultsReader` | Post-processes a solved `SimpleFEMSolver` into a `ShaftResults` and stores it in the library. |
| `StaticFailure` | Reserved for static failure criteria; not implemented. |

`ShaftResults` is organised in four groups:

| Group | Contents |
|---|---|
| Mesh | Node positions and the element list. |
| Raw FEM solution | Global stiffness, DOF partition, displacement and force vectors per plane, reaction vectors. |
| Torsion | Torque and shear-stress diagrams, per-source contributions. |
| Engineering quantities | Axial coordinate, bending moments per plane and resultant, shear forces per plane and resultant, deflections per plane and resultant, torque, diameter and section moduli along the shaft. |

`BearingNodeData` carries the label and position, the nodal displacements and rotations, the radial reactions per plane and their resultant, the axial reaction, the moment reactions, and the seat misalignment — the gradient of transverse displacement across the seat width, falling back to the nodal rotation for a zero-width seat. This misalignment is the input the ISO/TS 16281 solvers use as their prescribed tilt.

| `SimpleFEMResultsLibrary` member | Purpose |
|---|---|
| `store` | Add or overwrite the results for one shaft. |
| `get`, `get_or_none` | Retrieve by shaft name. |
| `remove`, `clear` | Drop one or all entries. |
| `names`, `iter`, `all_results` | Enumerate in insertion order. |

`ShaftResultsReader.read()` recovers internal forces, deflections, section properties, bearing reactions and node data in one pass, and stores the result under the shaft system's name.

### Stress concentration

`ShaftPostProcessor` enriches a `ShaftResults` with stress concentration factors at shoulders and keyways, producing corrected stress arrays for the fatigue and failure solvers. It performs no FEM and no equilibrium — it consumes an already-solved result.

| Class | Purpose |
|---|---|
| `StressConcentration` | One feature: position, kind (shoulder, keyway, groove, press fit), theoretical and fatigue factors in bending and torsion, notch sensitivity, and the geometry that produced them. |
| `PostProcessedResults` | The original `ShaftResults` by composition, plus the fatigue factor arrays, the corrected bending and shear stress arrays, the feature list, and the located maxima. |
| `ShaftPostProcessor` | `process()` walks the shaft's features and returns a `PostProcessedResults`. |

Shoulder factors come from the Peterson curve fits in Shigley's tables; notch sensitivity interpolates the Neuber constant against ultimate strength, referenced to the shear yield for torsion; the ultimate strength at each position is read from the material database, with a conservative fallback.

### `shaft/utils.py`

Stateless helpers, reusable by the fatigue solvers.

| Group | Functions |
|---|---|
| Marin factors | Surface finish, size, load type, temperature, reliability (from a percentage or from a standard normal deviate), and the combined corrected endurance limit, which returns every factor alongside the result. |
| Notch factors | Theoretical shoulder Kt in bending and torsion, the Neuber material constant, notch sensitivity, and the fatigue factor from Kt and q. |

The load-type factor is per load mode; for combined bending and torsion the caller must not apply it to both terms.

---

## Bearings — ISO/TS 16281

### How the package is organised

```
ISO_16281/
├── dispatch.py                 Solver resolution by capability and required attributes
├── library.py                  Generic solve utilities, cross-type results registry
├── rolling_bearing_solver.py   RollingBearingSolver — the entry point
├── Ball_Bearing/               Point contact: solver, results, postprocessing
└── Roller_Bearing/             Line contact:  solver, results, postprocessing
```

Each contact type owns a full vertical slice — solve, result shape and postprocessing — and never depends on the other at runtime. The two are wired together at exactly three points: the shared results registry, the dispatcher, and the orchestrator.

### Dispatch

A bearing is matched to a solver by the capabilities its family declares and the geometry attributes it actually carries. Each contact solver registers itself at import time.

| Name | Purpose |
|---|---|
| `resolve_solver_cls` | The single-row solver whose contract an assembled bearing satisfies. Ambiguity resolves to the most specific match. |
| `resolve_solver_cls_for_attrs` | The same match from a bare attribute set — how the rows of a multi-row bearing, which are plain dicts, are resolved. |
| `register_contact_solver` | The registration decorator, applied internally by the two single-row solvers. |
| `SolverDispatchError` | Raised, naming the offending bearing label, when nothing matches. |

Adding a bearing family therefore requires no edit to any dispatch table.

### `RollingBearingSolver`

The single entry point for a shaft's full bearing set, which may mix contact types and row counts.

| Member | Purpose |
|---|---|
| `solve` | Groups single-row bearings by resolved solver and dispatches each group; sends multi-row bearings individually to the multi-row solver registered on their row solver. Returns per-label lists of row results, in the order the caller supplied the bearings. |
| `postprocess_and_record` | The full pipeline — solve (or reuse an existing load distribution), then capacity, dynamic equivalent load and secant stiffness — recorded into a `BearingResultsLibrary`. |

`postprocess_and_record` takes a `catalog` mapping each bearing label to the arguments for its capacity and equivalent-load steps. Either sub-entry may be omitted to skip that step; stiffness is always computed. Capacity is obtained by calling the family through the bearing, so the orchestrator holds no ISO formulas of its own.

### Shared utilities and the results registry

| Name | Purpose |
|---|---|
| `check_bearing_ready` | Raises, naming the missing attribute, if a bearing has not been fully assembled for the requested solve. |
| `warn_if_floating_loaded` | Warns when a bearing declared floating is carrying axial load. |
| `run_root` | Root solve wrapper: a primary algorithm with a fallback, returning the solution, evaluation count, residual norm and success flag. |
| `BearingResultBundle` | Everything computed for one bearing label: bearing type, per-row load distribution, stiffness, capacity, dynamic equivalent load, and an open `extra` dictionary. |
| `BearingResultsLibrary` | Registry of bundles keyed by label — what lubrication, fatigue and life solvers read from. |

| `BearingResultsLibrary` member | Purpose |
|---|---|
| `add_load_distribution_library` | Records an entire per-solver local library at once. |
| `load_distribution_library` | Retrieves that whole sub-library for one solver class. |
| `set_load_distribution` | Sets one label directly, bypassing a local library. |
| `set_capacity`, `set_dynamic_equivalent_load`, `set_stiffness`, `set_bearing_type`, `set_extra` | Per-label setters; each overwrites. |
| `get`, `labels` | Retrieval and enumeration; `get` raises for an unrecorded label. |

### Result shapes

Single-row and multi-row bearings share one result type per contact model.

| Class | Purpose |
|---|---|
| `BallLoadDistributionResult` | The raw output of one point-contact raceway solve: radial and axial approach, misalignment, resultant-force angle, per-element deflection and contact angle, the diagnostic tilting moment, iteration count, residual and convergence flag. |
| `BallBearingResult` | The per-bearing container. `rows` is a list — length one for a single-row bearing, one entry per row otherwise. Multi-row solves also carry the outer-iteration load fractions and diagnostics. |
| `BallLoadDistributionLibrary` | Local registry of ball results by label. |
| `RollerLoadDistributionResult` | The line-contact equivalent, with axial approach fixed at zero and the lamina-model arrays added: lamina positions, per-roller tilt, per-lamina deflection and per-lamina load. |
| `RollerBearingResult` | The per-bearing container, same shape as the ball one. |
| `RollerLoadDistributionLibrary` | Local registry of roller results by label. |

Per-element contact forces are not stored — they are derived on demand from the converged deflections by the postprocessing functions.

### Ball bearings — point contact

**`ISO16281BallSolver`** — one class covers deep groove, angular contact, and each row of a multi-row thrust ball bearing, since the nominal contact angle is read off the bearing. Prescribed-tilt formulation: per raceway, a two-unknown root solve for radial and axial approach in the plane of the resultant radial force.

| Member | Purpose |
|---|---|
| `solve` | Solves every point-contact bearing in the given set against the FEM results library, returning a local library. |
| `solve_contact` | The two-equation root solve for one raceway — a whole single-row bearing, or one row of a multi-row one. |
| `elements` | Per-element elastic deflection and effective contact angle for a given approach and tilt. |
| `minimum_axial_load` | The smallest axial preload at which the contact closes, by bracketing on the axial approach. |
| `debug_radial_capacity` | Prints per-element capacities for manual cross-check, sourced from the same family call production uses. |

Initial approach values trust a non-negligible FEM hint and fall back to a Hertz-scale estimate otherwise.

**`ISO16281MultiRowBallSolverSharedDisplacement`** — the multi-row solver reached by dispatch. One root solve on the shared displacement of the rigid ring; each row contributes its own Hertzian reaction, summed. It returns the same result type as the single-row solver, so nothing downstream changes.

A second multi-row formulation based on load-split fractions is kept in the package for comparison. It is not part of dispatch: with negligible axial load and rows of differing stiffness its Jacobian degenerates and it fails to converge, which is why the shared-displacement formulation is the registered one.

**Postprocessing.** Every per-element function returns one entry per row.

| Function or class | Purpose |
|---|---|
| `Q_j` | Per-element contact force from the converged deflections. |
| `phi_j_global` | Element angular positions in the global frame. |
| `contact_distribution` | Angle and contact force pairs, in the global or local frame. |
| `BallBearingStiffness`, `bearing_stiffness` | Secant stiffness decomposed onto the two bending planes and the axis, with an axial regime label distinguishing no load, engaged contact and closing clearance. |
| `DynamicEquivalentRollingElementLoad` | Dynamic equivalent load per rolling element for the inner and outer raceway, with the exponent chosen by which ring rotates relative to the load. |
| `BasicReferenceRatingLife` | Basic reference rating life for one row and raceway pair. |
| `combine_row_L10r` | Combines per-row lives into a bearing-level life. |
| `basic_reference_rating_life` | Per-row lives and the combined bearing life in one call. |
| `DynamicEquivalentReferenceLoad` | The reference load corresponding to a computed life, radial and axial. |

### Roller bearings — line contact

**`ISO16281RollerSolver`** — the lamina-model solver for radial cylindrical roller bearings: zero nominal contact angle, no axial capacity, each roller sliced into at least thirty laminae and corrected for its logarithmic profile.

| Member | Purpose |
|---|---|
| `solve` | Solves every line-contact bearing in the given set, returning a local library. Validates the lamina count and the cached profile length. |
| `solve_contact` | The one-unknown root solve on radial approach. |
| `elements` | Per-roller and per-lamina deflection and load for a given approach and tilt. |
| `debug_radial_capacity` | Prints whole-roller and per-lamina capacities for manual cross-check. |

Only the radial approach is solved; the tilt is an input taken from the FEM shaft slope, and the moment equilibrium is evaluated afterwards as a diagnostic rather than as a solve constraint. The reference roller profile is read from the bearing, where it was cached at assembly — the solver never computes it.

A shared-displacement multi-row roller solver exists and is registered, mirroring the ball side. No multi-row roller family exists in `core/` yet, so it has not been exercised on a real case.

**Postprocessing.** Same per-row list convention as the ball side.

| Function or class | Purpose |
|---|---|
| `Q_j` | Total force per roller, summed over its laminae. |
| `phi_j_global`, `contact_distribution` | As the ball side. |
| `lamina_distribution` | Position and load along a single roller — the pressure-profile view. |
| `RollerBearingStiffness`, `bearing_stiffness` | Secant stiffness in the two bending planes. Axial stiffness is reported as unloaded, which is the physically correct statement for a radial roller bearing. |
| `stress_riser_factor` | Approximate edge-stress concentration along the roller. Its validity conditions — moderate load, small misalignment, logarithmic profile — are not checked. |
| `LaminaDynamicEquivalentLoad` | Dynamic equivalent load per lamina for both raceways. Compare against the family's per-lamina capacities, not the whole-roller ones. |
| `BasicReferenceRatingLife`, `combine_row_L10r`, `basic_reference_rating_life`, `DynamicEquivalentReferenceLoad` | As the ball side, with the line-contact exponents. |

Combining reference rating lives into a modified rating life is not implemented on either side.

---

## Bearings — rating life

**`BearingLifeSolver`** — ISO 281 basic rating life and static safety, separate from the ISO/TS 16281 internal-distribution stack.

| Member | Purpose |
|---|---|
| `solve_bearing` | Basic rating life and static safety factor for one bearing from its radial and axial load, speed and design life. |
| `extract_bearing_forces` | Radial and axial load per bearing label from a statics result. |

The equivalent load currently reduces to the radial load: the axial contribution factors are not yet applied.

---

## Gears

**`GearSolver`** — cylindrical gear geometry and mesh forces, following ISO 21771 and the MAAG conventions. All methods are pure functions with no internal state.

| Member | Purpose |
|---|---|
| `compute_geometry` | Full pair geometry with optional profile shift: working centre distance from an imposed value or from the involute equation, transverse and base quantities, tip, root and working diameters, base pitch, and the three contact ratios. |
| `compute_forces` | Tangential, radial, axial, base-tangential and normal forces from the pinion torque, plus the output torque and ratio. |
| `to_gear_element` | Assembles a gear element for injection into the system pipeline, referenced to the working pitch circle so the torque consistency check holds. |

**`gears/utils.py`** — stateless building blocks: the involute function, the working pressure angle and centre distance from the involute equation, transverse and overlap contact ratios, the minimum tooth count to avoid undercut, input validation, and the standard rack constants.

The ISO 6336 load capacity module is reserved and not yet written. Its data layer — application factor and dynamic factor by Method B and Method C — already exists in [`axisforge/database/`](../core/README.md#database).

---

## Mesh convergence

Grid Convergence Index by Richardson extrapolation on the resultant transverse displacement, across three or more refinement levels. It drives repeated submodel solves, which is why it lives with the solvers rather than with the mesh primitives it refines.

| Class | Purpose |
|---|---|
| `RichardsonGCI` | For one triplet of coarse, medium and fine solutions: the refinement ratios, the observed order of convergence, the relative errors, the extrapolated value, the two convergence indices and their pass flags. |
| `ConvergenceRecord` | The refinement history for one interval: levels attempted, metric history, index history, the converged flag, and the final node set. |
| `MeshRefinementResult` | One record per interval, plus the union of every final node set and a printable report. |
| `MeshConvergenceStudy` | The orchestrator. |

| `MeshConvergenceStudy` member | Purpose |
|---|---|
| `run` | Refines each requested interval until the convergence index falls below the threshold or the level cap is reached. |
| `intervals_from_shaft_system` | Derives the intervals worth refining from the shaft's elements, and reports what was skipped. |

Bearing extents are excluded automatically: the displacement there is a prescribed boundary condition, so refining it gains nothing. The default index threshold is 1% with a safety factor of 1.25 and a cap of eight levels. The convergence metric is the mean absolute transverse displacement per plane and its resultant over physically meaningful evaluation points inside the interval.

The union of final node sets is the set to lock into production runs as extra mandatory nodes.

---

## Lubrication

`solvers/lubrification/` is reserved and exports nothing. It is the home for film thickness, lubrication regime and grease solvers in a later phase, and is documented here so the empty package is not mistaken for an oversight.