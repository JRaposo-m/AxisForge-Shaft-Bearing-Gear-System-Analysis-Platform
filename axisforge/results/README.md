# axisforge/results — Module Reference

Result containers: the explicit, immutable-by-convention data shapes that every solver in AxisForge writes into and every downstream consumer reads from. This package computes nothing. It defines *what a result is* — its fields, its units, its sign convention and its invariants — so that no two parts of the platform have to agree on an implicit tuple layout.

`results/` is the narrow waist of the architecture. A solver depends on the containers it produces; a post-processor, a report writer or the UI depends on the containers it consumes. Neither depends on the other.

← back to [project root](../../README.md)

---

## Table of Contents

- [Scope](#scope)
- [Status](#status)
- [Position in the architecture](#position-in-the-architecture)
- [Structure](#structure)
- [Import surface](#import-surface)
- [Conventions](#conventions)
  - [Axis and plane convention](#axis-and-plane-convention)
  - [Units](#units)
  - [Sign convention](#sign-convention)
- [Module reference](#module-reference)
  - [`fem_results/shaft_results.py`](#fem_resultsshaft_resultspy)
  - [`bearings/load_distribution/single_row/ball_bearing_results.py`](#bearingsload_distributionsingle_rowball_bearing_resultspy)
  - [`bearings/load_distribution/single_row/roller_bearing_results.py`](#bearingsload_distributionsingle_rowroller_bearing_resultspy)
- [Reading the results](#reading-the-results)
  - [Shaft — governing values](#shaft--governing-values)
  - [Shaft — diagrams](#shaft--diagrams)
  - [Bearing seat state](#bearing-seat-state)
  - [Bearing internal load distribution](#bearing-internal-load-distribution)
  - [Convergence diagnostics](#convergence-diagnostics)
- [Consumers](#consumers)
- [Usage](#usage)
- [Design contracts](#design-contracts)
- [Extending this package](#extending-this-package)
- [Standards referenced](#standards-referenced)

---

## Scope

| In scope | Out of scope |
|---|---|
| Field definitions, units, shapes and array alignment | Any numerical procedure |
| Convergence and diagnostic metadata produced by a solve | The iteration that produced it |
| Row-aware containers for multi-row bearings | The load-split solve across rows |
| Documented invariants a consumer may rely on | Enforcement of those invariants at runtime |

A result object is a **record of a completed solve**, not a service. It has no `solve()`, no `update()`, no lazy recomputation and no reference back to the solver that filled it. Everything a downstream consumer needs is a public attribute, already evaluated.

---

## Status

| Group | Module | Contents | Status |
|---|---|---|---|
| `fem_results/` | `shaft_results.py` | `ShaftResults`, `BearingNodeData` | Implemented |
| `bearings/load_distribution/single_row/` | `ball_bearing_results.py` | `BallBearingResult`, `BallLoadDistributionResult` | Implemented |
| `bearings/load_distribution/single_row/` | `roller_bearing_results.py` | `RollerBearingResult`, `RollerLoadDistributionResult` | Implemented |
| `bearings/load_distribution/multi_row/` | thrust ball / thrust roller results | — | Not migrated; still local to the solver package, with a divergent API |
| `gears/` | gear rating results (ISO 6336) | — | Planned |
| `fatigue/` | fatigue and static failure results | — | Planned |
| `lubrication/` | film thickness and viscosity ratio results | — | Planned |

---

## Position in the architecture

The wiring law permits `solvers → results` and forbids the reverse. `results/` imports nothing from `axisforge` at all — only NumPy and the standard library.

```
                 core/            mesh/
                   │                │
                   └───────┬────────┘
                           ▼
                       solvers/                  ← computes
                           │  writes
                           ▼
                       results/                  ← this package: shapes only
                           │  reads
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
     fixtures/          ui/            downstream solvers
   (libraries,       (viewers,        (ISO/TS 16281 coupling,
    reports)          exporters)       fatigue, failure)
```

Two consequences follow, and both are deliberate:

- **A result container never knows which solver produced it.** The ISO/TS 16281 bearing solvers consume `ShaftResults` without importing anything from the shaft FEM package. The coupling is the data shape, nothing more.
- **Registries do not live here.** A result *shape* is a container; a *library* keyed by shaft or bearing label is an orchestration concern and lives in `fixtures/studies/` — `RigidBearingFEMResultsLibrary` for shafts, `BearingResultsLibrary` for bearings. `results/` defines what one result is; it does not define how a run collects many of them.

> **Open item — the multi-row containers have not moved yet.** `solvers/.../multi_row/thrust_bearings/iso_16281/ball_bearings/results.py` and its roller counterpart still define `BallBearingResult`, `RollerBearingResult` and their per-row types locally, under the same names as the copies here but with a different predicate: `is_multirow` there, `is_single` here. Two classes with one name and contradictory predicates is a trap for any consumer that can receive a result from either branch. They belong in `results/bearings/load_distribution/multi_row/`, and this document is written on the assumption that is where they are going.

---

## Structure

```
results/
├── fem_results/
│   └── shaft_results.py                     ShaftResults, BearingNodeData
└── bearings/
    └── load_distribution/
        └── single_row/
            ├── ball_bearing_results.py      BallBearingResult,
            │                                BallLoadDistributionResult
            └── roller_bearing_results.py    RollerBearingResult,
                                             RollerLoadDistributionResult
```

The tree mirrors the analysis it records, not the package that computes it: `results/bearings/load_distribution/single_row/` sits opposite `solvers/machine_elements/bearings/load_distribution/single_row/`. A reader who knows where a solver lives knows where its result lives.

---

## Import surface

| Import from | Names |
|---|---|
| `axisforge.results.fem_results.shaft_results` | `ShaftResults`, `BearingNodeData` |
| `axisforge.results.bearings.load_distribution.single_row.ball_bearing_results` | `BallBearingResult`, `BallLoadDistributionResult` |
| `axisforge.results.bearings.load_distribution.single_row.roller_bearing_results` | `RollerBearingResult`, `RollerLoadDistributionResult` |

Ball and roller results are imported explicitly and are never flattened into a single namespace. Point contact and line contact produce genuinely different quantities — a ball result has no per-lamina array, a roller result has no varying contact angle — and a shared namespace would hide which contact model a name belongs to.

---

## Conventions

Every field in this package obeys the conventions below. They are load-bearing: a consumer that reads a field is entitled to assume them, and changing one is a breaking change to the whole platform, not a local edit.

### Axis and plane convention

| Symbol | Meaning |
|---|---|
| `x` | Axial coordinate, increasing left → right |
| `XZ` | Horizontal plane — carries the tangential gear force `Wt` |
| `XY` | Vertical plane — carries the radial gear force `Wr`, opposed to gravity |
| Torsion | Accumulates left → right; positive is counter-clockwise viewed from `+x` |

A radial load applied at an angle `theta_deg` (measured `+Y → +Z`, right-hand about `+X`) contributes to **both** planes. Any field suffixed `_xz` or `_xy` is one plane's component of a quantity whose resultant is stored unsuffixed.

### Units

| Quantity | Unit |
|---|---|
| Length, position, deflection, displacement | mm |
| Force | N |
| Bending moment, moment reaction | N·mm |
| Torque along the shaft | N·m |
| Stress | MPa |
| Section modulus | mm³ |
| Rotation, misalignment, angular position | rad |
| Stiffness | N/mm |

> **Note — the one unit discontinuity.** Bending moments (`M`, `M_xz`, `M_xy`) are stored in **N·mm**; torque (`T`) is stored in **N·m**, because torque enters the model at the gear-system boundary in N·m and is not renormalised on the way in. This is documented rather than silently corrected, per the platform's *flag, do not silently fix* principle. Report writers and exporters must label the two columns with their own units.

Angles are radians throughout the result containers. Degrees appear only at the user interface, never in stored results.

### Sign convention

- Displacements and deflections are positive along the positive axis of their plane.
- Reactions are the constraint forces recovered from `f = K·d` at constrained degrees of freedom, and therefore oppose the applied load.
- `delta_r` is positive for a closing radial contact; `delta_a` is positive when axial contact is engaged and **negative when clearance has not yet been closed** — a negative `delta_a` is a valid, meaningful state, not an error.
- Contact deflections `delta_j` are floored at zero by the solver: an unloaded rolling element carries `delta_j = 0` and therefore `Q_j = 0`, never a negative value.

---

## Module reference

### `fem_results/shaft_results.py`

The complete output of one rigid-bearing shaft FEM solve.

#### `ShaftResults`

Organised in four groups. The grouping is part of the contract — a consumer that needs raw FEM arrays should not have to sift them out of engineering quantities.

| Group | Contents |
|---|---|
| **1 · Mesh** | Node positions and element connectivity, as produced by the 1D mesh generator. |
| **2 · Raw FEM solution** | Global stiffness matrix, DOF partition, displacement and force vectors per plane, reaction vectors. These are the arrays a downstream solver needs in order *not* to re-run the FEM. |
| **3 · Engineering quantities** | Node-aligned arrays derived from group 2: internal forces, deflections, section properties, stresses. |
| **4 · Bearing node data** | One `BearingNodeData` per bearing, in the same order as the shaft system's bearing list (sorted by axial position). |

Node-aligned arrays in group 3, all of length `n_nodes` and all indexed against `x_nodes`:

| Field | Unit | Quantity |
|---|---|---|
| `x_nodes` | mm | Axial coordinate |
| `M_xz`, `M_xy`, `M` | N·mm | Bending moment per plane and resultant |
| `V_xz`, `V_xy`, `V` | N | Shear force per plane and resultant |
| `v_xz`, `v_xy`, `v` | mm | Deflection per plane and resultant |
| `T` | N·m | Torque diagram |
| `d` | mm | Local diameter |
| `W`, `Wt` | mm³ | Bending and torsional section modulus |
| `sigma_b` | MPa | Bending stress |
| `tau` | MPa | Torsional shear stress |

Governing scalars, each stored with the axial location at which it occurs:

| Field | Location field | Quantity |
|---|---|---|
| `M_max` | `x_M_max` | Maximum resultant bending moment |
| `v_max` | `x_v_max` | Maximum resultant deflection |
| `sigma_b_max` | `x_sigma_b_max` | Maximum bending stress |
| `tau_max` | `x_tau_max` | Maximum torsional shear stress |

There is deliberately **no** `V_max`. Shear is not a governing quantity for the shaft criteria implemented today, and storing a maximum that nothing checks against would invite it to be used as though it were.

Torsion is additionally retained as a per-source breakdown (`torsion_contributions`), giving the contribution of each torque source at each node — the array to consult when a torque diagram does not look as expected.

#### `BearingNodeData`

The complete FEM nodal state at one bearing position. This is the interface across which the shaft solve hands over to the bearing solve; the ISO/TS 16281 solvers read this and never touch the raw FEM arrays.

| Field | Unit | Quantity |
|---|---|---|
| `label`, `position` | — , mm | Bearing identifier and axial location |
| `u` | mm | Axial displacement |
| `v_xz`, `v_xy` | mm | Transverse displacement per plane |
| `theta_xz`, `theta_xy` | rad | Nodal bending rotation per plane |
| `Fr_xz`, `Fr_xy`, `Fr` | N | Radial reaction per plane and resultant |
| `Fa` | N | Axial reaction — non-zero for a locating bearing, zero for a floating one |
| `M_xz`, `M_xy` | N·mm | Moment reaction per plane |
| `psi_xz`, `psi_xy` | rad | Seat misalignment per plane |

**Seat misalignment `psi`** is the quantity that makes the shaft and bearing analyses one coupled calculation rather than two independent ones. It is the shaft centreline slope **across the bearing seat width**: the gradient of transverse displacement between the seat bounds, falling back to the nodal rotation `theta` when the seat width is zero. It is computed during result read-back, with no second FEM pass, and it is the prescribed ring tilt the ISO/TS 16281 solvers impose. A seat of finite width and a knife-edge support give different `psi` for the same shaft — which is the point.

Access is by label, not by index:

```python
node = next(n for n in results.bearing_nodes if n.label == "A")
```

---

### `bearings/load_distribution/single_row/ball_bearing_results.py`

Point-contact internal load distribution, ISO/TS 16281.

#### `BallLoadDistributionResult` — one row

The raw output of one point-contact solve.

| Field | Unit | Quantity |
|---|---|---|
| `delta_r` | mm | Radial ring displacement |
| `delta_a` | mm | Axial ring displacement — negative while clearance is closing |
| `psi` | rad | Prescribed ring misalignment (from `BearingNodeData`) |
| `phi_Fr` | rad | Angle of the resultant radial force in the global frame |
| `delta_j` | mm, `(Z,)` | Contact deflection per rolling element |
| `alpha_j` | rad, `(Z,)` | Contact angle per rolling element — genuinely load-dependent for a ball |
| `Mz` | N·mm | Reaction moment at the converged state |
| `n_iter`, `residual`, `ok` | — , N, — | Convergence diagnostics |

`phi_j` is stored in the **local** frame, with zero aligned to the resultant radial force plane. Post-processing rotates it into the global frame using `phi_Fr`.

#### `BallBearingResult` — one bearing

The unified per-bearing container. `rows` is always a list: length 1 for a single-row bearing, length `i` for a multi-row bearing's rows, index-aligned with the bearing's own row list. `is_multirow` is derived from `len(rows) >= 2` — there is no separate flag that could contradict the list it describes.

| Field | Applies to | Quantity |
|---|---|---|
| `rows` | always | The per-row results |
| `f_r`, `f_a` | multi-row only | Converged radial and axial load fractions per row |
| `outer_n_iter`, `outer_residual`, `outer_ok` | multi-row only | Outer fixed-point iteration diagnostics |
| `delta_r`, `delta_a`, `psi` | always | Read from `rows[0]` — the shared rigid-ring displacement |
| `is_single` | always | Derived from `len(rows) == 1` |

Build through the classmethods rather than the constructor: `.single(row)` here, and `.multirow(rows, f_r, f_a, n_iter, residual, ok)` on the multi-row copy. They name the case at the call site instead of leaving a reader to infer it from which optional arguments were supplied.

The predicate is derived from the list it describes, so there is no separate flag that could contradict it. Note that the copy in the multi-row solver package spells the same idea as `is_multirow` — the negation. Until the two are unified, test the one belonging to the module you imported from, and do not assume the other exists.

`delta_r`, `delta_a` and `psi` are exposed on the bearing as well as on each row. For a multi-row result these are the shared rigid-ring displacement that every row agrees on at convergence; for a single-row result `rows[0]` is simply the only row. Either way, *"what is this bearing's ring displacement"* has one well-defined answer, so callers are not required to write `result.rows[0].delta_r`.

To branch on whether an outer load-split actually ran, test `is_multirow` — not `outer_ok is not None`. The two agree today; only the first is defined to remain correct.

---

### `bearings/load_distribution/single_row/roller_bearing_results.py`

Line-contact internal load distribution, ISO/TS 16281. Mirrors the ball container shape exactly, with the per-lamina extension that line contact requires.

#### `RollerLoadDistributionResult` — one row

Carries the same base fields as its ball counterpart, with these differences of meaning:

| Field | Difference |
|---|---|
| `delta_a` | Always `0.0` — a radial cylindrical roller bearing carries no axial load |
| `alpha_j` | The bearing's nominal contact angle broadcast to `(Z,)`; the contact normal of a cylindrical roller stays radial regardless of tilt |
| `delta_j` | Roller-centreline deflection, **before** the lamina and profile correction — the deflection that governs contact is `delta_jk`, not this |
| `Mz` | A diagnostic reaction moment evaluated at the converged state; it is not a solve constraint |

Line-contact fields, discretising each roller into `n_s` laminae:

| Field | Unit / shape | Quantity |
|---|---|---|
| `x_k` | mm, `(n_s,)` | Lamina positions along the roller |
| `psi_j` | rad, `(Z,)` | Per-roller local misalignment |
| `delta_jk` | mm, `(Z, n_s)` | Per-lamina elastic deflection |
| `q_jk` | N, `(Z, n_s)` | Per-lamina contact force |

#### `RollerBearingResult` — one bearing

The same `rows`-based container as `BallBearingResult`, with the same `.single()` / `.multirow()` construction.

> **Note on what "multi-row" means here.** For a *radial* cylindrical roller bearing, a second row is a capacity-rating multiplier applied to one raceway — solved as a single `delta_r`, with no rows list and no load split. That is the same treatment a multi-row deep-groove ball bearing receives. The `rows` list and `.multirow()` classmethod on this container exist for a future **thrust** roller family, where axially stacked rows genuinely share a load-split compatibility solve. The container shape is prepared ahead of need so it will not have to change when that family is written.

---

## Reading the results

This section is written for the engineer reading a result, not for the developer writing to one.

### Shaft — governing values

Start with the four governing scalars and their locations. Each answers a different question:

| Value | Question it answers | Typically governed by |
|---|---|---|
| `sigma_b_max` @ `x_sigma_b_max` | Is the shaft strong enough in bending? | A section change or a load introduction point |
| `tau_max` @ `x_tau_max` | Is the shaft strong enough in torsion? | The smallest diameter within the torque-carrying span |
| `v_max` @ `x_v_max` | Is the shaft stiff enough for the gears it carries? | Mid-span between bearings |
| `M_max` @ `x_M_max` | Where is the bending demand concentrated? | The dominant transverse load |

A governing location that coincides with a shoulder or a keyway is the signal to look at stress concentration: `sigma_b` and `tau` in `ShaftResults` are **nominal** section stresses. Notch effects are applied afterwards by the shaft post-processor and are not present in these arrays.

### Shaft — diagrams

All group 3 arrays are node-aligned against `x_nodes` and can be plotted directly. When reading them:

- **Plot the resultant and the two components together.** A resultant `M` that looks smooth can hide a sign reversal in `M_xz` or `M_xy`. The reversal, not the resultant, is what matters for a rotating-bending fatigue assessment.
- **A bending-moment diagram should close to zero at both free ends.** If it does not, the load set is not in equilibrium — check the applied loads before questioning the mesh.
- **Steps in `d`, `W` and `Wt` are geometric, not numerical.** A jump in `sigma_b` at a diameter change is the section modulus changing, not a solver artefact.
- **A visibly faceted deflection curve means the mesh is too coarse** for the quantity being read, even where forces and reactions are already converged. Confirm with a mesh convergence study rather than by eye.
- **`T` is a step diagram.** Torque changes only where a torque source acts. A gradient between sources indicates a distributed torque that was not intended.

### Bearing seat state

For each `BearingNodeData`, read in this order:

1. **`Fr` and `Fa`** — the load the bearing actually carries. An `Fa` of exactly zero on a bearing intended to locate axially means the axial constraint was not applied.
2. **`psi_xz`, `psi_xy`** — the misalignment imposed on the bearing. This is the input to the internal load distribution, and small values here have large consequences: misalignment of the order of 10⁻⁴ rad is enough to shift the load markedly onto one side of a rolling element set.
3. **`theta` versus `psi`** — for a seat of finite width these differ. When they are identical, the seat width is zero and the bearing is being treated as a knife-edge support.

Two distinct quantities exist at a bearing node and must not be conflated. **Reactions** (`R_xz`, `R_xy`, `R`, `R_axial` on `ShaftResults`) come from the reaction vector and are non-zero only at constrained degrees of freedom. **Nodal loads** (`Fr_xz`, `Fr_xy`, `Fr`, `Fa` on `BearingNodeData`) come from the total nodal force vector. At a bearing node with no other load applied at exactly that position they are numerically close, but they are computed from different arrays and are not guaranteed identical. Report them separately; do not present one as a check on the other.

### Bearing internal load distribution

- **Contact force per element** follows `Q_j = c_p · delta_j^1.5` for point contact. Elements with `delta_j = 0` are outside the loaded zone and carry nothing.
- **The loaded-zone extent** is the practical read on whether the bearing is correctly preloaded or over-cleared. A very narrow loaded zone under a nominally radial load indicates excessive clearance; a fully loaded ring indicates preload or an axial load large enough to close the clearance everywhere.
- **`delta_a < 0`** means the axial clearance has not closed. The bearing is carrying radial load only, and any axial stiffness read at this state is not yet engaged.
- **For roller bearings, read `q_jk`, not `delta_j`.** The per-lamina force distribution along the roller is where edge loading appears. A `q_jk` profile that rises sharply at the roller ends is edge stress — the condition roller profiling exists to avoid, and the reason ISO/TS 16281 discretises the contact in the first place. A flat profile across the laminae indicates a well-aligned, well-profiled contact.
- **Angular positions are stored in the local frame.** Compare a polar plot against `phi_Fr` before concluding that the loaded zone sits in an unexpected direction.

### Convergence diagnostics

Every solve stores `n_iter`, `residual` and `ok`. A result whose `ok` is `False` is a **recorded failed solve**, not an absent one: the fields are populated with the last iterate. Check `ok` before reading any physical quantity, and report `residual` alongside results in any document that will be reviewed by someone who did not run the analysis. For a multi-row bearing, `outer_ok` and `outer_residual` describe the load-split iteration and are separate from each row's own convergence.

---

## Consumers

| Container | Written by | Read by |
|---|---|---|
| `ShaftResults`, `BearingNodeData` | `solvers/.../shaft/static/results_reader` | `fixtures/studies/shafts/results_library` (`RigidBearingFEMResultsLibrary`), the ISO/TS 16281 ball and roller solvers, `fixtures/studies/text_report` |
| `BallBearingResult`, `BallLoadDistributionResult` | `solvers/.../single_row/iso_16281/ball_bearing/solver` | ball post-processing, `fixtures/studies/bearings/.../results_library`, `fixtures/studies/bearings/.../rolling_bearing_study` |
| `RollerBearingResult`, `RollerLoadDistributionResult` | `solvers/.../single_row/iso_16281/roller_bearing/solver` | roller post-processing, `fixtures/studies/bearings/.../results_library`, `fixtures/studies/bearings/.../rolling_bearing_study` |

Registries that collect these results by label — `RigidBearingFEMResultsLibrary` and `BearingResultsLibrary` — live in `fixtures/studies/`, not here. They are orchestration, and orchestration is not a result shape.

Note what is **absent** from the "read by" column: no result container is read by another result container, and none is read by `core/` or `mesh/`. That is the property that makes this package safe to change in isolation.

---

## Usage

Results are produced by solvers and read by everything downstream. A consumer never constructs one.

```python
from axisforge.results.fem_results.shaft_results import ShaftResults

# Produced by the shaft solve, retrieved from the study's results library
# (RigidBearingFEMResultsLibrary, from fixtures/studies/shafts/results_library)
result: ShaftResults = library.get("input_shaft")

# Governing values
print(f"sigma_b_max = {result.sigma_b_max:.1f} MPa @ x = {result.x_sigma_b_max:.1f} mm")

# Diagram
import matplotlib.pyplot as plt
plt.plot(result.x_nodes, result.M)          # N.mm against mm

# Bearing seat state — the input to the internal load distribution
node = next(n for n in result.bearing_nodes if n.label == "A")
print(f"Fr = {node.Fr:.1f} N, psi_xy = {node.psi_xy:.6f} rad")
```

```python
# Bearing internal load distribution
# (BearingResultsLibrary, from fixtures/studies/bearings/.../results_library)
bearing_result = bearing_library.get("A").load_distribution   # BallBearingResult

if not all(row.ok for row in bearing_result.rows):
    raise RuntimeError("bearing load distribution did not converge")

row = bearing_result.rows[0]                        # row 0 / the only row
loaded = (row.delta_j > 0.0).sum()
print(f"delta_r = {bearing_result.delta_r:.5f} mm, "
      f"{loaded} of {row.delta_j.size} elements loaded")
```

---

## Design contracts

- **No computation.** A result container performs no numerical work. Derived quantities that require a formula belong in the post-processing module of the solver package that owns the formula.
- **No internal dependencies.** Modules in `results/` import NumPy and the standard library only. They import nothing from `core/`, `mesh/`, `solvers/`, `fixtures/` or `ui/`, and they import nothing from each other.
- **No hidden state.** Every quantity a solve produced is a public attribute, already evaluated. There are no private caches and no lazy properties that trigger work.
- **Write once, read many.** A container is populated by exactly one producer and treated as read-only thereafter. Consumers do not mutate results; a modified result is a new result.
- **Units are fixed and documented per field.** A field's unit is part of the contract and is never converted in place. Where two related fields carry different units, the discrepancy is documented, not normalised.
- **Array alignment is part of the contract.** Node-aligned arrays share the index of `x_nodes`; per-row lists share the index of the bearing's own row list; per-element arrays share the index of the rolling-element set. A consumer may rely on this without checking.
- **A failed solve is still a result.** Convergence metadata is stored, never raised away. It is the consumer's responsibility to check `ok` before use.
- **One file per result family.** Each contact model, each analysis type and each element family gets its own module. Ball and roller results are never merged into a shared base class.
- **Structural compatibility over inheritance.** Where two containers share a shape, they satisfy the same protocol structurally rather than inheriting from a common base. Self-containment is preferred to a shared parent.

---

## Extending this package

To add a result type for a new analysis:

1. **Place it by the analysis, not by the solver.** The path under `results/` should mirror the path under `solvers/` that produces it.
2. **Define the fields with units in the class docstring.** The docstring is the contract; report writers and exporters are written against it and will quote it.
3. **State the array shapes and what each index means.** `(Z,)`, `(Z, n_s)` and `(n_nodes,)` are not interchangeable, and a consumer must not have to infer alignment.
4. **Carry the convergence metadata** — `n_iter`, `residual`, `ok` — for any container produced by an iterative solve.
5. **Add no imports from `axisforge`.** If a result needs a domain object to be interpretable, store the identifying label instead of the object.
6. **Use a `rows` list from the start** if the element can plausibly exist in a multi-row form, even when only the single-row case is implemented. Changing a container's shape later is a breaking change across every consumer.
7. **Register the module** in the package's public surface and add it to the [Status](#status) table above.

---

## Standards referenced

| Standard | Applies to |
|---|---|
| ISO/TS 16281 | Rolling bearing internal load distribution — contact deflection, lamina discretisation, dynamic equivalent load |
| ISO 281 | Rolling bearing dynamic load rating and basic rating life |
| ISO 6336 | Cylindrical gear load capacity — reserved for the planned gear result containers |
| ISO 21771 | Cylindrical gear geometry and nomenclature |

The containers in this package store the quantities these standards define. The formulae themselves belong to the solvers and to the machine elements, never to a result.