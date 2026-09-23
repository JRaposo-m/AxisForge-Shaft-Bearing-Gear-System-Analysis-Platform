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
  - [`bearings/load_distribution/load_distribution_results.py`](#bearingsload_distributionload_distribution_resultspy)
  - [`bearings/life/basic_life_results.py`](#bearingslifebasic_life_resultspy)
  - [`bearings/bearing_analysis_result.py`](#bearingsbearing_analysis_resultpy)
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
| `fem_results/` | `submodel_results.py` | `SubmodelResult` | Implemented |
| `convergence_results/` | `convergence_results.py` | `ConvergenceRecord`, `MeshRefinementResult` | Implemented |
| `bearings/load_distribution/` | `load_distribution_results.py` | `LoadDistributionResult` (ABC), `BallLoadDistributionResult`, `LineContactLoadDistributionResult` (ABC), `RollerLoadDistributionResult`, `BearingResult` (ABC), `BallBearingResult`, `RollerBearingResult` | Implemented |
| `bearings/life/` | `basic_life_results.py` | `BasicReferenceRatingLifeResult` | Implemented (ISO/TS 16281 basic reference life only) |
| `bearings/` | `bearing_analysis_result.py` | `BearingAnalysisResult` | Implemented — the only module that imports both `load_distribution/` and `life/` |
| `gears/` | gear rating results (ISO 6336) | — | Planned |
| `fatigue/` | fatigue and static failure results | — | Planned |
| `lubrication/` | film thickness and viscosity ratio results | — | Planned |

---

## Position in the architecture

The wiring law permits `solvers → results` and forbids the reverse. `results/` imports nothing from `axisforge` at runtime — only NumPy and the standard library. `TYPE_CHECKING`-only imports from `solvers/` exist purely for annotations (e.g. `ContactBearingStiffness`, the row-life classes) and are erased at runtime.

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
- **Registries do not live here.** A result *shape* is a container; a *library* keyed by shaft or bearing label is an orchestration concern and lives in `fixtures/studies/`. `results/` defines what one result is; it does not define how a run collects many of them.

---

## Structure

```
results/
├── _base.py                                     shared conventions (DC, is_abstract, check_*) — internal, not reexported
├── fem_results/
│   ├── shaft_results.py                         ShaftResults, BearingNodeData
│   └── submodel_results.py                      SubmodelResult
├── convergence_results/
│   └── convergence_results.py                   ConvergenceRecord, MeshRefinementResult
└── bearings/
    ├── load_distribution/
    │   └── load_distribution_results.py         LoadDistributionResult / BallLoadDistributionResult /
    │                                             LineContactLoadDistributionResult / RollerLoadDistributionResult
    │                                             BearingResult / BallBearingResult / RollerBearingResult
    ├── life/
    │   └── basic_life_results.py                BasicReferenceRatingLifeResult
    └── bearing_analysis_result.py                BearingAnalysisResult
```

One bearing produces **one** `BearingAnalysisResult`, whether it has one row or several — `n_rows >= 1` on `BearingResult`, no separate single-row/multi-row class. Contact type (ball / roller) is the only axis with concrete classes, because point contact and line contact genuinely produce different fields (a roller row carries `q_jk`, `x_k`, `psi_j`; a ball row does not).

---

## Import surface

| Import from | Names |
|---|---|
| `axisforge.results.fem_results.shaft_results` | `ShaftResults`, `BearingNodeData` |
| `axisforge.results.fem_results.submodel_results` | `SubmodelResult` |
| `axisforge.results.convergence_results.convergence_results` | `ConvergenceRecord`, `MeshRefinementResult` |
| `axisforge.results.bearings.load_distribution.load_distribution_results` | `LoadDistributionResult`, `BallLoadDistributionResult`, `LineContactLoadDistributionResult`, `RollerLoadDistributionResult`, `BearingResult`, `BallBearingResult`, `RollerBearingResult` |
| `axisforge.results.bearings.life.basic_life_results` | `BasicReferenceRatingLifeResult` |
| `axisforge.results.bearings.bearing_analysis_result` | `BearingAnalysisResult` |

Everything above is also re-exported eager from `axisforge.results` (see the package's own `__init__.py`) and from `axisforge` at the top level. Ball and roller row/bearing classes are imported explicitly and are never flattened into an undifferentiated namespace: point contact and line contact produce genuinely different quantities — a ball row has no per-lamina array, a roller row has no varying contact angle.

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
- `delta_r` is positive for a closing radial contact; `delta_a` is positive when axial contact is engaged and **negative when clearance has not yet been closed** — a negative `delta_a` is a valid, meaningful state, not an error. For a `RollerBearingResult` (radial-only), `delta_a` is always `0.0` by definition, not "0 by solve".
- Contact deflections `delta_j` are floored at zero by the solver: an unloaded rolling element carries `delta_j = 0` and therefore `Q_j = 0`, never a negative value.
- `Fa`/`f_a` on `BearingResult` can be `None`-shaped in effect for a preloaded arrangement: when the *total* axial load is ~0 (e.g. a back-to-back preloaded pair), `f_a` returns `None` rather than a fabricated equal split — read `row.Fa_row` directly in that case (see [Module reference](#bearingresult--one-bearing) below).

---

## Module reference

### `fem_results/shaft_results.py`

The complete output of one `RigidSupportFEMSolver` shaft solve (bending + axial) plus one `TorsionSolver` pass, combined by `ShaftResultsReader`.

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
| `u` | mm | Axial displacement, node-aligned like `v_xz`/`v_xy` |
| `theta_xz`, `theta_xy` | rad | Bending rotation per plane, node-aligned like `v_xz`/`v_xy` |
| `T` | N·m | Torque diagram — produced by the separate `TorsionSolver` (see [`solvers/README.md`](../solvers/README.md#torsion)), not by the bending/axial FEM solve itself |
| `phi` | rad | Twist angle, same values as `phi_total` below, referenced to `phi = 0` at the first node |
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
| `phi_max` | `x_phi_max` | Maximum absolute twist angle |

There is deliberately **no** `V_max`. Shear is not a governing quantity for the shaft criteria implemented today, and storing a maximum that nothing checks against would invite it to be used as though it were.

Torsion is additionally retained as a per-source breakdown (`torsion_contributions`), giving the contribution of each torque source at each node — the array to consult when a torque diagram does not look as expected. `torsion_contributions` and the torque/twist arrays (`T`, `T_total`, `tau_total`, `phi_total`) are populated from `TorsionSolver.solve()` (`solvers/machine_elements/shaft/static_solvers/torsion.py`), called independently of the bending/axial FEM solve and read back into `ShaftResults` by `ShaftResultsReader` alongside everything else.

`ShaftResults` also carries a `name: str` field and a raw `K: np.ndarray` (the last-assembled global stiffness matrix), both alongside the raw-solution group described above.

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

#### `fem_results/submodel_results.py`

`SubmodelResult` — the result of a submodel (cut-out) FEM re-solve driven by boundary values extracted from a full-shaft solve. See the module docstring for the exact field list; this container follows the same conventions as `ShaftResults`.

---

### `bearings/load_distribution/load_distribution_results.py`

ISO/TS 16281 internal load distribution. Every equation lives in `solvers/` (`contact_solver.py`, `contact_postprocessing.py`); this module stores only shapes.

**Two levels.**

| Level | Class | Carries |
|---|---|---|
| Bearing | `BearingResult` (ABC) | Shared state of the solve (`delta_r`, `delta_a`, `psi`, `phi_Fr`), the applied load at the FEM node (`Fr_xz`, `Fr_xy`, `Fa`), convergence of *the* solve (`n_iter`, `residual`, `ok`), bearing-level `stiffness` |
| Row | `LoadDistributionResult` (ABC) | Only what differs between rows: element kinematics, contact forces, row reactions |

**Why shared state lives on the bearing, not on each row.** The multi-row solver is co-located / shared-displacement (`contact_solver.MultiRowSolverBase`): every row sees the identical `delta_r`, `delta_a`, `psi` by construction, and there is *one* root-find, so *one* `n_iter`/`residual`/`ok`. Duplicating that state onto every row would let two rows of the same bearing silently disagree, which cannot happen in the current formulation. If axial row offset is modelled later, per-row *local* displacements would be added to the row as additional optional fields; the bearing level stays unchanged.

#### `LoadDistributionResult` — one row

| Field | Unit / shape | Quantity |
|---|---|---|
| `phi_j` | rad, `(Z,)` | Element angular position, **local** frame — zero aligned to `phi_Fr` |
| `delta_j` | mm, `(Z,)` | Element deflection (roller: raw approach, eq. 38 — may be negative before flooring) |
| `alpha_j` | rad, `(Z,)` | Operating contact angle |
| `Q_j` | N, `(Z,)`, `>= 0` | Element contact force — kept explicitly, not just derivable, because it is what plots and reports read directly |
| `Fr_row` | N | Row reaction along `phi_Fr` |
| `Fa_row` | N | Row axial reaction, signed — `0.0` for every row of a radial roller bearing |
| `Mz` | N·mm | Row moment reaction |

`Z` and `n_loaded` (elements with `Q_j > 0`) are derived properties, not stored fields.

#### `BallLoadDistributionResult` — point contact

Adds nothing to the base row; `Q_j = c_p · delta_j^1.5`, computed by the solver.

#### `LineContactLoadDistributionResult` (ABC) / `RollerLoadDistributionResult` — line contact

Adds the lamina discretisation:

| Field | Unit / shape | Quantity |
|---|---|---|
| `x_k` | mm, `(n_s,)` | Lamina mid-positions along the roller |
| `psi_j` | rad, `(Z,)` | Per-roller tilt, eq. 39 |
| `delta_jk` | mm, `(Z, n_s)` | Per-lamina deflection, eq. 41 |
| `q_jk` | N, `(Z, n_s)` | Per-lamina contact force, eq. 36 |

Checked invariant: `Q_j == q_jk.sum(axis=1)` — stored redundantly (both `Q_j` and `q_jk`) for a uniform row API across contact types, but validated at construction so the two can never silently disagree.

`RollerLoadDistributionResult` is the **radial** cylindrical roller row: `delta_a` is not a row field (it lives on the bearing and is forced to `0.0` — see below), and `alpha_j` is the bearing's nominal contact angle broadcast to `(Z,)`, since a cylindrical roller's contact normal stays radial regardless of tilt.

#### `BearingResult` (ABC) — one bearing

| Field | Quantity |
|---|---|
| `label`, `rows` | Identifier; `rows` is a non-empty tuple, length 1 for single-row, length *i* for multi-row — **one class covers both row counts**, there is no separate single-row/multi-row class |
| `delta_r`, `delta_a`, `psi`, `phi_Fr` | Shared solve state, see above |
| `Fr_xz`, `Fr_xy`, `Fa` | Applied load at the FEM bearing node |
| `n_iter`, `residual`, `ok` | Convergence of the one root-find (`n_iter` is presently `nfev` from the root solver — number of function evaluations, not iteration count; flagged for a possible rename) |
| `stiffness` | `ContactBearingStiffness \| None` — bearing-level postprocessing, attached via `with_stiffness()` after `solve()`, never mutated in place |

Construct through the classmethods, never the bare constructor at a call site: `.single(row, **shared)` for one row, `.multirow(rows, **shared)` for `>= 2` rows — they name the case being built instead of leaving a reader to infer it from which arguments were supplied.

| Property | Returns |
|---|---|
| `n_rows`, `is_single`, `is_line_contact` | Structure queries |
| `.row` | The row of a single-row bearing; raises `AttributeError` on multi-row — use `.rows[i]` there |
| `Fr` | `hypot(Fr_xz, Fr_xy)` |
| `Mz` | Sum of `row.Mz` across rows |
| `f_r`, `f_a` | Per-row share of `Fr`/`Fa`. **`None` when the corresponding total load is ~0** — a share of nothing is undefined. This matters for a preloaded back-to-back pair: total `Fa` can be `0` while each row carries `+Fa_row`/`-Fa_row` internally; read `rows[i].Fa_row` directly in that case rather than trusting a fabricated 50/50 split. |
| `equilibrium_error` | `(Fr - sum(Fr_row), Fa - sum(Fa_row))` |
| `phi_j_global(i)` | Row `i` element positions rotated into the global frame, wrapped to `[0, 2π)` |

#### `BallBearingResult` / `RollerBearingResult` — the two concrete bearings

The only axis with concrete classes is contact type. `RollerBearingResult` additionally enforces, at construction, that it is **radial-only**: `Fa == 0.0`, `delta_a == 0.0`, and every `row.Fa_row == 0.0` — not "zero by solve", zero by definition, because a radial cylindrical roller bearing (NU/N) has no axial capacity.

> **Open item, flagged in the source, not yet resolved.** A future **thrust** roller family is the mirror case — axial load only, no `Fr`/`delta_r` — and needs its own treatment (a `duty` axis or a separate result class), not a reuse of `RollerBearingResult`. The corresponding note also still needs writing explicitly for the ball single-row thrust case (`bearing.duty == "thrust"`) in `contact_solver.py`; today it is documented in `contact_postprocessing.py`/`contact_solver.py` module docstrings but the equivalent single-row-thrust note has not yet been added at the call site.
>
> **Open item.** Bearing-level stiffness (`ContactBearingStiffness`, computed by `contact_postprocessing.bearing_stiffness()`) can evaluate to `float('inf')` when the projected displacement is ~0. This is acknowledged as wrong (a physically meaningful stiffness cannot be infinite) but deliberately left unresolved for now.

---

### `bearings/life/basic_life_results.py`

Basic **reference** rating life, ISO/TS 16281:2008 — a bearing-level container that wraps, rather than duplicates, the per-row life objects computed in `contact_postprocessing.py`.

#### `BasicReferenceRatingLifeResult`

| Field | Quantity |
|---|---|
| `label`, `rows` | Bearing identifier; `rows` is a non-empty tuple of `BallBasicReferenceRatingLife` or `RollerBasicReferenceRatingLife` (your existing classes from `contact_postprocessing.py`, stored as-is — no parallel copy) |
| `L10r` | Combined bearing basic reference life `[10^6 rev]` — equal to `rows[0].L10r` for a single row, or the Zaretsky combination (eq. 49a, `<= min(rows).L10r`) for multiple rows |
| `pref` | The `DynamicEquivalentReferenceLoadBase` instance (Ball or Roller variant) that produced `L10r` |

`Pref_r`/`Pref_a` are properties delegating to `self.pref`, giving ball and roller a uniform read surface. Every equation stays in `contact_postprocessing.py`; this container only combines and validates.

> **Planned sibling in this file:** ISO 281 basic rating life `L10` (catalogue method, `P = X·Fr + Y·Fa`) — not yet started.

---

### `bearings/bearing_analysis_result.py`

#### `BearingAnalysisResult`

The single per-bearing aggregate, and the **only** module in `results/` that imports from both `load_distribution/` and `life/` — neither of those imports the other.

| Field | Quantity |
|---|---|
| `label` | Bearing identifier, checked to match across `load_distribution` and `basic_life` |
| `load_distribution` | `BearingResult` — always present |
| `basic_life` | `BasicReferenceRatingLifeResult \| None` — `None` when the solver was run with `postprocess=False` |

This is what `solve()` in `contact_solver.py` returns, one per bearing label: `dict[label, BearingAnalysisResult]`. A future `modified_life` field (ISO/TS 16281 `a_ISO`) has an already-reserved, commented slot in the source.

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
- **The loaded-zone extent** (`row.n_loaded`) is the practical read on whether the bearing is correctly preloaded or over-cleared. A very narrow loaded zone under a nominally radial load indicates excessive clearance; a fully loaded ring indicates preload or an axial load large enough to close the clearance everywhere.
- **`delta_a < 0`** means the axial clearance has not closed. The bearing is carrying radial load only, and any axial stiffness read at this state is not yet engaged.
- **For roller bearings, read `q_jk`, not `delta_j`.** The per-lamina force distribution along the roller is where edge loading appears. A `q_jk` profile that rises sharply at the roller ends is edge stress — the condition roller profiling exists to avoid, and the reason ISO/TS 16281 discretises the contact in the first place. A flat profile across the laminae indicates a well-aligned, well-profiled contact.
- **Angular positions are stored in the local frame.** Use `bearing_result.phi_j_global(i)` before concluding that the loaded zone sits in an unexpected direction.
- **A preloaded, nominally-radial-load-only pair.** If `bearing_result.Fa` is ~0 but you need to know how much axial load *row 0* is actually carrying, read `bearing_result.rows[0].Fa_row` directly — `f_a` returns `None` in this case by design, not a misleading `0.5`.

### Convergence diagnostics

Every solve stores `n_iter`, `residual` and `ok`. A result whose `ok` is `False` is a **recorded failed solve**, not an absent one: the fields are populated with the last iterate. Check `ok` before reading any physical quantity, and report `residual` alongside results in any document that will be reviewed by someone who did not run the analysis.

---

## Consumers

| Container | Written by | Read by |
|---|---|---|
| `ShaftResults`, `BearingNodeData`, `SubmodelResult` | `solvers/.../shaft/static_solvers/results_reader.py` | Fixture-layer results libraries, the ISO/TS 16281 ball and roller solvers, report writers |
| `ConvergenceRecord`, `MeshRefinementResult` | `solvers/mesh/convergence_solver.py` | The mesh convergence study report, fixture-layer convergence studies |
| `LoadDistributionResult` and subclasses, `BearingResult` and subclasses | `solvers/machine_elements/bearings/load_distribution/iso_16281/contact_solver.py` | `contact_postprocessing.py` (stiffness, life), fixture-layer bearing studies and report writers |
| `BasicReferenceRatingLifeResult` | `contact_solver.py` (via `contact_postprocessing.py`'s life classes) | fixture-layer bearing studies, report writers |
| `BearingAnalysisResult` | `contact_solver.py`'s `solve()` — one per bearing label | Everything downstream that needs both load distribution and life for one bearing |

Registries that collect these results by label live in `fixtures/studies/`, not here. They are orchestration, and orchestration is not a result shape.

Note what is **absent** from the "read by" column: no result container is read by another result container except through `BearingAnalysisResult` (which exists precisely to be that one exception, explicitly), and none is read by `core/` or `mesh/`. That is the property that makes this package safe to change in isolation.

---

## Usage

Results are produced by solvers and read by everything downstream. A consumer never constructs one directly — except through the documented `.single()`/`.multirow()` classmethods, which a solver uses to build a `BearingResult`.

```python
from axisforge.results.fem_results.shaft_results import ShaftResults

# Produced by the shaft solve, retrieved from the study's results library
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
# Bearing internal load distribution + life, from one solve() call
analyses: dict[str, "BearingAnalysisResult"] = solver.solve(shaft_system, postprocess=True)

analysis = analyses["A"]
bearing = analysis.load_distribution                # BallBearingResult / RollerBearingResult

if not bearing.ok:
    raise RuntimeError("bearing load distribution did not converge")

row = bearing.rows[0]                                # row 0 / the only row
loaded = row.n_loaded
print(f"delta_r = {bearing.delta_r:.5f} mm, {loaded} of {row.Z} elements loaded")

if analysis.basic_life is not None:
    print(f"L10r = {analysis.basic_life.L10r:.2f} x1e6 rev")
```

---

## Design contracts

- **No computation.** A result container performs no numerical work. Derived quantities that require a formula belong in the post-processing module of the solver package that owns the formula.
- **No internal dependencies.** Modules in `results/` import NumPy and the standard library only, at runtime. `TYPE_CHECKING`-only imports from `solvers/` are permitted for annotations and are erased at runtime. They import nothing from `core/`, `mesh/`, `fixtures/` or `ui/` at all, and — with the single, explicit exception of `bearing_analysis_result.py` — nothing from each other.
- **No hidden state.** Every quantity a solve produced is a public attribute, already evaluated. There are no private caches and no lazy properties that trigger work.
- **Write once, read many; frozen and constructed through named classmethods.** Every dataclass in this package is `frozen=True, kw_only=True`. A container is populated by exactly one producer and treated as read-only thereafter. Consumers do not mutate results; `with_stiffness()` and similar "updates" return a new object via `dataclasses.replace()`. Where a shape has more than one valid construction (single-row vs. multi-row), it is built through named classmethods (`.single()`, `.multirow()`) rather than the bare constructor, so the call site states which case it is building.
- **Units are fixed and documented per field.** A field's unit is part of the contract and is never converted in place. Where two related fields carry different units, the discrepancy is documented, not normalised.
- **Array alignment is part of the contract.** Node-aligned arrays share the index of `x_nodes`; per-row tuples share the index of the bearing's own row list; per-element arrays share the index of the rolling-element set. A consumer may rely on this without checking.
- **A failed solve is still a result.** Convergence metadata is stored, never raised away. It is the consumer's responsibility to check `ok` before use.
- **One row shape per contact model, sharing structure through an ABC, not a flat namespace.** Ball and roller rows/bearings are never flattened into an undifferentiated namespace at the import surface — but they *do* share a common abstract base (`LoadDistributionResult`, `BearingResult`) where the shape genuinely is the same, so that shared validation and shared properties (`Z`, `n_loaded`, `f_r`, `f_a`, `phi_j_global`, …) are written once. This replaces an earlier, stricter contract ("ball and roller results are never merged into a shared base class") once the actual row/bearing shapes were confronted directly: the two contact types share far more structure than they differ in, and the previous rule would have meant re-deriving the same validation twice.
- **Flag, do not silently fix.** The `float('inf')` bearing stiffness at zero projected displacement, and the thrust-vs-radial treatment gaps, are marked in the source and named in this document rather than quietly corrected.

---

## Extending this package

To add a result type for a new analysis:

1. **Place it by the analysis, not by the solver.** The path under `results/` should mirror the path under `solvers/` that produces it.
2. **Define the fields with units in the class docstring.** The docstring is the contract; report writers and exporters are written against it and will quote it.
3. **State the array shapes and what each index means.** `(Z,)`, `(Z, n_s)` and `(n_nodes,)` are not interchangeable, and a consumer must not have to infer alignment.
4. **Carry the convergence metadata** — `n_iter`, `residual`, `ok` — for any container produced by an iterative solve.
5. **Add no runtime imports from `axisforge`.** `TYPE_CHECKING`-only imports are fine for annotations. If a result needs a domain object to be interpretable at runtime, store the identifying label instead of the object.
6. **Use a `rows: tuple[...]` field from the start** if the element can plausibly exist in a multi-row form, even when only the single-row case is implemented, and provide `.single()`/`.multirow()` classmethods rather than requiring callers to build the tuple by hand. Changing a container's shape later is a breaking change across every consumer.
7. **Share structure through an ABC when two variants genuinely have the same shape**, rather than duplicating fields and validation across sibling modules. Give each concrete leaf its own class regardless — the import surface should still say plainly which contact model or analysis produced a given object.
8. **Register the module** in the package's public surface (`results/__init__.py`) and add it to the [Status](#status) table above.

---

## Standards referenced

| Standard | Applies to |
|---|---|
| ISO/TS 16281 | Rolling bearing internal load distribution — contact deflection, lamina discretisation, dynamic equivalent load, basic reference rating life |
| ISO 281 | Rolling bearing dynamic load rating and basic rating life (catalogue method — planned, see `bearings/life/`) |
| ISO 6336 | Cylindrical gear load capacity — reserved for the planned gear result containers |
| ISO 21771 | Cylindrical gear geometry and nomenclature |

The containers in this package store the quantities these standards define. The formulae themselves belong to the solvers and to the machine elements, never to a result.
