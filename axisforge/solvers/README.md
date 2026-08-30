# axisforge/solvers — Module Reference

Solvers consume the domain objects documented in [`core/README.md`](../core/README.md) (`Shaft`, `ShaftSystem`, `Bearing`, `SpurHelicalGear`, ...) and the mesh objects in [`mesh/README.md`](../mesh/README.md), and publish results as explicit, GUI-independent containers. No solver imports another solver's internals — only the shared result containers.

← back to [project root](../../README.md)

---

## Table of Contents

- [Package layout](#package-layout)
- [solvers — Shaft FEM](#solvers--shaft-fem)
- [solvers — Shaft post-processing and utilities](#solvers--shaft-post-processing-and-utilities)
- [solvers — Bearings (ISO/TS 16281)](#solvers--bearings-isots-16281)
- [solvers — Bearing life (ISO 281)](#solvers--bearing-life-iso-281)
- [solvers — Gears](#solvers--gears)
- [solvers — Mesh convergence](#solvers--mesh-convergence)
- [solvers — Lubrication (placeholder)](#solvers--lubrication-placeholder)

---

## Package layout

```
solvers/
├── machine_elements/
│   ├── shaft/
│   │   ├── oneD_analysis/
│   │   │   ├── build_stiffness_matrix.py     ← StiffnessMatrixBuilder
│   │   │   ├── FEM_solvers/                  ← SimpleFEMSolver, SubmodelSolver, SubmodelResult
│   │   │   └── static/
│   │   │       ├── static_analysis.py        ← BearingNodeData, ShaftResults,
│   │   │       │                                SimpleFEMResultsLibrary, ShaftResultsReader,
│   │   │       │                                StaticFailure
│   │   │       └── shaft_post_processor.py   ← StressConcentration, PostProcessedResults,
│   │   │                                        ShaftPostProcessor
│   │   └── utils.py                          ← Marin factors, Peterson Kt, Neuber q, Kf
│   ├── bearings/
│   │   ├── ISO_16281/
│   │   │   ├── dispatch.py                   ← register_contact_solver, resolve_solver_cls,
│   │   │   │                                    resolve_solver_cls_for_attrs, SolverDispatchError
│   │   │   ├── library.py                    ← generic utilities + BearingResultsLibrary
│   │   │   ├── rolling_bearing_solver.py     ← RollingBearingSolver (the single entry point)
│   │   │   ├── Ball_Bearing/                 ← single_row_solver, multirow_solver,
│   │   │   │                                    ball_bearing_multirow_solver (reference),
│   │   │   │                                    results, postprocessing
│   │   │   └── Roller_Bearing/               ← single_row_solver, multirow_solver,
│   │   │                                        results, postprocessing
│   │   └── life.py                           ← BearingLifeSolver (ISO 281 L10)
│   └── gears/
│       ├── geometry.py                       ← GearSolver
│       ├── utils.py                          ← involute, solve_alpha_tw, contact ratios, ...
│       └── SpurHelicalGears/LoadCapacity_solver/load_capacity.py   ← empty (ISO 6336, planned)
├── mesh/
│   └── mesh_convergence_study.py             ← ConvergenceRecord, MeshRefinementResult,
│                                                RichardsonGCI, MeshConvergenceStudy
└── lubrification/                            ← Phase 4+, intentionally empty
```

Every package declares its surface in `__init__.py` (lazy roll-up — see the [root README](../../README.md#package-surface--the-__init__py-roll-up-system)).

- `ISO_16281/__init__.py` → `RollingBearingSolver`, `BearingResultsLibrary`, `warn_if_floating_loaded`, `resolve_solver_cls`, `resolve_solver_cls_for_attrs`, `SolverDispatchError`. `Ball_Bearing/` and `Roller_Bearing/` are **not** flattened into it, and `register_contact_solver` is not re-exported.
- `Ball_Bearing/__init__.py` → `ISO16281BallSolver`, `debug_radial_capacity`, `ISO16281MultiRowBallSolverSharedDisplacement`, `BallLoadDistributionResult`, `BallBearingResult`, `BallLoadDistributionLibrary`, `Q_j`, `phi_j_global`, `contact_distribution`, `bearing_stiffness`, `BallBearingStiffness`, `DynamicEquivalentRollingElementLoad`.
- `Roller_Bearing/__init__.py` → the same set plus `lamina_distribution` and `stress_riser_factor`, minus the multi-row solver.
- `FEM_solvers/__init__.py` → `SimpleFEMSolver`, `SubmodelSolver`, `SubmodelResult`. `static/__init__.py` → `BearingNodeData`, `ShaftResults`, `SimpleFEMResultsLibrary`, `ShaftResultsReader`.
- `solvers/mesh/__init__.py` → `ConvergenceRecord`, `MeshRefinementResult`, `RichardsonGCI`, `MeshConvergenceStudy`.
- **`solvers/machine_elements/gears/__init__.py` is still on the old eager form** (`from solvers.gears.geometry import GearSolver`) — a stale import path that will fail; convert it to the lazy pattern.

---

## solvers — Shaft FEM

**`oneD_analysis/build_stiffness_matrix.py` — `StiffnessMatrixBuilder`**
Assembles the global stiffness matrix from element contributions. Currently wires the Timoshenko beam theory; extensible via the `_BEAM_THEORIES` registry.
- `build_stiffness_matrix(mesh, elements)` — global K (3 DOF/node: u, v, θ).

**`oneD_analysis/FEM_solvers/simple_fem_solver.py` — `SimpleFEMSolver`**
Orchestrates the full pipeline: `Mesh1D` → `Elem.from_mesh` → `StiffnessMatrixBuilder` → boundary conditions → two independent planar solves (XZ, XY) sharing the same K → superposition → torsion diagram on the same nodes. Every intermediate quantity is stored as a public attribute (numerical transparency).
- `solve(shaft_system, extra_mandatory=None)` — runs the full solve; stores `x_nodes`, `elements`, `free_dofs`, `constrained_dofs`, displacement vectors `d_total_xz`/`d_total_xy`, external force vectors `f_xz_ext`/`f_xy_ext`, per-load-case `d_contributions`, torsion arrays `T_total`/`tau_total` and `torsion_contributions`.
- `return_values(x_nodes, [x_lo, x_hi])` — nodal solution quantities within an interval (for submodelling).
- `validate_torsion_equilibrium(shaft_system, tol=1e-6)` — returns the residual as a list of error strings.
- Distributed loads are integrated with Gauss quadrature whose order is chosen adaptively (`_assemble_distributed_load_vector` → `TimoshenkoBeam.gauss_order`).
- Options: `theory="timoshenko"`, `constraint_bearing="rigid"`, `distribute_gear_labels` (which gear mesh loads are treated as distributed over face width).
- One solver instance per shaft — a second `solve()` overwrites.

**`oneD_analysis/FEM_solvers/submodel_solver.py` — `SubmodelSolver`, `SubmodelResult`**
Wraps `SimpleFEMSolver` and restricts metric evaluation to a subdomain `[x_lo, x_hi]`, with Lagrange-multiplier boundary injection at the cut nodes. Used exclusively by the convergence study (now in [`solvers/mesh/`](#solvers--mesh-convergence)). `SubmodelResult` carries `x_lo`, `x_hi`, `grade`, `x_nodes`, the subdomain displacement vectors `d_xz`/`d_xy` and the cut-node reaction multipliers `lam_xz`/`lam_xy`.
- `solve(global_solver, shaft_system, grade)` — refines the subdomain with `Grader`, rebuilds the local stiffness and load vectors (distributed loads clamped to the subdomain), and injects the global solution at the cut nodes as prescribed values.

**`oneD_analysis/static/static_analysis.py`**

| Class | Purpose |
|-------|---------|
| `BearingNodeData` | Complete FEM nodal state at a bearing position (displacements, reactions, seat misalignment ψ). |
| `ShaftResults` | Full FEM solution + post-processed engineering quantities for one shaft. |
| `SimpleFEMResultsLibrary` | Registry of `ShaftResults` keyed by shaft name — the canonical source all downstream solvers read from. |
| `ShaftResultsReader` | Post-processes a solved `SimpleFEMSolver` into a `ShaftResults` and stores it in the library. |
| `StaticFailure` | Static failure criteria per material/criterion — stub, not implemented. |

`BearingNodeData` carries `label`, `position`, `u`, `v_xz`, `v_xy`, `theta_xz`, `theta_xy`, `Fr_xz`, `Fr_xy`, `Fr`, `Fa`, `M_xz`, `M_xy`, and the seat-slope misalignment `psi_xz`/`psi_xy` (gradient of v across the seat, or nodal θ for zero-width seats). `ShaftResults` is organised into mesh (`x_nodes`, `elements`), raw FEM solution (`K`, `free_dofs`, `constrained_dofs`, `d_total_*`, `f_*`), torsion (`T_total`, `tau_total`, `torsion_contributions`), post-processed engineering quantities (`x`, `M_xz`, `M_xy`, `M`, `V_xz`, `V_xy`, `V`, `v_xz`, `v_xy`, `v`, `T`, `d`, `W`) and per-bearing node data. The library provides `store` / `get` / `get_or_none` / `remove` / `clear` / `names` / `iter` / `all_results`. `ShaftResultsReader.read(library)` recovers internal forces, deflections, section properties, bearing reactions and node data in one pass and stores the result under `shaft_system.name`.

---

## solvers — Shaft post-processing and utilities

**`oneD_analysis/static/shaft_post_processor.py`** *(new)* — enriches a `ShaftResults` with stress-concentration factors at shoulders and keyways, producing corrected stress arrays ready for the fatigue/failure solvers. No FEM, no equilibrium: it consumes an already-solved `ShaftResults`.

| Class | Purpose |
|---|---|
| `StressConcentration` | One feature: `x`, `feature` (`"shoulder"` / `"keyway"` / `"groove"` / `"press_fit"`), `Kt_bending`, `Kt_torsion`, `Kf_bending`, `Kf_torsion`, `q_bending`, `q_torsion`, `r`, `D`, `d`, `note`. |
| `PostProcessedResults` | `ShaftResults` (by composition, not inheritance) + `Kf_b`, `Kf_t`, `sigma_b_corrected`, `tau_corrected`, `features`, and the located maxima `sigma_b_corr_max` / `x_sigma_b_corr_max` / `tau_corr_max` / `x_tau_corr_max`. |
| `ShaftPostProcessor` | `ShaftPostProcessor(shaft_system, results).process()` → `PostProcessedResults`. |

Internals: `_shoulder_scf` / `_keyway_scf` build one `StressConcentration` per feature; `_kt_bending_shoulder` / `_kt_torsion_shoulder` are the Peterson Eq. 5.2 / Shigley Table 6-2 curve fits (`Kt ≈ C1 + C2·(2r/d) + C3·(2r/d)² + C4·(2r/d)³`); `_notch_sensitivity` interpolates √a from the Neuber table against Su (torsion referenced to `Ssy ≈ 0.577·Su`); `_get_Su` reads the material database at position x, falling back to 700 MPa.

**`shaft/utils.py`** — pure helper functions, stateless, reusable by the fatigue solvers:
- Marin factors: `ka_surface_finish(Sut, finish)`, `kb_size(diameter_mm)`, `kc_load(load_type)`, `kd_temperature(T)`, `ke_reliability(percent)`, `ke_reliability_from_z(z)`, and the combined `endurance_limit_corrected(...)` → dict of every factor plus `Se'`.
- Notch factors: `kt_shoulder_bending(r_over_d, D_over_d=1.5)`, `kt_shoulder_torsion(...)`, `neuber_constant_sqrt_a(Sut, loading)`, `notch_sensitivity(r_mm, Sut, loading)`, `kf_from_kt(Kt, q)`.

> `kc_load` returns Shigley's per-load-type factor; for combined bending + torsion the caller must not apply it to both terms — see the function's own docstring.

---

## solvers — Bearings (ISO/TS 16281)

> **Two structural changes since the previous doc pass.**
> **(1) Dispatch is no longer keyed by `BearingType`.** The `_SOLVER_MAP` / `_POSTPROC_MAP` enum tables are gone; `dispatch.py` now resolves a solver by matching a bearing's declared `CAPABILITIES` and `REQUIRED_FOR` attributes against a registry that each contact solver joins at import time via `@register_contact_solver`. Adding a family no longer means editing a table.
> **(2) Single-row and multi-row now share one result type.** `BallBearingResult` / `RollerBearingResult` wrap `rows: list[...LoadDistributionResult]` — length 1 for an ordinary bearing, length `i` for a multi-row one. `MultiRowBallLoadDistributionResult` is retired. Downstream code no longer branches on row count.
> Capacity remains owned by `core/`: `bearing.family.per_element_dynamic_capacity(...)` / `per_lamina_dynamic_capacity(...)`.

The package is split by contact type, mirroring `core/.../families/{ball_bearing,roller_bearing}`, with a shared library, a capability-based dispatcher, and an orchestrator on top:

```
ISO_16281/
├── dispatch.py                         ← register_contact_solver, resolve_solver_cls,
│                                          resolve_solver_cls_for_attrs, SolverDispatchError
├── library.py                          ← generic utilities, cross-type results registry
├── rolling_bearing_solver.py           ← RollingBearingSolver — the single entry point
├── Ball_Bearing/
│   ├── single_row_solver.py            ← ISO16281BallSolver, debug_radial_capacity
│   ├── multirow_solver.py              ← ISO16281MultiRowBallSolverSharedDisplacement
│   │                                      (registered as ISO16281BallSolver.MULTIROW_SOLVER)
│   ├── ball_bearing_multirow_solver.py ← ISO16281MultiRowBallSolver — kept for comparison only
│   ├── results.py                      ← BallLoadDistributionResult, BallBearingResult,
│   │                                      BallLoadDistributionLibrary
│   └── postprocessing.py               ← Q_j, phi_j_global, contact_distribution,
│                                          BallBearingStiffness, bearing_stiffness,
│                                          DynamicEquivalentRollingElementLoad,
│                                          BasicReferenceRatingLife,
│                                          DynamicEquivalentReferenceLoad,
│                                          combine_row_L10r, basic_reference_rating_life
└── Roller_Bearing/
    ├── single_row_solver.py            ← ISO16281RollerSolver, debug_radial_capacity
    ├── multirow_solver.py              ← ISO16281MultiRowRollerSolverSharedDisplacement
    ├── results.py                      ← RollerLoadDistributionResult, RollerBearingResult,
    │                                      RollerLoadDistributionLibrary
    └── postprocessing.py               ← as the ball side, plus lamina_distribution,
                                           stress_riser_factor, LaminaDynamicEquivalentLoad
```

Each contact type owns its full vertical slice — solve, result shape, postprocessing — and never imports the other at runtime (`Roller_Bearing/postprocessing.py` reuses the ball side's `combine_row_L10r` / `BasicReferenceRatingLife.from_loads` deliberately, as the formula is genuinely identical). The two packages are wired together at exactly three points: `library.py`'s `BearingResultsLibrary` (duck-typed aggregation), `dispatch.py`'s registry, and `rolling_bearing_solver.py`.

### `dispatch.py` — solver lookup by capability

- `register_contact_solver(cls)` — decorator applied by `Ball_Bearing/single_row_solver.py` and `Roller_Bearing/single_row_solver.py` at import time. A registered class declares which `REQUIRED_ATTRS` it needs.
- `resolve_solver_cls(bearing, label)` → the single-row solver class whose contract this assembled bearing satisfies, chosen by `bearing.is_enabled(analysis)` + `family.REQUIRED_FOR`. Ambiguity is broken by "most specific wins" (largest satisfied attribute set).
- `resolve_solver_cls_for_attrs(attrs, label)` → the same match from a bare attribute set. This is what makes multi-row bearings work: their `rows` are plain dicts that never pass through `Bearing.assemble()`, so the row's own attribute set is matched directly.
- `SolverDispatchError(RuntimeError)` — raised naming the offending label when nothing matches, or when a multi-row bearing's resolved row solver has no `MULTIROW_SOLVER`.

### `library.py` — generic utilities and the cross-type results registry

**Block 1 — generic, contact-agnostic utilities**, used by every solver:
- `check_bearing_ready(bearing, label, required_attrs)` — raises `RuntimeError` naming whichever attribute is still `None`.
- `warn_if_floating_loaded(bearing, label, Fa, eps=FA_FLOATING_EPS)` — warns if a bearing marked `arrangement="floating"` carries `Fa` above the floor.
- `run_root(fun, x0, tol)` — `scipy.optimize.root`, `hybr` with an `lm` fallback if `hybr` doesn't converge or converges to a worse residual. Returns `(x, nfev, residual_norm, success)`.

**Block 2 — `BearingResultBundle` / `BearingResultsLibrary`**, the per-bearing-label registry every downstream consumer (life, lubrication, fatigue) reads from:
- `BearingResultBundle` — `label`, `bearing_type`, `load_distribution` (a **list** of per-row results), `stiffness`, `capacity`, `dynamic_equivalent_load` (also a list, index-aligned with the rows), `extra: dict[str, object]`.
- `add_load_distribution_library(solver_cls, local_library, bearings)` — the primary path: an entire per-solver local library handed over wholesale, keyed by the **solver class** (not by `BearingType` any more).
- `load_distribution_library(solver_cls)` — retrieves that whole sub-library.
- `set_load_distribution(label, result, bearing_type=None)` — single-label path, used for multi-row results that bypass a local library.
- `set_bearing_type` / `set_capacity` / `set_dynamic_equivalent_load` / `set_stiffness` / `set_extra` / `get` / `labels`. `set_*()` overwrites; `get()` raises `KeyError` for an unset label.

### `rolling_bearing_solver.py` — `RollingBearingSolver`

Orchestrates a shaft's full, possibly mixed bearing set — mixed contact types (a locating DGBB plus a floating cylindrical roller bearing) *and* mixed row counts on the same shaft.

- `_row_solver_cls_for(bearing, label)` — the single-row solver whose contract this bearing's rows satisfy. Used both for normal dispatch and to pick the postprocessing adapter for a multi-row bearing (the adapter belongs to the *contact type*, not to the multi-row solver).
- `_multirow_solver_cls_for(bearing, label)` — `.MULTIROW_SOLVER` on the resolved row solver; raises `SolverDispatchError` if unset.
- `solve(shaft_system, bearings, library, psi_override=None, results=None)` → `{label: list[LoadDistributionResult]}`, ordered to match the caller's `bearings` dict. Single-row bearings are grouped by resolved solver and dispatched per group; multi-row bearings (`.rows` count ≥ 2) go individually to their multi-row solver, and the resulting `BallBearingResult`/`RollerBearingResult` is stored whole under `results.set_extra(label, "multirow_result", ...)`.
- `postprocess_and_record(shaft_system, bearings, library, catalog, load_distribution=None, results=None, psi_override=None)` → `BearingResultsLibrary`. Full pipeline: `solve()` (or reuse an already-computed `load_distribution`) → capacity → dynamic equivalent load → secant stiffness, dispatched per bearing through `_POSTPROC`.

  **Capacity dispatch.** `_PostprocAdapter` carries only `result_cls`, `derel_cls` and `stiffness_fn` — the three things that genuinely differ per contact type and have no `core/` equivalent. Capacity is computed by calling

```python
  b.family.per_element_dynamic_capacity(b, **cap_kwargs)
```

  directly — every `BearingFamily` already knows which of its own formulas applies (the thrust families auto-dispatch on `alpha_0 == 90°`; radial families never had a choice). `catalog` is `{label: {"capacity": {...}, "dynamic_equivalent_load": {...}}}`; `catalog[label]["capacity"]` is forwarded as `**cap_kwargs` — `{"Cr": ...}` for a radial family, `{"Ca": ...}` for a thrust family, optionally `{"i": ..., "lambda_v": ...}` for roller families. Either sub-dict may be omitted per label to skip that step — but **stiffness is computed unconditionally for every bearing in `bearings`**. The adapter lookup runs *before* any capacity/derel/stiffness work, so a bearing with no registered adapter fails fast rather than leaving a partially-populated bundle.

### Ball_Bearing — point contact

**`single_row_solver.py` — `ISO16281BallSolver`** *(was `ball_bearing_solver.py`)*

Coupled shaft–bearing solver for point-contact bearings — `alpha_0` is read off the bearing instance, so one class covers DGBB, angular contact, and each row of a multi-row thrust ball bearing. Prescribed-ψ formulation: per raceway, a single 2-DOF root solve (δr, δa) in the plane of the resultant radial force.
- `REQUIRED_ATTRS` declares its contract to `dispatch.py`; `MULTIROW_SOLVER` is set by `multirow_solver.py` at import time.
- `solve(shaft_system, bearings, library, psi_override=None)` → `BallLoadDistributionLibrary` (entries are `BallBearingResult`, wrapped via `.single()`).
- `elements(bearing, delta_r, delta_a, Vpsi)` *(staticmethod, promoted from `_elements`)* — per-element elastic deflection and effective contact angle; `phi_j` local, 0 aligned with the resultant radial force; `delta_j` zero-floored.
- `solve_contact(bearing, Fr_xz, Fr_xy, Fa, delta_r_init, delta_a_init, psi, phi_Fr)` → `BallLoadDistributionResult` — the 2-equation root for ONE raceway (a whole single-row bearing, or one row of a multi-row one).
- `minimum_axial_load(bearing, Fr_xz, Fr_xy, psi, delta_r_init=0.0, delta_a_init=0.0, Fa_bracket=(0.0, 5e4), xtol=1e-6)` → `(Fa_min, BallLoadDistributionResult)` — smallest axial preload such that δa ≥ 0, via `brentq` on δa(Fa).
- `_initial_delta_r` / `_initial_delta_a` — trust a non-negligible FEM hint, fall back to a Hertz-scale estimate otherwise.
- `debug_radial_capacity(bearing, Cr, i=None, label="")` — manual cross-check; sources `Q_ci`/`Q_ce` from `bearing.family.per_element_dynamic_capacity()`, the *same* call production makes, so a debug print can never drift.

**`multirow_solver.py` — `ISO16281MultiRowBallSolverSharedDisplacement`**

The multi-row solver actually reached by dispatch. One flat root-find on the **shared** unknowns (δr, δa) of the rigid ring; each row contributes its own Hertzian reaction directly, summed.
- `solve_bearing(bearing, Fr_xz, Fr_xy, Fa, psi, label="")` → `BallBearingResult` via `.multirow()`, with `f_r`, `f_a`, `outer_n_iter`, `outer_residual`, `outer_ok` populated. `bearing` must expose `.rows` (each carrying its own `cp`) and `.i`.

**`ball_bearing_multirow_solver.py` — `ISO16281MultiRowBallSolver`** *(reference only)*

The earlier load-split-fraction formulation, kept in the codebase purely so the comparison script can run both side by side. **Empirically superseded:** on a real gearbox model with a double-row bearing at `Fa ≈ 0` and heterogeneous rows (different `cp`), it fails to converge (`outer_ok=False`, residual stuck ~6e-3, hits the iteration cap) — the axial-side counterpart of the known `Fr ≈ 0` degeneracy its own `FR_NEGLIGIBLE_EPS` handles: when `Fa_total ≈ 0`, any per-row axial fraction satisfies `f_a_row·Fa_total = 0` and that Jacobian column goes singular. The shared-displacement solver never divides by a possibly-zero total and converged cleanly (residual ~1e-6…1e-7) on every case tested.

**`results.py` — `BallLoadDistributionResult`, `BallBearingResult`, `BallLoadDistributionLibrary`**
- `BallLoadDistributionResult` — the raw output of ONE point-contact solve, unchanged: `delta_r`, `delta_a`, `psi`, `phi_Fr`, `delta_j`, `alpha_j`, `Mz`, `n_iter`, `residual`, `ok`. `Q_j` is deliberately *not* stored — derived on demand by `postprocessing.Q_j()`. For a multi-row bearing, row *j*'s entry is exactly what this class would hold had row *j* been solved alone with its converged share of the load.
- `BallBearingResult` *(new)* — the unified per-**bearing** result. `rows: list[BallLoadDistributionResult]` is the only required field; `is_multirow` is derived from `len(rows) >= 2`, never stored as a second flag that could drift. `f_r`, `f_a`, `outer_n_iter`, `outer_residual`, `outer_ok` are optional (None for a single-row result). Constructors: `.single(row)` and `.multirow(rows, f_r, f_a, n_iter, residual, ok)`. Convenience `delta_r` / `delta_a` / `psi` properties read `rows[0]`.
- `BallLoadDistributionLibrary` — local, single-contact-type registry of `BallBearingResult`: `set` / `get` / `labels` / `__contains__` / `__iter__` / `__len__`.

**`postprocessing.py`**

All the per-row functions return a **list, one entry per row** (length 1 for a single-row bearing) — the uniform shape that removed the row-count branching downstream.
- `Q_j(bearing, result)` — per-row, per-element contact force `cp · delta_j^1.5`.
- `phi_j_global(bearing, result)` — per-row ball angular positions in the global frame, wrapped to `[0, 2π)`.
- `contact_distribution(bearing, result, frame="global"|"local")` — per-row `(Z_row × 2)` arrays `[phi, Q_j]`.
- `BallBearingStiffness.from_result(label, result, Fr_xz, Fr_xy, Fa, eps=1e-9)` — secant stiffness: `Kr_xz`, `Kr_xy`, `Ka`, and an `Ka_regime` of `"no_load"` / `"engaged"` / `"closing_clearance"`. `bearing_stiffness(bearing, result, Fr_xz, Fr_xy, Fa, eps=1e-9)` is the public wrapper and returns a single stiffness for the whole bearing regardless of row count (it reads `rows[0]` — the rows share one rigid-ring displacement).
- `DynamicEquivalentRollingElementLoad` — §4.3.2 eq.(25)-(28), power mean over `Q_j` with exponent 3 (rotating relative to the load) or 10/3 (stationary). `from_distribution(...)` is the single-row primitive; `from_bearing_result(...)` is the row-aware entry point and **always returns a list**.
- `BasicReferenceRatingLife.from_loads(label, Q_ci, Q_ei, Q_ce, Q_ee)` — L10r for one row/raceway pair, eq.(29).
- `combine_row_L10r(L10r_rows, e=_E_BALL)` — bearing-level L10r from n rows, Zaretsky eq.(49a): `L^-e = Σ Lᵢ^-e`.
- `basic_reference_rating_life(label, Q_ci_rows, Q_ei_rows, Q_ce_rows, Q_ee_rows, e=_E_BALL)` → `(per-row records, combined L10r)`.
- `DynamicEquivalentReferenceLoad.from_L10r(label, L10r_bearing, Cr=None, Ca=None)` — eq.(30)-(31): `Pref_r = Cr/L10r^(1/3)`, `Pref_a = Ca/L10r^(1/3)`.

### Roller_Bearing — line contact

**`single_row_solver.py` — `ISO16281RollerSolver`** *(was `roller_bearing_solver.py`)*

§5.2 lamina-model solver for radial cylindrical roller bearings (NU/N-type, zero nominal contact angle, no axial capacity). Each roller sliced into `n_s` (≥30) laminae, corrected for the roller's logarithmic profile.
- `REQUIRED_ATTRS = ("Z", "Dwe", "Lwe", "Dpw", "phi_j", "s", "n_s", "x_k", "cL", "cs", "alpha_0", "P_xk")`.
- `solve(shaft_system, bearings, library, psi_override=None)` → `RollerLoadDistributionLibrary` (entries are `RollerBearingResult`). `_check_lamina_count()` validates `n_s >= 30` and `len(x_k) == n_s`.
- Only **δr** is solved (radial force balance, eq.45); **ψ is an input**, taken from the FEM shaft slope projection (or `psi_override`), same as the ball solver — eq.(46) is evaluated afterwards as a diagnostic `Mz`, not a solve constraint.
- `elements(bearing, delta_r, psi)` *(staticmethod)* — per-roller and per-lamina deflection/force. It reads the precomputed `bearing.P_xk`, cached at `core/` assembly time by each subtype's `_reference_roller_profile()` (eq.42-44); this solver never computes the profile itself.
- `debug_radial_capacity(bearing, Cr, i=1, lambda_v=None, label="")` — mirrors the ball side; sources `Q_ci`/`Q_ce` and `q_ci`/`q_ce` from the family, never a local reimplementation.

**`multirow_solver.py` — `ISO16281MultiRowRollerSolverSharedDisplacement`**

Structural counterpart of the ball side, one shared unknown (δr) instead of two. Registered as `ISO16281RollerSolver.MULTIROW_SOLVER`. **Caveat, flagged:** unlike the ball side there is no real multi-row roller family in `core/` to validate it against — it is preferred going forward because it mirrors the validated ball architecture, not because it has been proven on a case.

**`results.py` — `RollerLoadDistributionResult`, `RollerBearingResult`, `RollerLoadDistributionLibrary`**
- `RollerLoadDistributionResult` — base fields (`delta_a` always `0.0` — no axial load) plus the lamina-model extras `x_k`, `psi_j`, `delta_jk` (Z×n_s), `q_jk` (Z×n_s).
- `RollerBearingResult` — same container shape as `BallBearingResult`; `rows` is length 1 for `CylindricalRollerFamily`. `.multirow()` exists but nothing produces that shape yet.
- `RollerLoadDistributionLibrary` — same shape as its ball counterpart.

**`postprocessing.py`**
- `Q_j(bearing, result)` — per-row, per-roller total force `Σ_k q_jk`.
- `phi_j_global`, `contact_distribution` — mirror the ball side, per-row lists.
- `lamina_distribution(bearing, result, j)` → `(n_s × 2)` `[x_k, q_j,k]` for a single roller `j` — the pressure-profile-along-the-roller view.
- `RollerBearingStiffness.from_result(...)` / `bearing_stiffness(bearing, result, Fr_xz, Fr_xy, eps=1e-9)` — same secant projection as the ball side, deliberately duplicated. `Fa` is fixed at `0.0`, so `Ka` comes back `inf` / `"no_load"` — the physically correct statement for a radial roller bearing.
- `stress_riser_factor(n_s)` — §5.3.3 eq.(60) approximation for the edge-stress concentration factor. Validity conditions (medium load, misalignment < 4′, logarithmic profile) are **not checked**.
- `LaminaDynamicEquivalentLoad` — §5.3.4 eq.(61)-(64), **per lamina** (`q_kei`, `q_kee` are `(n_s,)` arrays); `from_distribution(...)` single-row primitive, `from_bearing_result(...)` row-aware list. Compare against `bearing.family.per_lamina_dynamic_capacity()`'s `q_ci`/`q_ce`, not `Q_ci`/`Q_ce`.
- `BasicReferenceRatingLife` — L10r for one row/raceway pair, eq.(65), summed over the `n_s` laminae. `basic_reference_rating_life(...)` and `combine_row_L10r(..., e=_E_ROLLER)` mirror the ball side.
- `DynamicEquivalentReferenceLoad.from_L10r(label, L10r, Cr=None, Ca=None)` — eq.(66)-(67), exponent 3/10.

> §5.4 — combining reference rating lives across laminae into a modified rating life — is still not implemented on either side.

---

## solvers — Bearing life (ISO 281)

**`bearings/life.py` — `BearingLifeSolver`**

ISO 281 basic rating life and static safety, kept separate from the ISO/TS 16281 internal-distribution stack (which produces the *reference* rating life inputs).
- `solve_bearing(bearing, Fr, Fa, speed_rpm, design_life_hours=DEFAULT_DESIGN_LIFE_HOURS)` → `BearingLifeResult`. `L10 = (C/P)^p` [10⁶ rev], `p = 3` for ball / `10/3` for roller.
- `extract_bearing_forces(statics_result, system)` → `{label: (Fr, Fa)}` from a statics result's reactions.

> **Phase 1 simplification, stated in the module docstring:** `P = X·Fr + Y·Fa` with `X = 1.0`, `Y = 0.0`, i.e. `P = Fr` — axial load is not yet folded into the equivalent load.

---

## solvers — Gears

**`gears/geometry.py` — `GearSolver`**
Cylindrical gear geometry and mesh force calculation (MAAG / ISO 21771 / Shigley §13-7). Pure functions, no internal state.
- `compute_geometry(mn, z1, z2, alpha_n_deg, beta_deg, al=None, x1=0.0, x2=0.0, b=0.0)` — full pair geometry with optional profile shift; working centre distance from `al` or the involute equation (`brentq`); tip/root/working diameters; contact ratios. Returns `GearGeometryResult`.
- `compute_forces(T1_Nm, geometry)` → `GearForceResult` — `Ft = T1/rl1`, `Fbt = T1/rb1`, `Fr = Fbt·sin(αtw)`, `Fa = Fbt·tan(βb)` (exact zero for spur), `Fbn = Fbt/cos βb`, `Fn = Ft/cos βb` (GEARpie/MAAG convention — noted in the source as deliberately different from `√(Ft²+Fr²+Fa²)`).
- `to_gear_element(position, forces, geometry, label="")` — assembles a `GearElement` using `dl1` (working pitch circle) so the torque-consistency check passes.

**`gears/utils.py`** — geometry helpers: `involute(alpha)`, `solve_alpha_tw(...)` (involute equation via `brentq`), `contact_ratio_alpha(...)`, `contact_ratio_beta(...)`, `undercut_z_min(alpha_n_deg, beta_deg)`, `validate_geometry_inputs(...)`, plus the rack constants HAP/HFP.

**`gears/SpurHelicalGears/LoadCapacity_solver/load_capacity.py`** — reserved for ISO 6336-2/-3. **Currently empty**; the supporting data layer (K_A table, K_v Method B and Method C) already exists under [`axisforge/database/`](../core/README.md#adjacent-axisforgedatabase).

Result containers `GearGeometryResult` / `GearForceResult` are documented in [`models/README.md`](../models/README.md).

---

## solvers — Mesh convergence

Moved out of `mesh/` — it is a solver-driven study, not mesh generation. The primitives it refines (`Mesh1D`, `Grader`) stay in [`mesh/README.md`](../mesh/README.md).

**`mesh/mesh_convergence_study.py` — `RichardsonGCI`, `MeshConvergenceStudy`, `ConvergenceRecord`, `MeshRefinementResult`**

Grid Convergence Index via Richardson extrapolation on the resultant transverse displacement, across ≥3 refinement levels.
- `RichardsonGCI(f_coarse, f_medium, f_fine, x_nodes_coarse, x_nodes_medium, x_nodes_fine, gci_threshold=0.01, safety_factor=1.25)` — computes the refinement ratios `r_m_c` / `r_f_m`, the observed order `p`, the relative errors, the extrapolated `f_h0`, `GCI_m_c` / `GCI_f_m` and the `converged` flags.
- `MeshConvergenceStudy(global_solver, gci_threshold=0.01, safety_factor=1.25, max_levels=8)`.
  - `run(shaft_system, intervals)` → `MeshRefinementResult`, one `ConvergenceRecord` per load.
  - `intervals_from_shaft_system(shaft_system)` *(staticmethod)* → `(intervals, skipped)` — bearing extents are excluded automatically (prescribed BC, no physical gain) and reported in `skipped`.
  - `_eval_points_for_interval(...)` — physically meaningful evaluation points inside the interval; raises if an overlapping element or load makes the interval ill-posed.
  - Metric: mean `|v_xz|`, mean `|v_xy|` and their resultant over the evaluation points, extracted from a `SubmodelResult`.
- `MeshRefinementResult.all_extra_nodes` — the union of every converged `x_final`, sorted, ready to pass to `Mesh1D(extra_mandatory=...)` for production runs.
- `MeshRefinementResult.print_report(unit_label="mm")` — per-interval convergence status, ASCII only.

---

## solvers — Lubrication (placeholder)

**`lubrification/__init__.py`** — intentionally empty (`__all__: list[str] = []`). Phase 4 will add `FilmSolver`, `RegimeSolver`, `GreaseSolver`; Phase 5+ a CFD interface. Documented here so the empty package is not mistaken for an oversight.