# axisforge/solvers — Module Reference

Solvers consume the domain objects documented in [`core/README.md`](../core/README.md) (`Shaft`, `ShaftSystem`, `Bearing` subtypes, `SpurHelicalGear`, ...) and the mesh objects in [`mesh/README.md`](../mesh/README.md), and publish results as explicit, GUI-independent containers. No solver imports another solver's internals — only the shared result containers.

← back to [project root](../../README.md)

---

## Table of Contents

- [solvers — Shaft FEM](#solvers--shaft-fem)
- [solvers — Bearings (ISO/TS 16281)](#solvers--bearings-isots-16281)
- [solvers — Gears](#solvers--gears)

---

## solvers — Shaft FEM

**`build_stiffness_matrix.py` — `StiffnessMatrixBuilder`**
Assembles the global stiffness matrix from element contributions. Currently wires the Timoshenko beam theory; extensible via the `_BEAM_THEORIES` registry.
- `build_stiffness_matrix(mesh, elements)` — global K (3 DOF/node: u, v, θ).

**`FEM_solvers/simple_fem_solver.py` — `SimpleFEMSolver`**
Orchestrates the full pipeline: `Mesh1D` → `Elem.from_mesh` → `StiffnessMatrixBuilder` → boundary conditions → two independent planar solves (XZ, XY) sharing the same K → superposition → torsion diagram on the same nodes. Every intermediate quantity is stored as a public attribute (numerical transparency).
- `solve(shaft_system, extra_mandatory)` — runs the full solve; stores `x_nodes`, `elements`, `free_dofs`, displacement vectors `d_total_xz`/`d_total_xy`, external force vectors, torsion arrays `T_total`/`tau_total`, and reactions.
- `return_values(x_nodes, [x_lo, x_hi])` — nodal solution quantities within an interval (for submodelling).
- Options: `theory`, `constraint_bearing`, `distribute_gear_labels` (which gear mesh loads are treated as distributed over face width).

**`FEM_solvers/submodel_solver.py` — `SubmodelSolver`, `SubmodelResult`**
Wraps `SimpleFEMSolver` and restricts metric evaluation to a subdomain [x_lo, x_hi], with Lagrange-multiplier boundary injection at the cut nodes. Used exclusively by the convergence study (see [`mesh/README.md`](../mesh/README.md#mesh--1d-shaft-mesh)). `SubmodelResult` carries the subdomain displacement vectors and cut-node reaction multipliers.

**`static/static_analysis.py`**

| Class | Purpose |
|-------|---------|
| `BearingNodeData` | Complete FEM nodal state at a bearing position (displacements, reactions, seat misalignment ψ). |
| `ShaftResults` | Full FEM solution + post-processed engineering quantities for one shaft. |
| `SimpleFEMResultsLibrary` | Registry of `ShaftResults` keyed by shaft name — the canonical source all downstream solvers read from. |
| `ShaftResultsReader` | Post-processes a solved `SimpleFEMSolver` into a `ShaftResults` and stores it in the library. |

`BearingNodeData` carries `Fr_xz`, `Fr_xy`, `Fr`, `Fa`, moment reactions, displacements, and the seat-slope misalignment `psi_xz`/`psi_xy` (gradient of v across the seat, or nodal θ for zero-width seats). `ShaftResults` is organised into mesh, raw FEM solution, post-processed engineering quantities (internal forces, deflections, section stresses σ_b, τ), and per-bearing node data. The library provides `store`/`get`/`get_or_none`/`remove`/`clear`/`names`/`iter`/`all_results`. `ShaftResultsReader.read(library)` recovers internal forces, deflections, section properties, bearing reactions and node data in one pass.

---

## solvers — Bearings (ISO/TS 16281)

> **Re-verified against source — 2026-08-20.** Capacity (Q_ci/Q_ce, Cr/Ca) has been migrated out of both per-type solver modules and out of this package entirely: it's now owned by `core/machine_elements/Bearings/families/{ball_bearing,roller_bearing}/{radial,thrust}/functions/capacity.py`, called uniformly via `bearing.family.per_element_dynamic_capacity(bearing, ...)` / `bearing.family.per_lamina_dynamic_capacity(bearing, Q_ci, Q_ce)`. The `RollingElementCapacity`/`RollerElementCapacity` dataclasses that used to live in `ball_bearing_solver.py`/`roller_bearing_solver.py` are gone — see each module's docstring below for what replaced them. This supersedes the previous verification pass, which still described those dataclasses as current.

The bearings solver package is split by contact type, mirroring `core/machine_elements/Bearings/families/{ball_bearing,roller_bearing}` (see [`core/README.md`](../core/README.md#coremachine_elements--bearings)), with a shared library of generic utilities and a type-dispatching orchestrator on top:

```
ISO_16281/
├── library.py                          ← generic utilities, cross-type results registry
├── rolling_bearing_solver.py           ← RollingBearingSolver -- per-BearingType dispatch
├── Ball_Bearing/
│   ├── ball_bearing_solver.py          ← ISO16281BallSolver, debug_radial_capacity
│   ├── ball_bearing_results.py         ← BallLoadDistributionResult, BallLoadDistributionLibrary
│   └── ball_bearing_postprocessing.py  ← Q_j, phi_j_global, contact_distribution,
│                                          BallBearingStiffness, bearing_stiffness,
│                                          DynamicEquivalentRollingElementLoad
└── Roller_Bearing/
    ├── roller_bearing_solver.py           ← ISO16281RollerSolver, debug_radial_capacity
    ├── roller_bearing_results.py          ← RollerLoadDistributionResult,
    │                                         RollerLoadDistributionLibrary
    └── roller_bearing_postprocessing.py   ← Q_j, phi_j_global, contact_distribution,
                                              lamina_distribution, RollerBearingStiffness,
                                              bearing_stiffness, stress_riser_factor,
                                              LaminaDynamicEquivalentLoad
```

Each contact type owns its full vertical slice — solve, result shape, and postprocessing — and never imports the other. The two packages are wired together at exactly two points: `library.py`'s `BearingResultsLibrary`, which aggregates results from either side under one label without needing to know which side produced them (duck typing, not a shared base class), and `rolling_bearing_solver.py`'s `RollingBearingSolver`, the only module in the package aware that more than one contact type exists.

### `library.py` — generic utilities and the cross-type results registry

**Block 1 — generic, contact-agnostic utilities**, used by both per-type solvers:
- `check_bearing_ready(bearing, label, required_attrs)` — raises `RuntimeError` naming whichever of `required_attrs` is still `None` on the bearing.
- `warn_if_floating_loaded(bearing, label, Fa, eps=FA_FLOATING_EPS)` — warns if a bearing marked `arrangement="floating"` is carrying `Fa` above the 1e-6 N floor.
- `run_root(fun, x0, tol)` — `scipy.optimize.root`, `hybr` with an `lm` fallback if `hybr` doesn't converge or converges to a worse residual than `lm`. Returns `(x, nfev, residual_norm, success)`.

**Block 2 — `BearingResultBundle` / `BearingResultsLibrary`**, the per-bearing-label registry every downstream consumer (lubrication, fatigue, ...) is meant to read from:
- `BearingResultBundle` — dataclass holding everything computed for one label: `bearing_type`, `load_distribution`, `stiffness`, `capacity`, `dynamic_equivalent_load`, `extra: dict[str, object]`. `capacity` is now typed `CapacityResult = tuple[float, float]` (a `TYPE_CHECKING`-only alias) — whatever `bearing.family.per_element_dynamic_capacity()` returns, for any family, ball or roller alike. No per-contact-type import needed for it anymore.
- `BearingResultsLibrary.add_load_distribution_library(bearing_type, local_library)` — the primary way load distribution results enter this registry: an entire per-type local library handed over wholesale, by reference, one call per `BearingType` group.
- `set_capacity` / `set_dynamic_equivalent_load` / `set_stiffness` / `set_extra` / `get` / `labels` / `load_distribution_library(bearing_type)` round out the registry. `set_*()` overwrites; `get()` raises `KeyError` for an unset label.

### `rolling_bearing_solver.py` — `RollingBearingSolver`

Orchestrates a shaft's full, possibly mixed-type bearing set (e.g. a locating deep-groove ball bearing plus a floating cylindrical roller bearing on the same shaft). Only module in the package that imports both `Ball_Bearing` and `Roller_Bearing`.

- `_SOLVER_MAP`: `DEEP_GROOVE_BALL` / `ANGULAR_CONTACT` → `ISO16281BallSolver`; `CYLINDRICAL_ROLLER` → `ISO16281RollerSolver`. `SELF_ALIGNING_BALL`/`THRUST_BALL`/`THRUST_CYLINDRICAL_ROLLER`/`THRUST_NEEDLE_ROLLER` have core-level capacity support already but no internal load-distribution solver wired yet (Phase 2), same for `TAPERED_ROLLER`/`SPHERICAL_ROLLER` (needs an extra coordinate transform neither solver implements). An unregistered `BearingType` raises `NotImplementedError` at solve time, naming the offending labels.
- `solve(shaft_system, bearings, library, psi_override=None, results=None)` → `{label: LoadDistributionResult}`, ordered to match the caller's `bearings` dict. Groups `bearings` by `BearingType`, dispatches each group to its per-type solver, and — if a `BearingResultsLibrary` is passed as `results` — records each group's local library into it via `add_load_distribution_library()`.
- `postprocess_and_record(shaft_system, bearings, library, catalog, load_distribution=None, results=None, psi_override=None)` → `BearingResultsLibrary`. Full pipeline: `solve()` (or reuse an already-computed `load_distribution` dict) → capacity → dynamic equivalent load → secant stiffness, dispatched per bearing through `_POSTPROC_MAP`.

  **Capacity dispatch — no `capacity_cls`/`"method"` key anymore.** `_PostprocAdapter` now carries only `derel_cls` and `stiffness_fn` (the two things that genuinely still differ per contact type and have no `core/` equivalent). Capacity is computed by calling

```python
  b.family.per_element_dynamic_capacity(b, **cap_kwargs)
```

  directly — every `BearingFamily` already knows which of its own capacity formulas applies (`ThrustBallFamily`/`ThrustCylindricalRollerFamily`/`ThrustNeedleRollerFamily` auto-dispatch on `alpha_0 == 90°` internally; radial families never had a choice). `catalog` is `{label: {"capacity": {...}, "dynamic_equivalent_load": {...}}}`; `catalog[label]["capacity"]` is forwarded as `**cap_kwargs` — `{"Cr": ...}` for a radial ball/roller family, `{"Ca": ...}` for a thrust family, optionally `{"i": ..., "lambda_v": ...}` for roller families that expose them. Either sub-dict may be omitted per label to skip that step — but **stiffness is computed unconditionally for every bearing in `bearings`**. The adapter-registered check (`_POSTPROC_MAP.get(b.bearing_type)`) runs *before* any capacity/derel/stiffness work, so a bearing type with no registered adapter fails fast rather than leaving a partially-populated bundle.

### Ball_Bearing — point contact (`DEEP_GROOVE_BALL`, `ANGULAR_CONTACT`)

**`ball_bearing_solver.py` — `ISO16281BallSolver`**

Coupled shaft–bearing solver for point-contact bearings, covering both `DEEP_GROOVE_BALL` and `ANGULAR_CONTACT` — `alpha_0` is read off the bearing instance, so one class covers both families. Prescribed-ψ formulation: per bearing, a single 2-DOF root solve (δr, δa) in the plane of the resultant radial force.
- `solve(shaft_system, bearings, library, psi_override=None)` → `BallLoadDistributionLibrary`.
- `minimum_axial_load(bearing, Fr_xz, Fr_xy, psi, delta_r_init=0.0, delta_a_init=0.0, Fa_bracket=(0.0, 5e4), xtol=1e-6)` → `(Fa_min, BallLoadDistributionResult)` — smallest axial preload such that δa ≥ 0, via `brentq` on δa(Fa).
- **`debug_radial_capacity(bearing, Cr, i=None, label="")`** — manual cross-check, prints `Q_ci`/`Q_ce`. No longer carries its own copy of eq.(19)-(20): sources its numbers from `bearing.family.per_element_dynamic_capacity()`, the *same* call `postprocess_and_record()` makes, so a debug print here can never silently drift from what production returns.
- Seeding for the root solve (`_initial_delta_r`, `_initial_delta_a`) trusts a non-negligible FEM hint and only falls back to a Hertz-scale estimate when the hint is ~0.

There is no longer a `RollingElementCapacity` dataclass in this file — it used to duplicate eq.(19)-(24) against `core/`; removed, see the module's own docstring for the full reasoning.

**`ball_bearing_results.py` — `BallLoadDistributionResult`, `BallLoadDistributionLibrary`**
- `BallLoadDistributionResult` — `__slots__` container, no dependency on `library.py`: `delta_r`, `delta_a`, `psi`, `phi_Fr`, `delta_j`, `alpha_j`, `Mz`, `n_iter`, `residual`, `ok`. `Q_j` is deliberately *not* stored — derived on demand by `ball_bearing_postprocessing.Q_j()`.
- `BallLoadDistributionLibrary` — local, single-type registry: `set`/`get`/`labels`/`__contains__`/`__iter__`/`__len__`.

**`ball_bearing_postprocessing.py`**
- `Q_j(bearing, result)` — per-element contact force, `cp * delta_j^1.5`.
- `phi_j_global(bearing, result)` — ball angular positions in the global frame.
- `contact_distribution(bearing, result, frame="global"|"local")` → (Z×2) array `[phi, Q_j]`.
- `BallBearingStiffness.from_result(label, result, Fr_xz, Fr_xy, Fa, eps=1e-9)` — secant stiffness: `Kr_xz`, `Kr_xy`, `Ka`, and an `Ka_regime` of `"no_load"` / `"engaged"` / `"closing_clearance"`. `bearing_stiffness(bearing, result, Fr_xz, Fr_xy, Fa, eps=1e-9)` is the thin wrapper most callers use.
- `DynamicEquivalentRollingElementLoad.from_distribution(bearing, result, inner_rotating=True, outer_rotating=False, label="")` — §4.3.2 eq.(25)-(28); power-mean over `Q_j`, exponent 3 (rotating relative to the load) or 10/3 (stationary).

`BallBearingStiffness` is defined directly here (not shared with the roller side) — deliberately kept duplicated, unaffected by the capacity migration.

### Roller_Bearing — line contact (`CYLINDRICAL_ROLLER`)

**`roller_bearing_solver.py` — `ISO16281RollerSolver`**

§5.2 lamina-model solver for radial cylindrical roller bearings (NU/N-type, zero nominal contact angle, no axial capacity). Each roller sliced into `n_s` (≥30) laminae, corrected for the roller's logarithmic profile.
- `solve(shaft_system, bearings, library, psi_override=None)` → `RollerLoadDistributionLibrary`. Requires each bearing to carry `Z, Dwe, Lwe, Dpw, phi_j, s, n_s, x_k, cL, cs, alpha_0`; `_check_lamina_count()` validates `n_s >= 30` and `len(x_k) == n_s`.
- Only **δr** is solved (radial force balance, eq.45); **psi is an input**, taken from the FEM shaft slope projection (or `psi_override`), same as `ISO16281BallSolver` — eq.(46) is evaluated afterwards as a diagnostic `Mz`, not a solve constraint.
- Scope: `BearingType.CYLINDRICAL_ROLLER` only.
- **The roller-profile discrepancy flagged in the previous doc pass is resolved**, not just clarified: `_elements()` reads the precomputed `bearing.P_xk` directly — cached on the bearing at `core/` assembly time by each subtype's own `_reference_roller_profile()` (eq.42-44) — and there genuinely is no `roller_profile()` method on this class anymore; this solver never computes the profile itself.
- **`debug_radial_capacity(bearing, Cr, i=1, lambda_v=None, label="")`** — mirrors the ball side's debug helper: sources `Q_ci`/`Q_ce` from `bearing.family.per_element_dynamic_capacity()` and `q_ci`/`q_ce` from `bearing.family.per_lamina_dynamic_capacity()`, never a local reimplementation.

There is no longer a `RollerElementCapacity` dataclass (and no more module constants `_LAMBDA_V_RADIAL`/`_LAMBDA_V_TRUST`) in this file — moved to `core/`, with `lambda_v` now owned by each concrete roller subtype (`CylindricalRollerFamily.LAMBDA_V_RADIAL`, `ThrustCylindricalRollerFamily`/`ThrustNeedleRollerFamily.LAMBDA_V_THRUST`) rather than hardcoded here. This also makes the previous doc pass's flagged `TypeError` bug (`thrust_nonzero_alpha`/`thrust_90deg` constructing the dataclass with a nonexistent `Ca=` keyword) moot — that code no longer exists; the equivalent, correct methods now live in `core/machine_elements/Bearings/families/roller_bearing/thrust/functions/capacity.py` (see [`core/README.md`](../core/README.md#coremachine_elements--bearings), including the still-open `1.038` coefficient discrepancy flagged there, not silently fixed).

**`roller_bearing_results.py` — `RollerLoadDistributionResult`, `RollerLoadDistributionLibrary`**
- `RollerLoadDistributionResult` — `__slots__` container: base fields (`delta_a` always `0.0` — no axial load) plus lamina-model extras `x_k`, `psi_j`, `delta_jk` (Z×n_s), `q_jk` (Z×n_s).
- `RollerLoadDistributionLibrary` — same shape as `BallLoadDistributionLibrary`.

**`roller_bearing_postprocessing.py`**
- `Q_j(bearing, result)` — per-roller total force, `sum_k q_jk`.
- `phi_j_global(bearing, result)`, `contact_distribution(bearing, result, frame=...)` — mirror the ball side.
- `lamina_distribution(bearing, result, j)` → (n_s×2) `[x_k, q_j,k]` for a single roller `j`.
- `RollerBearingStiffness.from_result(...)` / `bearing_stiffness(bearing, result, Fr_xz, Fr_xy, eps=1e-9)` — same secant-projection shape as the ball side, deliberately duplicated. `Fa` is always fixed at `0.0`, so `Ka` comes back `inf` / `"no_load"` — the physically correct statement for a radial roller bearing.
- `stress_riser_factor(n_s)` — §5.3.3 eq.(60) approximation for the edge-stress concentration factor. Validity conditions (medium load, misalignment < 4′, roller-profile-shaped profile) are **not checked**.
- `LaminaDynamicEquivalentLoad.from_distribution(bearing, result, inner_rotating=True, outer_rotating=False, label="")` — §5.3.4 eq.(61)-(64), **per lamina** (`q_kei`, `q_kee` are `(n_s,)` arrays). Compare against `bearing.family.per_lamina_dynamic_capacity()`'s `q_ci`/`q_ce`, not `Q_ci`/`Q_ce`. Stops at eq.(64) — combining across laminae into a bearing rating life is §5.4, not implemented.

`RollerBearingStiffness` duplicates `BallBearingStiffness` rather than sharing it — unaffected by the capacity migration, same as the ball side.

---

## solvers — Gears

**`gears/geometry.py` — `GearSolver`**
Cylindrical gear geometry and mesh force calculation (MAAG / ISO 21771 / Shigley §13-7). Pure functions, no internal state.
- `compute_geometry(mn, z1, z2, alpha_n_deg, beta_deg, al, x1, x2, b)` — full pair geometry with optional profile shift; working centre distance from `al` or the involute equation (`brentq`); tip/root/working diameters; contact ratios. Returns `GearGeometryResult`.
- `compute_forces(T1_Nm, geometry)` — Ft, Fr, Fa from input torque. Returns `GearForceResult`.
- `to_gear_element(position, forces, geometry)` — assembles a `GearElement` for injection into the pipeline.

**`gears/utils.py`** — geometry helpers: rack constants (HAP, HFP), `solve_alpha_tw` (involute equation), `contact_ratio_alpha`/`contact_ratio_beta`, `undercut_z_min`, `validate_geometry_inputs`.

Result containers `GearGeometryResult` / `GearForceResult` are documented in [`models/README.md`](../models/README.md).