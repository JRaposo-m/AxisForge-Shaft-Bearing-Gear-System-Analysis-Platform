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

> **Verified against source — 2026-08-18.** Both contact-type packages (`Ball_Bearing/`, `Roller_Bearing/`) and the two orchestration files (`library.py`, `rolling_bearing_solver.py`) below are now confirmed against the current files, superseding the previous "pending re-verification" note. The `ball_bearing_results.py` / `roller_bearing_results.py` split it flagged is the confirmed, current shape. Two source-level discrepancies turned up during this pass and are called out inline where they occur, rather than silently corrected — see `RollerElementCapacity.thrust_nonzero_alpha`/`.thrust_90deg` and `ISO16281RollerSolver`'s roller-profile handling below.

The bearings solver package is split by contact type, mirroring the `core/machine_elements/Bearings/subtypes` split (see [`core/README.md`](../core/README.md#coremachine_elements--bearings)), with a shared library of generic utilities and a type-dispatching orchestrator on top:

```
ISO_16281/
├── library.py                          ← generic utilities, cross-type results registry
├── rolling_bearing_solver.py           ← RollingBearingSolver — per-BearingType dispatch
├── Ball_Bearing/
│   ├── ball_bearing_solver.py          ← ISO16281BallSolver, RollingElementCapacity,
│   │                                      debug_radial_capacity
│   ├── ball_bearing_results.py         ← BallLoadDistributionResult, BallLoadDistributionLibrary
│   └── ball_bearing_postprocessing.py  ← Q_j, phi_j_global, contact_distribution,
│                                          BallBearingStiffness, bearing_stiffness,
│                                          DynamicEquivalentRollingElementLoad
└── Roller_Bearing/
    ├── roller_bearing_solver.py           ← ISO16281RollerSolver, RollerElementCapacity,
    │                                         debug_radial_capacity
    ├── roller_bearing_results.py          ← RollerLoadDistributionResult,
    │                                         RollerLoadDistributionLibrary
    └── roller_bearing_postprocessing.py   ← Q_j, phi_j_global, contact_distribution,
                                              lamina_distribution, RollerBearingStiffness,
                                              bearing_stiffness, stress_riser_factor,
                                              LaminaDynamicEquivalentLoad
```

Each contact type owns its full vertical slice — solve, result shape, and postprocessing — and never imports the other. The two packages are wired together at exactly two points: `library.py`'s `BearingResultsLibrary`, which aggregates results from either side under one label without needing to know which side produced them (duck typing, not a shared base class), and `rolling_bearing_solver.py`'s `RollingBearingSolver`, the only module in the package aware that more than one contact type exists.

### `library.py` — generic utilities and the cross-type results registry

Two blocks, both confirmed against the current file:

**Block 1 — generic, contact-agnostic utilities**, used by both per-type solvers:
- `check_bearing_ready(bearing, label, required_attrs)` — raises `RuntimeError` naming whichever of `required_attrs` is still `None` on the bearing, i.e. the type-specific geometry/contact setup hasn't run yet.
- `warn_if_floating_loaded(bearing, label, Fa, eps=FA_FLOATING_EPS)` — warns (doesn't raise) if a bearing marked `arrangement="floating"` is carrying `Fa` above the 1e-6 N floor.
- `run_root(fun, x0, tol)` — `scipy.optimize.root`, `hybr` with an `lm` fallback if `hybr` doesn't converge or converges to a worse residual than `lm`. Returns `(x, nfev, residual_norm, success)`.

**Block 2 — `BearingResultBundle` / `BearingResultsLibrary`**, the per-bearing-label registry every downstream consumer (lubrication, fatigue, ...) is meant to read from:
- `BearingResultBundle` — plain dataclass holding everything computed for one label: `bearing_type`, `load_distribution`, `stiffness`, `capacity`, `dynamic_equivalent_load`, and an open `extra: dict[str, object]` slot for analyses this package doesn't define.
- `BearingResultsLibrary.add_load_distribution_library(bearing_type, local_library)` — the primary way load distribution results enter this registry: an entire per-type local library (`BallLoadDistributionLibrary` or `RollerLoadDistributionLibrary`) is handed over wholesale, by reference, one call per `BearingType` group. `set_load_distribution(label, result, bearing_type=None)` exists for the single-label case but is secondary.
- `set_capacity` / `set_dynamic_equivalent_load` / `set_stiffness` / `set_extra` / `get` / `labels` / `load_distribution_library(bearing_type)` round out the registry. `set_*()` overwrites rather than accumulating history; `get()` raises `KeyError` for a label nothing has been recorded on yet.

`LoadDistributionResult` and `BearingStiffness` are `TYPE_CHECKING`-only unions (`Ball... | Roller...`) here, not classes — there is no shared base class or Protocol either side implements; the two contact types stay independent by duck typing.

### `rolling_bearing_solver.py` — `RollingBearingSolver`

Orchestrates a shaft's full, possibly mixed-type bearing set (e.g. a locating deep-groove ball bearing plus a floating cylindrical roller bearing on the same shaft). This is the only module in the package that imports both `Ball_Bearing` and `Roller_Bearing`.

- `_SOLVER_MAP`: `DEEP_GROOVE_BALL` / `ANGULAR_CONTACT` → `ISO16281BallSolver`; `CYLINDRICAL_ROLLER` → `ISO16281RollerSolver`. An unregistered `BearingType` raises `NotImplementedError` at solve time, naming the offending labels.
- `solve(shaft_system, bearings, library, psi_override=None, results=None)` → `{label: LoadDistributionResult}`, ordered to match the caller's `bearings` dict regardless of internal grouping order. Groups `bearings` by `BearingType`, dispatches each group to its per-type solver, and — if a `BearingResultsLibrary` is passed as `results` — records each group's local library into it via `add_load_distribution_library()`. `psi_override` entries are split per group and forwarded only to the solver that owns that label.
- `postprocess_and_record(shaft_system, bearings, library, catalog, load_distribution=None, results=None, psi_override=None)` → `BearingResultsLibrary`. The full pipeline: `solve()` (or reuse an already-computed `load_distribution` dict) → capacity → dynamic equivalent load → secant stiffness, dispatched per bearing through `_POSTPROC_MAP` (a `_PostprocAdapter` per contact-type family — `capacity_cls`, `derel_cls`, `stiffness_fn` — absorbing the signature mismatch between the ball side's `bearing_stiffness()` (takes `Fa`) and the roller side's (doesn't; radial roller bearings carry no axial load)). `catalog` is `{label: {"capacity": {...}, "dynamic_equivalent_load": {...}}}`; the `"capacity"` sub-dict's `"method"` key defaults to `"radial"` and can be set to `"thrust_nonzero_alpha"` or `"thrust_90deg"` for thrust arrangements, and must include `Cr` or `Ca` as appropriate. Either sub-dict may be omitted per label to skip that step — but **stiffness is computed unconditionally for every bearing in `bearings`**, regardless of what `catalog` contains for that label. Raises `NotImplementedError` if a bearing's type has no entry in `_POSTPROC_MAP`.

### Ball_Bearing — point contact (`DEEP_GROOVE_BALL`, `ANGULAR_CONTACT`)

**`ball_bearing_solver.py` — `ISO16281BallSolver`**

Coupled shaft–bearing solver for point-contact bearings, covering both `BearingType.DEEP_GROOVE_BALL` and `BearingType.ANGULAR_CONTACT` — the contact angle α₀ is read off the bearing instance rather than hardcoded, so one class covers both families. Uses the prescribed-ψ formulation: per bearing, a single 2-DOF root solve (δr, δa) in the plane of the resultant radial force, misalignment prescribed from the FEM seat slope projected onto that plane.
- `solve(shaft_system, bearings, library, psi_override=None)` → `BallLoadDistributionLibrary`.
- `minimum_axial_load(bearing, Fr_xz, Fr_xy, psi, delta_r_init=0.0, delta_a_init=0.0, Fa_bracket=(0.0, 5e4), xtol=1e-6)` → `(Fa_min, BallLoadDistributionResult)` — smallest axial preload such that δa ≥ 0, via `brentq` on δa(Fa).
- `debug_radial_capacity(bearing, Cr, label="")` — module-level, prints the full eq.(19)/(20) intermediate breakdown; not used by any production path.
- Seeding for the root solve (`_initial_delta_r`, `_initial_delta_a`) trusts a non-negligible FEM hint (from shaft nodal displacements) and only falls back to a Hertz-scale estimate when the hint is ~0 (e.g. first solve of a load case).

**`RollingElementCapacity`** *(frozen dataclass, in `ball_bearing_solver.py`)* — per-element dynamic capacity Q_ci / Q_ce, ISO/TS 16281 §4.3.1. Cr/Ca supplied externally.
- `.radial(bearing, Cr, i=1, label="")` — radial ball bearings, eq.(19)/(20).
- `.thrust_nonzero_alpha(bearing, Ca, label="")` — thrust ball bearings α≠90°, eq.(21)/(22).
- `.thrust_90deg(bearing, Ca, label="")` — thrust ball bearings α=90°, eq.(23)/(24).

**`ball_bearing_results.py` — `BallLoadDistributionResult`, `BallLoadDistributionLibrary`**
- `BallLoadDistributionResult` — `__slots__` container, no dependency on `library.py`: `delta_r`, `delta_a`, `psi`, `phi_Fr`, `delta_j`, `alpha_j`, `Mz`, `n_iter`, `residual`, `ok`. Contact force per element (`Q_j = cp * delta_j^1.5`) is deliberately *not* stored here — it's derived on demand by `ball_bearing_postprocessing.Q_j()`.
- `BallLoadDistributionLibrary` — local, single-type registry: `set`/`get`/`labels`/`__contains__`/`__iter__`/`__len__`. `get()` raises `KeyError` for an unset label. This is what `ISO16281BallSolver.solve()` returns; `RollingBearingSolver` is what reads it back out and hands it to `BearingResultsLibrary.add_load_distribution_library()` — this class never touches the cross-type registry itself.

**`ball_bearing_postprocessing.py`**
- `Q_j(bearing, result)` — per-element contact force, `cp * delta_j^1.5`.
- `phi_j_global(bearing, result)` — ball angular positions in the global frame, `(bearing.phi_j + result.phi_Fr) % 2π`.
- `contact_distribution(bearing, result, frame="global"|"local")` → (Z×2) array `[phi, Q_j]`.
- `BallBearingStiffness.from_result(label, result, Fr_xz, Fr_xy, Fa, eps=1e-9)` — secant stiffness from a converged result, projecting δr onto XZ/XY: `Kr_xz`, `Kr_xy`, `Ka`, and an `Ka_regime` of `"no_load"` / `"engaged"` / `"closing_clearance"`. Any component is `inf` when its displacement is negligible. `bearing_stiffness(bearing, result, Fr_xz, Fr_xy, Fa, eps=1e-9)` is the thin wrapper most callers use.
- `DynamicEquivalentRollingElementLoad.from_distribution(bearing, result, inner_rotating=True, outer_rotating=False, label="")` — §4.3.2 eq.(25)-(28); `Q_ei`/`Q_ee` via a power-mean over `Q_j` with exponent 3 (rotating relative to the load) or 10/3 (stationary). `inner_rotating`/`outer_rotating` are independent flags, not mutually exclusive.

`BallBearingStiffness` is defined directly in this file rather than imported from `library.py` — the secant-stiffness projection is duplicated (not shared) between the ball and roller sides; see the note under `RollerBearingStiffness` below.

### Roller_Bearing — line contact (`CYLINDRICAL_ROLLER`)

**`roller_bearing_solver.py` — `ISO16281RollerSolver`**

§5.2 lamina-model solver for radial cylindrical roller bearings (NU/N-type, zero nominal contact angle, no axial capacity). Each roller is sliced into `n_s` (≥ 30, §5.2.2) identical laminae; deflection and force are resolved per lamina rather than per roller, corrected for the roller's logarithmic profile so a purely cylindrical roller's theoretical edge-stress singularity doesn't appear in the model.

- `solve(shaft_system, bearings, library, psi_override=None)` → `RollerLoadDistributionLibrary`. Requires each bearing to already carry `Z, Dwe, Lwe, Dpw, phi_j, s, n_s, x_k, cL, cs, alpha_0` (`_REQUIRED_ATTRS`) — `x_k`, the lamina midpoints, must be precomputed by the bearing's own geometry setup; `_check_lamina_count()` validates `n_s >= 30` and `len(x_k) == n_s` before solving.
- Only **δr** is solved as an unknown (radial force balance, eq.45); **psi is an input**, not jointly solved with δr — taken from the FEM shaft slope projection (or `psi_override`) exactly as in `ISO16281BallSolver`, to keep the shaft↔bearing coupling one-directional. eq.(46) is evaluated afterwards as a diagnostic `Mz`, not a solve constraint.
- Scope: `BearingType.CYLINDRICAL_ROLLER` only. Tapered/spherical roller need an extra coordinate transform this module doesn't implement, and aren't wired into `RollingBearingSolver`'s dispatch table.
- `debug_radial_capacity(bearing, Cr, i=1, lambda_v=0.83, label="")` — module-level, mirrors the ball side's debug helper.
- **Discrepancy to reconcile:** this file's own module docstring (and the `Roller_Bearing/__init__.py` docstring) describe a `roller_profile()` static method on `ISO16281RollerSolver` implementing the eq.(42)-(44) profile correction, called every iteration by `_elements()`. As currently written, `_elements()` instead reads a precomputed `bearing.P_xk` attribute directly (`# eq.(42)-(44) -- cached on the bearing (Sec 6.2 profile)`) and no `roller_profile()` method is defined on the class. Either the method was factored out onto the bearing's geometry setup and the docstrings weren't updated, or the method exists elsewhere and wasn't part of this review pass — worth a quick check against the file before relying on the docstrings' description.

**`RollerElementCapacity`** *(frozen dataclass, in `roller_bearing_solver.py`)* — §5.3.1.2 eq.(47)-(49) per-roller Q_ci/Q_ce, plus per-lamina q_ci/q_ce (§5.3.2, eq.56-57) via `.per_lamina()`.
- `.radial(bearing, Cr, i=1, lambda_v=0.83, label="")` — eq.(47)/(48). This is the path production code uses today and is confirmed correct.
- `.thrust_nonzero_alpha(bearing, Ca, lambda_v=0.73, label="")` / `.thrust_90deg(bearing, Ca, lambda_v=0.73, label="")` — **known bug:** both classmethods construct the frozen dataclass with an `Ca=` keyword, but `RollerElementCapacity`'s declared fields are `label, Q_ci, Q_ce, q_ci, q_ce, Cr, lambda_v, i` — there is no `Ca` field, and `Cr` (which has no default) is omitted from the call. As written, both raise `TypeError` at call time. Needs a fix (add a `Ca` field, or pass `Cr=None` and rename) before either is usable.

**`roller_bearing_results.py` — `RollerLoadDistributionResult`, `RollerLoadDistributionLibrary`**
- `RollerLoadDistributionResult` — `__slots__` container, fully self-contained (no import from `library.py` or the ball side): the same base fields as `BallLoadDistributionResult` (`delta_a` is always `0.0` — radial roller bearings carry no axial load) plus the lamina-model extras `x_k`, `psi_j`, `delta_jk` (Z×n_s), `q_jk` (Z×n_s).
- `RollerLoadDistributionLibrary` — same shape as `BallLoadDistributionLibrary`: `set`/`get`/`labels`/`__contains__`/`__iter__`/`__len__`.

**`roller_bearing_postprocessing.py`**
- `Q_j(bearing, result)` — per-roller total force, `sum_k q_jk`.
- `phi_j_global(bearing, result)`, `contact_distribution(bearing, result, frame=...)` — mirror the ball side.
- `lamina_distribution(bearing, result, j)` → (n_s×2) `[x_k, q_j,k]` for a single roller `j` — the along-length load profile, e.g. for plotting the most heavily loaded roller.
- `RollerBearingStiffness.from_result(label, result, Fr_xz, Fr_xy, Fa, eps=1e-9)` / `bearing_stiffness(bearing, result, Fr_xz, Fr_xy, eps=1e-9)` — same secant-projection shape as the ball side's `BallBearingStiffness`, deliberately duplicated rather than shared (see below). `bearing_stiffness()` always calls `from_result` with `Fa=0.0` fixed, so `Ka` comes back `inf` with `Ka_regime="no_load"` — the physically correct statement for a radial roller bearing, not a special case.
- `stress_riser_factor(n_s)` — §5.3.3 eq.(60) approximation for the edge-stress concentration factor f[k] (f_i = f_e under this approximation), needing only `k` and `n_s`. The standard's stated validity conditions (medium load, total misalignment < 4′, `roller_profile()`-shaped profile) are **not checked** by this function or its caller.
- `LaminaDynamicEquivalentLoad.from_distribution(bearing, result, inner_rotating=True, outer_rotating=False, label="")` — §5.3.4 eq.(61)-(64), **per lamina** (`q_kei`, `q_kee` are `(n_s,)` arrays, not one scalar per bearing) — corrects an earlier per-roller version of this class that used the wrong quantity. Exponent 4 (rotating relative to the load) or 4.5 (stationary). Because `f_i[k] = f_e[k]` under the eq.(60) approximation, `q_kei` and `q_kee` differ only through `inner_rotating`/`outer_rotating`. Stops at eq.(64); combining across laminae into a single bearing rating life is §5.4, not implemented anywhere in this package.

`RollerBearingStiffness` duplicates `BallBearingStiffness`'s formula rather than sharing it with the ball side or importing a base from `library.py` — consistent with `RollerLoadDistributionResult`/`BallLoadDistributionResult` each being self-contained rather than sharing a base class.

---

## solvers — Gears

**`gears/geometry.py` — `GearSolver`**
Cylindrical gear geometry and mesh force calculation (MAAG / ISO 21771 / Shigley §13-7). Pure functions, no internal state.
- `compute_geometry(mn, z1, z2, alpha_n_deg, beta_deg, al, x1, x2, b)` — full pair geometry with optional profile shift; working centre distance from `al` or the involute equation (`brentq`); tip/root/working diameters; contact ratios. Returns `GearGeometryResult`.
- `compute_forces(T1_Nm, geometry)` — Ft, Fr, Fa from input torque. Returns `GearForceResult`.
- `to_gear_element(position, forces, geometry)` — assembles a `GearElement` for injection into the pipeline.

**`gears/utils.py`** — geometry helpers: rack constants (HAP, HFP), `solve_alpha_tw` (involute equation), `contact_ratio_alpha`/`contact_ratio_beta`, `undercut_z_min`, `validate_geometry_inputs`.

Result containers `GearGeometryResult` / `GearForceResult` are documented in [`models/README.md`](../models/README.md).