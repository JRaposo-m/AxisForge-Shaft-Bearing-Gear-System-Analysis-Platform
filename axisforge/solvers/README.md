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

> **Pending re-verification (2026-08-17):** this package is being actively restructured — faster than this document could be re-checked against source in one pass. The `Ball_Bearing/` entries below are confirmed against the current file. Everything under "Not yet re-verified" further down was last checked before `ball_bearing.py` picked up its own `ball_bearing_results.py` split, so it likely still describes the previous shape (e.g. a shared `LoadDistributionResult` returned as a bare dict) rather than the current one. Treat that part as a lead, not a fact, until someone checks it against the file and removes this note.

The bearings solver package is split by contact type, mirroring the `core/machine_elements/Bearings/subtypes` split (see [`core/README.md`](../core/README.md#coremachine_elements--bearings)), with a shared library of generic utilities and a type-dispatching orchestrator on top:

```
ISO_16281/
├── library.py                          ← generic utilities, cross-type results registry
├── rolling_bearing_solver.py           ← RollingBearingSolver — per-BearingType dispatch
├── Ball_Bearing/
│   ├── ball_bearing.py                 ← ISO16281BallSolver, RollingElementCapacity
│   ├── ball_bearing_results.py         ← BallLoadDistributionResult, BallLoadDistributionLibrary
│   └── ball_bearing_postprocessing.py  ← contact_distribution, bearing_stiffness,
│                                          DynamicEquivalentRollingElementLoad
└── Roller_Bearing/
    ├── roller_bearing.py               ← ISO16281RollerSolver, RollerLoadDistributionResult,
    │                                      RollerElementCapacity
    └── roller_bearing_postprocessing.py← Q_j, bearing_stiffness, LaminaDynamicEquivalentLoad
```

### Confirmed current

**`Ball_Bearing/ball_bearing.py` — `ISO16281BallSolver`**

Coupled shaft–bearing solver for point-contact bearings, covering **both** `BearingType.DEEP_GROOVE_BALL` and `BearingType.ANGULAR_CONTACT` — the contact angle α₀ is read off the bearing instance rather than hardcoded, so one class covers both families. Uses the prescribed-ψ formulation: per bearing, a single 2-DOF root solve (δr, δa) in the plane of the resultant radial force, misalignment prescribed from the FEM seat slope projected onto that plane.
- `solve(shaft_system, bearings, library, psi_override=None)` → a **`BallLoadDistributionLibrary`** (not a bare dict — see below).
- `minimum_axial_load(bearing, Fr_xz, Fr_xy, psi, delta_r_init=0.0, delta_a_init=0.0, Fa_bracket=(0.0, 5e4), xtol=1e-6)` → `(Fa_min, BallLoadDistributionResult)` — smallest axial preload such that δa ≥ 0, via `brentq`.
- `debug_radial_capacity(bearing, Cr, i=1, label="")` — module-level function, prints the full eq.(19)/(20) intermediate breakdown for a radial capacity check; not used by any production path.
- Seeding for the root solve (`_initial_delta_r`, `_initial_delta_a`) trusts a non-negligible FEM hint (from shaft nodal displacements) and only falls back to a Hertz-scale estimate when the hint is ~0 (e.g. first solve of a load case).

**`RollingElementCapacity`** *(frozen dataclass, in `ball_bearing.py`)* — per-element dynamic capacity Q_ci / Q_ce, ISO/TS 16281 §4.3.1. Cr/Ca supplied externally.
- `radial(bearing, Cr, i=1)` — radial ball bearings, eq.(19)/(20).
- `thrust_nonzero_alpha(bearing, Ca)` — thrust ball bearings α≠90°, eq.(21)/(22).
- `thrust_90deg(bearing, Ca)` — thrust ball bearings α=90°, eq.(23)/(24).

**`Ball_Bearing/ball_bearing_results.py`** — new file, split out of `ball_bearing.py`. Holds the point-contact result shape and its registry, kept local to this contact type rather than in the cross-type `library.py`.
- `BallLoadDistributionResult` — one bearing's solve output (δr, δa, ψ, φ(Fr), per-element δ_j/α_j, Mz, solver diagnostics), same field set `solve()`'s internals construct it with.
- `BallLoadDistributionLibrary` — local, single-type registry `solve()` returns; built with `.set(label, result)` per bearing. `rolling_bearing_solver.RollingBearingSolver` is what reads it back out and merges it with the roller side's equivalent — this class itself never touches the cross-type `BearingResultsLibrary` in `library.py`.
- *(Field-by-field detail and `BallLoadDistributionLibrary`'s full method list not yet confirmed — paste the file to fill this in.)*

### Not yet re-verified — likely describes an earlier version

Everything below matches what was true before the `ball_bearing_results.py` split above; it needs the same paste-and-check treatment before it can be trusted.

**`library.py`** — as last checked, held a *shared* `LoadDistributionResult` and `BearingStiffness` used by both contact types, plus `RollingBearingTypeSolver` (Protocol), `check_bearing_ready`, `warn_if_floating_loaded`, `run_root`, and the cross-type `BearingResultBundle`/`BearingResultsLibrary` registry. Given `ball_bearing.py` no longer imports `LoadDistributionResult` from here, this file's Block 1 (`LoadDistributionResult`, `BearingStiffness`) has probably changed shape too — possibly `BearingStiffness` now builds from either per-type result class instead of one shared one.

**`rolling_bearing_solver.py` — `RollingBearingSolver`** — as last checked, grouped bearings by `BearingType`, dispatched each group to its per-type solver, and merged the results into one `{label: LoadDistributionResult}` dict. With the ball side now returning a `BallLoadDistributionLibrary` instead of a dict, this merge step has almost certainly changed — probably now reads labels out of each per-type library object rather than dict-updating.
- `_SOLVER_MAP`: `DEEP_GROOVE_BALL` / `ANGULAR_CONTACT` → `ISO16281BallSolver`; `CYLINDRICAL_ROLLER` → `ISO16281RollerSolver`. An unregistered `BearingType` raises `NotImplementedError` at solve time — this part is architectural and unlikely to have changed.

**`Ball_Bearing/ball_bearing_postprocessing.py`** — as last checked, imported `LoadDistributionResult`/`BearingStiffness` from `library.py`; probably now imports `BallLoadDistributionResult` from `ball_bearing_results.py` instead. Functions: `phi_j_global`, `contact_distribution(frame="global"|"local")`, `bearing_stiffness(...)` (wraps `BearingStiffness.from_result`), and `DynamicEquivalentRollingElementLoad` *(frozen dataclass, Q_ei/Q_ee, §4.3.2 eq.25–28, independent `inner_rotating`/`outer_rotating` flags)`.

**`Roller_Bearing/roller_bearing.py` — `ISO16281RollerSolver`** — as last checked: ISO/TS 16281 §5.2 lamina-model solver for `BearingType.CYLINDRICAL_ROLLER` (zero contact angle, no axial capacity), self-contained from the ball side. Reduces to one unknown δr from the radial force balance eq.(45); ψ projected from FEM slopes or taken from `psi_override`; eq.(46) evaluated after as diagnostic `Mz`. `solve(...)` returned `{label: RollerLoadDistributionResult}` — check whether this mirrored the ball side and moved to a `RollerLoadDistributionLibrary` in its own `roller_bearing_results.py`. `roller_profile(x_k, Dwe, Lwe)` *(static)* — crowning depth, eq.(42)–(44). Scope stays NU/N-type only; tapered/spherical roller need an extra coordinate transform not implemented here.

**`RollerLoadDistributionResult`** *(as last checked, extended the shared `LoadDistributionResult`)* — adds `x_k` (lamina positions), `psi_j` (per-roller misalignment), `delta_jk`/`q_jk` (per-lamina deflection/force, shape (Z, n_s)). If the ball side dropped its dependency on the shared base class, check whether this one still extends it or has been made local too.

**`RollerElementCapacity`** *(frozen dataclass, in `roller_bearing.py`)* — per-roller Q_ci/Q_ce §5.3.1.2 eq.(47)–(48), plus per-lamina q_ci/q_ce §5.3.2 eq.(56)–(57) via `per_lamina()`. `radial(bearing, Cr, i=1, lambda_v=0.83)`.

**`Roller_Bearing/roller_bearing_postprocessing.py`** — mirrors the ball postprocessing file for line contact. `Q_j`, `contact_distribution`, `lamina_distribution(bearing, result, j)`, `bearing_stiffness(...)` (Fa always 0.0 → `Ka_regime` always `"no_load"`), `stress_riser_factor(n_s)` (§5.3.3 eq.60 approximation), and `LaminaDynamicEquivalentLoad` *(frozen dataclass, per-lamina k, not per-roller — §5.3.4 eq.61–64; corrects an earlier per-roller version)*.

---

## solvers — Gears

**`gears/geometry.py` — `GearSolver`**
Cylindrical gear geometry and mesh force calculation (MAAG / ISO 21771 / Shigley §13-7). Pure functions, no internal state.
- `compute_geometry(mn, z1, z2, alpha_n_deg, beta_deg, al, x1, x2, b)` — full pair geometry with optional profile shift; working centre distance from `al` or the involute equation (`brentq`); tip/root/working diameters; contact ratios. Returns `GearGeometryResult`.
- `compute_forces(T1_Nm, geometry)` — Ft, Fr, Fa from input torque. Returns `GearForceResult`.
- `to_gear_element(position, forces, geometry)` — assembles a `GearElement` for injection into the pipeline.

**`gears/utils.py`** — geometry helpers: rack constants (HAP, HFP), `solve_alpha_tw` (involute equation), `contact_ratio_alpha`/`contact_ratio_beta`, `undercut_z_min`, `validate_geometry_inputs`.

Result containers `GearGeometryResult` / `GearForceResult` are documented in [`models/README.md`](../models/README.md).
