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

The bearings solver package is split by contact type, mirroring the `core/machine_elements/Bearings/subtypes` split (see [`core/README.md`](../core/README.md#coremachine_elements--bearings)), with a shared result library and a type-dispatching orchestrator on top:

```
ISO_16281/
├── library.py                          ← shared result shapes, generic utilities, results registry
├── rolling_bearing_solver.py           ← RollingBearingSolver — per-BearingType dispatch
├── Ball_Bearing/
│   ├── ball_bearing.py                 ← ISO16281BallSolver, RollingElementCapacity
│   └── ball_bearing_postprocessing.py  ← contact_distribution, bearing_stiffness,
│                                          DynamicEquivalentRollingElementLoad
└── Roller_Bearing/
    ├── roller_bearing.py               ← ISO16281RollerSolver, RollerLoadDistributionResult,
    │                                      RollerElementCapacity
    └── roller_bearing_postprocessing.py← Q_j, bearing_stiffness, LaminaDynamicEquivalentLoad
```

**`library.py`** — shared data contract and generic utilities used by every per-type solver. Deliberately contains no contact physics — no kinematics, no capacity formulas, no load-deflection exponents; those live entirely inside each contact type's own module. Renamed from `common.py`; now also owns the per-bearing results registry.

| Class / function | Purpose |
|-------|---------|
| `LoadDistributionResult` | Output shape for one bearing's internal load distribution solve, identical regardless of contact type: ring displacements δr/δa, prescribed misalignment ψ, resultant-force angle φ(Fr), per-element deflection δ_j and contact angle α_j, moment reaction Mz, solver diagnostics (iterations, residual, success flag). Contact force per element (`Q_j = cp·δ_j^n`, n contact-type-dependent) is computed by each type solver's own `Q_j()`, not stored here. |
| `BearingStiffness` | Secant stiffness (Kr_xz, Kr_xy, Ka) projected from a converged `LoadDistributionResult` onto the global axes, with an axial engagement regime (`no_load` / `engaged` / `closing_clearance`). Built via `from_result` — the single implementation every per-type `bearing_stiffness()` convenience wraps. Renamed from `BearingStiffnessState`. |
| `RollingBearingTypeSolver` | `typing.Protocol` describing the `solve()` shape `RollingBearingSolver` expects from any per-type solver — satisfied by matching the shape, no subclassing required, keeping ball and roller solvers independent of each other and of the orchestrator. |
| `check_bearing_ready(bearing, label, required_attrs)` | Raises if the bearing hasn't had its type-specific setup called yet. |
| `warn_if_floating_loaded(bearing, label, Fa)` | Warns if a `"floating"` bearing carries a non-zero axial reaction — almost certainly a modelling error upstream. |
| `run_root(fun, x0, tol)` | Shared `scipy.optimize.root` wrapper (`hybr` with `lm` fallback). |
| `BearingResultBundle` | Every computed ISO/TS 16281 result for a single bearing, gathered under its label: `load_distribution`, `stiffness`, `capacity`, `dynamic_equivalent_load` (the latter two typed as plain `object`, since their concrete classes differ by contact type). |
| `BearingResultsLibrary` | Registry of `BearingResultBundle` keyed by bearing label — mirrors `SimpleFEMResultsLibrary`'s usage pattern. `set_load_distribution`/`set_stiffness`/`set_capacity`/`set_dynamic_equivalent_load`, `get(label)`, `labels()`. |

**`rolling_bearing_solver.py` — `RollingBearingSolver`**

Orchestrator for a shaft's full bearing set, which can legitimately mix contact types (e.g. a locating deep-groove ball bearing plus a floating cylindrical roller bearing on the same shaft). Groups the bearings passed to it by `BearingType`, dispatches each group to the registered per-type solver, and merges results back into a single dict — in the caller's original order, regardless of how the groups were split internally. This is the only module in the package aware of more than one contact type; post-processing utilities that bake in contact-type physics (capacity, dynamic equivalent load, per-type `bearing_stiffness`) are **not** re-exposed here — call them on the concrete per-type module for a bearing whose type is already known.

- `solve(shaft_system, bearings, library, psi_override=None)` → `{label: LoadDistributionResult}`.
- `_SOLVER_MAP`: `DEEP_GROOVE_BALL` / `ANGULAR_CONTACT` → `ISO16281BallSolver`; `CYLINDRICAL_ROLLER` → `ISO16281RollerSolver`. An unregistered `BearingType` raises `NotImplementedError` at solve time (fails loudly rather than silently dropping bearings from the results).

**`Ball_Bearing/ball_bearing.py` — `ISO16281BallSolver`** *(renamed from `IterativeBearingFEMSolver`)*

Coupled shaft–bearing solver for point-contact (ball) bearings using the prescribed-ψ formulation. Per bearing it performs a single 2-DOF root solve (δr, δa) in the plane of the resultant radial force, with misalignment prescribed from the FEM seat slope projected onto that plane.
- `solve(shaft_system, bearings, library, psi_override=None)` → `{label: LoadDistributionResult}` for all bearings, reading FEM data from the library via `BearingNodeData`.
- `minimum_axial_load(bearing, ...)` — smallest axial preload Fa_min such that δa ≥ 0, via `brentq`.
- Solver core uses `scipy.optimize.root` (`hybr` with `lm` fallback), via `library.run_root`.

**`RollingElementCapacity`** *(frozen dataclass)* — per-element dynamic capacity Q_ci / Q_ce, ISO/TS 16281 §4.3.1. Cr/Ca are supplied externally.
- `radial(bearing, Cr, i=1)` — radial ball bearings, eq.(19)/(20).
- `thrust_nonzero_alpha(bearing, Ca)` — thrust ball bearings α≠90°, eq.(21)/(22).
- `thrust_90deg(bearing, Ca)` — thrust ball bearings α=90°, eq.(23)/(24).

Module-level helpers `_geometry_bracket` and `_check_geometry` encapsulate the shared §4.3.1 geometry factor and its validity guards.

**`Ball_Bearing/ball_bearing_postprocessing.py`**
- `phi_j_global(bearing, result)` — per-element angular position in the global frame: `(bearing.phi_j + phi_Fr) % 2π`.
- `contact_distribution(bearing, result, frame="global")` — per-element (φ, Q_j) pairs, shape (Z, 2); `frame="local"` keeps φ as stored (debugging).
- `bearing_stiffness(bearing, result, Fr_xz, Fr_xy, Fa)` — thin wrapper around `BearingStiffness.from_result`.
- `DynamicEquivalentRollingElementLoad` *(frozen dataclass)* — dynamic equivalent rolling element loads Q_ei / Q_ee, ISO/TS 16281 §4.3.2, eq.(25)–(28), with independent `inner_rotating`/`outer_rotating` flags (not mutually exclusive, so a rotating-load case can be represented too). Built via `from_distribution`.

**`Roller_Bearing/roller_bearing.py` — `ISO16281RollerSolver`**

ISO/TS 16281 §5.2 lamina-model internal load distribution solver for line-contact (`BearingType.CYLINDRICAL_ROLLER`, zero nominal contact angle, no axial capacity) bearings. Self-contained — does not import or depend on `ISO16281BallSolver`. The internal load distribution reduces to a single unknown, δr, solved from the radial force balance eq.(45); ψ is projected from FEM shaft slopes by default (`psi = psi_xz·cos(phi_Fr) + psi_xy·sin(phi_Fr)`), or taken from `psi_override` when `psi_input=True` — same convention as `ISO16281BallSolver`. Eq.(46) is evaluated afterwards as a diagnostic `Mz`, not a solve constraint.
- `solve(shaft_system, bearings, library, psi_override=None)` → `{label: RollerLoadDistributionResult}`.
- `roller_profile(x_k, Dwe, Lwe)` *(static)* — crowning depth P(x_k) [mm], eq.(42)–(44); two regimes depending on Lwe/Dwe (full-length logarithmic crown vs flat centre with end crowning).
- Scope: `BearingType.CYLINDRICAL_ROLLER` (NU/N-type) only. Tapered and spherical roller bearings share the lamina mechanics but need an additional coordinate transform (cone half-angle / crown-osculation) not implemented here, and are not wired into `RollingBearingSolver`'s dispatch table yet.

**`RollerLoadDistributionResult`** *(extends `LoadDistributionResult`)* — adds lamina-level fields: `x_k` (lamina positions, eq.38), `psi_j` (per-roller local misalignment, eq.39), `delta_jk` (per-lamina elastic deflection, eq.41), `q_jk` (per-lamina contact force, eq.36) — both shape (Z, n_s).

**`RollerElementCapacity`** *(frozen dataclass)* — per-roller dynamic load capacity Q_ci / Q_ce, ISO/TS 16281 §5.3.1.2, eq.(47)–(48), plus the per-lamina dynamic load rating q_ci / q_ce, §5.3.2, eq.(56)–(57).
- `radial(bearing, Cr, i=1, lambda_v=0.83)` — radial roller bearing capacity, eq.(47)–(49).
- `per_lamina(bearing, Q_ci, Q_ce)` — q_ci/q_ce from the whole-roller Q_ci/Q_ce, eq.(56)–(57).

**`Roller_Bearing/roller_bearing_postprocessing.py`** — mirrors `Ball_Bearing/ball_bearing_postprocessing.py` for line contact; all functions here consume an already-converged `RollerLoadDistributionResult`, none run `scipy.optimize`.
- `Q_j(bearing, result)` — total contact force per roller [N], summed over its n_s laminae.
- `contact_distribution`, `lamina_distribution(bearing, result, j)` — per-roller and per-lamina force distributions (the latter for inspecting edge loading on a specific roller).
- `bearing_stiffness(bearing, result, Fr_xz, Fr_xy)` — thin wrapper around `BearingStiffness.from_result`; Fa is always 0.0 (radial roller bearings carry no axial load by design), so `Ka_regime` always comes back `"no_load"`.
- `stress_riser_factor(n_s)` — ISO/TS 16281 §5.3.3 eq.(60) approximation for the edge-stress concentration factor f[k], used when the actual Hertzian contact pressure per lamina is not computed.
- `LaminaDynamicEquivalentLoad` *(frozen dataclass)* — dynamic equivalent load **per lamina** k (arrays over k, not a scalar per bearing), ISO/TS 16281 §5.3.4, eq.(61)–(64), via `from_distribution(bearing, result, inner_rotating=True, outer_rotating=False)`. Corrects an earlier per-roller formulation that used the wrong quantity; compare its `q_kei`/`q_kee` against `RollerElementCapacity`'s per-lamina `q_ci`/`q_ce`, not against the whole-roller `Q_ci`/`Q_ce`. Combining across laminae into a single bearing rating life is ISO/TS 16281 §5.4, not implemented here.

---

## solvers — Gears

**`gears/geometry.py` — `GearSolver`**
Cylindrical gear geometry and mesh force calculation (MAAG / ISO 21771 / Shigley §13-7). Pure functions, no internal state.
- `compute_geometry(mn, z1, z2, alpha_n_deg, beta_deg, al, x1, x2, b)` — full pair geometry with optional profile shift; working centre distance from `al` or the involute equation (`brentq`); tip/root/working diameters; contact ratios. Returns `GearGeometryResult`.
- `compute_forces(T1_Nm, geometry)` — Ft, Fr, Fa from input torque. Returns `GearForceResult`.
- `to_gear_element(position, forces, geometry)` — assembles a `GearElement` for injection into the pipeline.

**`gears/utils.py`** — geometry helpers: rack constants (HAP, HFP), `solve_alpha_tw` (involute equation), `contact_ratio_alpha`/`contact_ratio_beta`, `undercut_z_min`, `validate_geometry_inputs`.

Result containers `GearGeometryResult` / `GearForceResult` are documented in [`models/README.md`](../models/README.md).
