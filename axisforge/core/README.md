# axisforge/core — Module Reference

Domain model for the platform: shaft geometry, bearing catalogue + internal geometry, gear geometry, the multi-shaft system graph, gear meshing physics, applied loads, and materials. These classes carry data and geometry — the solvers that consume them live in [`solvers/README.md`](../solvers/README.md).

← back to [project root](../../README.md)

---

## Table of Contents

- [core/machine_elements — Shaft](#coremachine_elements--shaft)
- [core/machine_elements — Bearings](#coremachine_elements--bearings)
- [core/machine_elements — Gears](#coremachine_elements--gears)
- [core/mechanical_system — Systems](#coremechanical_system--systems)
- [core/mechanical_system — Gear Meshing](#coremechanical_system--gear-meshing)
- [core — Loads](#core--loads)
- [core — Materials](#core--materials)

---

## core/machine_elements — Shaft

**`shaft.py`**

| Class | Purpose |
|-------|---------|
| `Shoulder` | Fillet transition geometry between two adjacent sections (step change). |
| `ShaftSection` | Single uniform cylindrical segment with cross-section properties. |
| `Shaft` | Ordered sequence of `ShaftSection` objects forming a complete shaft. |

**`Shoulder`** — dataclass carrying `fillet_radius`, `diameter_large`, `diameter_small`. Validates that the fillet is positive and does not exceed the step height.
- `r_over_d` — r/d ratio (primary Peterson interpolation).
- `D_over_d` — D/d ratio (secondary Peterson interpolation).
- `validate()` — returns a list of geometry errors.

**`ShaftSection`** — dataclass with `length`, `diameter`, `inner_diameter`, `material_id`, `surface_finish_ra`, optional `shoulder_left`/`shoulder_right`, and keyways.
- `area` — cross-sectional area A [mm²].
- `second_moment_of_area` — I = π/64·(d⁴−dᵢ⁴) [mm⁴].
- `polar_moment` — J = π/32·(d⁴−dᵢ⁴) [mm⁴].
- `section_modulus` — W = I/(d/2) [mm³], for σ_b = M/W.
- `polar_section_modulus` — Wt = J/(d/2) [mm³], for τ = T/Wt.
- `validate()` — geometry errors, delegating to shoulders and keyways.

**`Shaft`** — datum x=0 is the left face of the first section; interior boundaries belong to the right section.
- `add_section(section)` — append a section to the right end.
- `total_length`, `n_sections` — aggregate properties.
- `axial_start(index)`, `axial_end(index)` — absolute face positions [mm].
- `section_at(x)` — returns `(ShaftSection, index)` containing position x.
- `diameter_at(x)`, `I_at(x)`, `J_at(x)`, `W_at(x)`, `Wt_at(x)` — section properties at any axial position.
- `shoulders()` — list of `(x_position, Shoulder)` for all steps.
- `validate()` / `validate_or_raise()` — full geometry consistency check, including adjacent shoulder–diameter matching.

---

## core/machine_elements — Bearings

Every bearing is built through one orchestrator, `Bearing.assemble()`, and is immutable from that point on. There is no base-class-with-subclasses hierarchy and no registry/factory dispatch — a "subtype" is a `BearingFamily` instance, passed directly to `assemble()`. Adding a new bearing family means writing a new `BearingFamily` subclass anywhere (in `families/`, or ad hoc in a script) and handing it to `assemble()`; nothing in `core/` needs to change.

```
Bearings/
├── __init__.py          ← Bearing, BearingCatalog, BearingFamily, BearingType
├── bearing.py            ← Bearing -- the orchestrator (assemble(), immutability, validate())
├── bearing_types.py      ← BearingType enum
├── catalog.py             ← BearingCatalog -- generic catalogue data (frozen dataclass)
├── family.py               ← BearingFamily (ABC) -- the pluggable family contract
└── families/
    ├── ball_bearing/
    │   ├── radial/
    │   │   ├── functions/
    │   │   │   ├── capacity.py            ← BearingCapacity, RollingElementCapacity (radial ball)
    │   │   │   └── contact_stiffness.py    ← contact_angle_and_clearance, gamma, raceway_contact_radius,
    │   │   │                                  hertz_spring_constant, SelfAligningContactStiffness
    │   │   └── subtypes/
    │   │       ├── deep_groove_ball.py    ← DeepGrooveBallFamily
    │   │       ├── angular_contact.py     ← AngularContactFamily
    │   │       └── self_aligning.py       ← SelfAligningBallFamily
    │   └── thrust/
    │       ├── functions/
    │       │   ├── capacity.py            ← BearingCapacity, RollingElementCapacity (thrust ball)
    │       │   └── contact_stiffness.py
    │       └── subtypes/
    │           └── thrust_ball.py         ← ThrustBallFamily
    └── roller_bearing/
        ├── radial/
        │   ├── functions/
        │   │   ├── capacity.py            ← BearingCapacity, RollingElementCapacity (radial roller)
        │   │   └── contact_stiffness.py    ← lamina_positions, gamma, line_contact_spring_constant
        │   └── subtypes/
        │       └── cylindrical_roller.py  ← CylindricalRollerFamily
        └── thrust/
            ├── functions/
            │   ├── capacity.py            ← BearingCapacity, RollingElementCapacity (thrust roller)
            │   └── contact_stiffness.py
            └── subtypes/
                ├── thrust_cylindrical_roller.py  ← ThrustCylindricalRollerFamily
                └── thrust_needle_roller.py       ← ThrustNeedleRollerFamily
```

The ISO/TS 16281 *solvers* that consume `Bearing` instances live under `solvers/machine_elements/bearings/ISO_16281/` — see [`solvers/README.md`](../solvers/README.md#solvers--bearings-isots-16281).

**`bearing_types.py` — `BearingType`**

Enum of rolling bearing families: `DEEP_GROOVE_BALL`, `ANGULAR_CONTACT`, `SELF_ALIGNING_BALL`, `THRUST_BALL`, `CYLINDRICAL_ROLLER`, `TAPERED_ROLLER`, `SPHERICAL_ROLLER`, `THRUST_CYLINDRICAL_ROLLER`, `THRUST_NEEDLE_ROLLER`. Mirrored onto `Bearing.bearing_type` at assembly by each family's `BEARING_TYPE` class attribute — kept purely as a label for solver-side dispatch tables (`_SOLVER_MAP`, `_POSTPROC_MAP` in `rolling_bearing_solver.py`), not for any dispatch inside `core/` itself.

**`catalog.py` — `BearingCatalog`**

Frozen dataclass carrying generic, family-agnostic catalogue data: `d`, `D`, `b=0.0`, `C=0.0`, `C0=0.0`, `designation=""`, `label=""`, `position=0.0`, `arrangement="locating"` (`"locating"` / `"floating"` / `"non-locating"`).
- `validate()` — catalogue-level checks only (`position >= 0`, `C >= 0`, `C0 >= 0`, `arrangement` in the allowed set); geometry guards live on the family, not here.
- `validate_or_raise()` — raises `ValueError` if `validate()` returns anything. Called automatically, once, inside `Bearing.assemble()` — a `BearingCatalog` is never separately validated by the caller.

**`family.py` — `BearingFamily` (ABC)**

The contract every bearing family must implement to be pluggable into `Bearing.assemble()`. A family owns exactly two things: how to turn raw geometry inputs into the flat computed attributes solvers read (`assemble_geometry()`), and what it declares itself capable of (`CAPABILITIES`, `REQUIRED_FOR`) so `assemble()` can validate "do I have enough data for the analyses I said I'd run" at assembly time instead of failing deep inside a solver later. A family never runs solvers or computes load-dependent quantities (X, Y, life, ...) — those live entirely in `solvers/`.

- `CAPABILITIES: frozenset[str]` — analysis names this family supports (e.g. `{"point_contact"}`, `{"line_contact"}`).
- `REQUIRED_FOR: dict[str, frozenset[str]]` — analysis name → computed-geometry attribute names that must be non-`None` after `assemble_geometry()`.
- `BEARING_TYPE` — `BearingType` label, mirrored onto `Bearing.bearing_type`.
- `DUTY` — `"radial"` | `"thrust"`, mirrored onto `Bearing.duty`.
- `name` *(abstract property)* — short identifier, e.g. `"deep_groove_ball"`.
- `assemble_geometry(catalog, **geometry_kwargs)` *(abstract)* — pure function of its inputs; returns the `dict[str, Any]` `Bearing.assemble()` mirrors onto the instance via `setattr()`.

**`bearing.py` — `Bearing`**

The orchestrator. A rolling bearing = catalogue data + family-derived geometry, assembled ONCE via `Bearing.assemble()` and immutable from that point on — `__setattr__` raises `AttributeError` on any write once `_assembled` is `True`. This is the single source of truth every solver reads from; it never learns about load cases, X/Y factors, life, or any other analysis result — those belong entirely to whichever solver computes them, keyed by bearing label + load case.

- `Bearing.assemble(family, catalog, geometry, analyses=None)` *(classmethod)* — the ONLY way to build a `Bearing`. Runs `catalog.validate_or_raise()`, checks the requested `analyses` against `family.CAPABILITIES`, calls `family.assemble_geometry(catalog, **geometry)` and mirrors every returned attribute onto the instance, checks `family.REQUIRED_FOR` completeness for each enabled analysis, then seals the instance. Raises `ValueError` (catalogue), `NotImplementedError` (unsupported analysis), or `RuntimeError` (missing geometry field).
- `family` *(property)* — the `BearingFamily` instance this bearing was assembled with; callers dispatch family-specific methods off the bearing itself (`bearing.family.per_element_dynamic_capacity(bearing)`) rather than re-importing the concrete subtype class.
- `is_enabled(analysis)` — `True` if `analysis` was requested at assembly.
- `has_internal_geometry()` — `True` once `assemble()` has completed (there is no partially-assembled state).
- `is_locating()` — `arrangement == "locating"`.
- `validate()` — call-site compatibility with `ShaftSystem.validate()`'s `for e in b.validate()` loop. In normal use this can never actually find anything (`catalog.validate_or_raise()` already ran inside `assemble()`, and the instance is immutable from that point on) — it re-checks the same catalogue-level conditions plus `REQUIRED_FOR` completeness against this instance's own copied attributes, as genuine defense-in-depth (e.g. against `__setattr__` being bypassed via `object.__setattr__`), not an unconditional `[]`.
- `summary()` / `__repr__()`.

```python
from axisforge.core.machine_elements.Bearings import Bearing, BearingCatalog
from axisforge.core.machine_elements.Bearings.families.ball_bearing.radial.subtypes.deep_groove_ball import (
    DeepGrooveBallFamily,
)

bearing = Bearing.assemble(
    family=DeepGrooveBallFamily(),
    catalog=BearingCatalog(d=20, D=47, b=14, C=12700, C0=6550,
                            designation="6204", position=20.0, label="brg1a"),
    geometry=dict(Dw=7.94, Dpw=33.5, Z=8, E=206000, nu=0.3, s=0.010),
    analyses={"point_contact": True},
)
```

### families/ball_bearing/radial — point contact, radial duty

Three subtypes share the same point-contact math (`functions/contact_stiffness.py`: `contact_angle_and_clearance`, `gamma`, `raceway_contact_radius`, `hertz_spring_constant`) and the same eq.(19)-(20) capacity formula (`functions/capacity.py`), differing only in how each is idiomatically specified and in its own ISO 281:2007 Table 1 constants (declared per-subtype, not shared, so one table row can change without affecting another).

**`functions/capacity.py`**
- `BearingCapacity.dynamic(Z, Dw, alpha_0, ri, re, gamma, reduction_factor, i=1)` — Cr [N], Formula (13)/(14), `fc` from Formula (15). `_A1_0089_N` constant corrected to `98.0665` (was ported as `98_066.5`, three decimal places too high, giving `fc` ~1000x above literature — see `_fc()` docstring for the corrected numbers, verified against the 6204 example: `fc ≈ 59.2`, in range against NASA/TP-2016-218937).
- `BearingCapacity.static(...)` — Ca, ISO 76:2006 Sec 5. Still `NotImplementedError` — `f_0` formula text not provided yet.
- `RollingElementCapacity.radial(Z, alpha_0, ri, re, Dw, gamma, Cr, i=1)` *(classmethod)* — `(Q_ci, Q_ce)` [N], ISO/TS 16281 Sec 4.3.1.2 eq.(19)-(20). `Cr` is supplied by the caller, never resolved internally.

**`subtypes/deep_groove_ball.py` — `DeepGrooveBallFamily`**

Specified by clearance (`s`) only — `alpha_0_deg` is not a parameter. `ri`/`re` are never inputs, always `reference_raceway_radii(Dw) = (0.52·Dw, 0.52·Dw)` (ISO 281:2007 Table 1). `REDUCTION_FACTOR_BY_ROWS = {1: 0.95, 2: 0.90}`.
- `assemble_geometry(catalog, Dw, Dpw, Z, E, s, nu=0.3, i=1)` → `ri, re, Dw, Dpw, Z, s, E, nu, A, alpha_0, Ri, phi_j, gamma, cp, i, reduction_factor`.
- `per_element_dynamic_capacity(bearing, Cr=None)` — `(Q_ci, Q_ce)`; `Cr` defaults to `bearing.C`.
- `dynamic_capacity(bearing)` — Cr via `BearingCapacity.dynamic()`.

**`subtypes/angular_contact.py` — `AngularContactFamily`**

Specified by nominal contact angle (`alpha_0_deg`, bounded to `(0, 45]` — above 45° is thrust duty) — not by clearance. Same `reference_raceway_radii` ratios as DGBB (`0.52·Dw`), but `REDUCTION_FACTOR_BY_ROWS = {1: 0.95, 2: 0.95}` (unlike DGBB, lambda doesn't drop for double row).
- `assemble_geometry(catalog, Dw, Dpw, Z, E, alpha_0_deg, nu=0.3, i=1)` — same output fields as DGBB.
- `per_element_dynamic_capacity(bearing, Cr=None)` / `dynamic_capacity(bearing)` — same eq.(19)-(20)/Formula (13)-(15) as DGBB.

**`subtypes/self_aligning.py` — `SelfAligningBallFamily`**

Specified by `alpha_0_deg` (bounded `(0, 45]`), for a structural reason, not just idiom consistency: `re` is *derived from `gamma`* (`re = 0.5·(1/gamma + 1)·Dw`, ISO 281:2007 Table 1, self-aligning row — not a fixed ratio of `Dw` like every other ball subtype), and `gamma` depends on `alpha_0` — so `alpha_0` must be known up front to avoid a circular dependency. `ri = 0.53·Dw`; `REDUCTION_FACTOR_BY_ROWS = {1: 1.0, 2: 1.0}`.
- `assemble_geometry(catalog, Dw, Dpw, Z, E, alpha_0_deg, nu=0.3, i=1)`. Uses `SelfAligningContactStiffness.hertz_spring_constant()` (not the generic `hertz_spring_constant()`) — this subtype's outer race has `F_e(rho)=0` identically, which the general elliptical chi-solve can't handle; `SelfAligningContactStiffness.outer_race_spring_term()` is still stubbed (`NotImplementedError`).
- `per_element_dynamic_capacity(bearing, Cr=None)` — same eq.(19)-(20). No `dynamic_capacity()` yet (depends on the still-stubbed `outer_race_spring_term()`).

### families/ball_bearing/thrust — point contact, thrust duty

**`functions/capacity.py`**
- `BearingCapacity.dynamic()` / `.static()` — Ca, ISO 281:2007 Formula (20)/(21) (eta-based, not the radial side's `fc`). Both still `NotImplementedError` — formula text not provided yet.
- `RollingElementCapacity.thrust_nonzero_alpha(Z, alpha_0, ri, re, Dw, gamma, Ca)` — ISO/TS 16281 Sec 4.3.1.3 eq.(21)-(22).
- `RollingElementCapacity.thrust_90deg(Z, ri, re, Dw, Ca)` — ISO/TS 16281 Sec 4.3.1.4 eq.(23)-(24); at 90°, `gamma=0` and the geometry bracket reduces to the groove radii ratio alone.

**`subtypes/thrust_ball.py` — `ThrustBallFamily`**

Specified by `alpha_0_deg`, bounded `(45, 90]` (mirrors `AngularContactFamily`'s `(0, 45]` from the other side). ISO 281:2007 Table 1, Table No. 4: `ri = re = 0.535·Dw`, `REDUCTION_FACTOR = 0.90` (single scalar, independent of row count — no row-count column in this table row, so `i >= 1` unrestricted, unlike DGBB/ACB). `eta(alpha_0) = 1 - sin(alpha_0)/3` recorded (Table 1 eta column) but not yet wired into `BearingCapacity.dynamic()`.
- `assemble_geometry(catalog, Dw, Dpw, Z, E, alpha_0_deg, nu=0.3, i=1)`.
- `per_element_dynamic_capacity(bearing, Ca=None)` — dispatches on `bearing.alpha_0`: exactly 90° → `thrust_90deg()`, anything else → `thrust_nonzero_alpha()`. `Ca` defaults to `bearing.C`.

### families/roller_bearing/radial — line contact, radial duty

**`functions/contact_stiffness.py`** — `lamina_positions(Lwe, n_s)`, `gamma(Dwe, Dpw, alpha_0)`, `line_contact_spring_constant(Lwe, n_s)` → `(cL, cs)`.

**`functions/capacity.py`**
- `BearingCapacity.dynamic()` / `.static()` — Cr/Ca, ISO 281 Sec 6.2 / ISO 76 Sec 6. Both `NotImplementedError` — `fc`/`f_0` tables not provided yet.
- `RollingElementCapacity.radial(Z, alpha_0, gamma, Cr, lambda_v, i=1)` *(classmethod)* — `(Q_ci, Q_ce)` [N], whole-roller, ISO/TS 16281 Sec 5.3.1.2 eq.(47)-(48). `lambda_v` has no default here — always the subtype's own table value (`CylindricalRollerFamily.LAMBDA_V_RADIAL`).
- `RollingElementCapacity.per_lamina(Q_ci, Q_ce, n_s)` — `(q_ci, q_ce)` [N] per-lamina, eq.(56)-(57).

**`subtypes/cylindrical_roller.py` — `CylindricalRollerFamily`**

NU/N-type, zero nominal contact angle, no axial capacity. `LAMBDA_V_RADIAL = 0.83` (ISO 281:2007 Table 2, Table No. 7). Rejects `catalog.arrangement == "locating"` at assembly (an NU/N-type bearing has no flange to react axial load) — only `"floating"`/`"non-locating"` are valid.
- `assemble_geometry(catalog, Dwe, Lwe, Dpw, Z, s, n_s=30, alpha_0_deg=0.0)` → `Dwe, Lwe, Dpw, Z, s, n_s, alpha_0, x_k, phi_j, gamma, cL, cs, P_xk`. `n_s >= 30` enforced (ISO/TS 16281 Sec 5.2.2).
- `_reference_roller_profile(x_k, Dwe, Lwe)` — the ISO/TS 16281 Sec 6.2 eq.(42)-(44) reference roller profile; cached onto `bearing.P_xk` at assembly, not recomputed per solve iteration. Lives on the subtype (not `functions/`) because a future tapered/spherical roller family needs a different profile formula.
- `per_element_dynamic_capacity(bearing, Cr=None, i=1, lambda_v=None)` — `(Q_ci, Q_ce)`, whole-roller. `Cr` defaults to `bearing.C`; `lambda_v` defaults to `LAMBDA_V_RADIAL`.
- `per_lamina_dynamic_capacity(bearing, Q_ci, Q_ce)` — `(q_ci, q_ce)`, per-lamina, eq.(56)-(57).

### families/roller_bearing/thrust — line contact, thrust duty

**`functions/capacity.py`**
- `BearingCapacity.dynamic()` / `.static()` — Ca, ISO 281:2007 (eta-based, Table 2 Table No. 10: `eta = 1 - 0.15·sin(alpha)`). Both `NotImplementedError`.
- `RollingElementCapacity.thrust_nonzero_alpha(Z, alpha_0, gamma, Ca, lambda_v)`. **Flagged, not silently fixed:** this method's `base` term does NOT carry the leading `1.038` coefficient that the radial side's eq.(47)-(48) `base` does — ported exactly as given from the source draft; looks like a possible typo/omission, needs confirming against the actual ISO/TS 16281 clause before trusting it.
- `RollingElementCapacity.thrust_90deg(Z, Ca, lambda_v)` — `Q_ci == Q_ce` here (bracket collapses to a fixed `2**(2/9)` constant, no geometry dependence left).
- `RollingElementCapacity.per_lamina(Q_ci, Q_ce, n_s)` — same formula as the radial side, duplicated for self-containment.

**`subtypes/thrust_cylindrical_roller.py` — `ThrustCylindricalRollerFamily`** / **`subtypes/thrust_needle_roller.py` — `ThrustNeedleRollerFamily`**

Structurally identical: `ALPHA_0_DEG = 90.0` fixed (flat-race, pure axial — flagged assumption, not confirmed against a pasted source for a variable-angle case), `LAMBDA_V_THRUST = 0.73` (ISO 281:2007 Table 2, Table No. 10, "Thrust roller bearings" — not split by roller shape). Same reference roller profile as the radial side (confirmed identical, same eq.(42)-(44)), duplicated rather than imported cross-duty. Kept as separate classes (not folded together) since needle-roller-specific geometry constraints may diverge later.
- `assemble_geometry(catalog, Dwe, Lwe, Dpw, Z, s, n_s=30)`.
- `per_element_dynamic_capacity(bearing, Ca=None, lambda_v=None)` *(classmethod)* — always dispatches to `thrust_90deg()` (alpha_0 fixed at 90°).
- `per_lamina_dynamic_capacity(bearing, Q_ci, Q_ce)`.

Consumed by `ISO16281RollerSolver` (`solvers/machine_elements/bearings/ISO_16281/Roller_Bearing/roller_bearing_solver.py`, currently wired for `BearingType.CYLINDRICAL_ROLLER` only — the thrust roller subtypes have core-level capacity support but no internal load-distribution solver written yet), dispatched through `RollingBearingSolver` alongside any ball-family bearings on the same shaft. `_elements()` there reads the cached `bearing.P_xk` directly instead of recomputing the profile every root-solve iteration.

---

## core/machine_elements — Gears

**`spur_helical_gear.py`**

**`SpurHelicalGear`** — geometry model for an external spur (β=0) or helical (β>0) gear. Carries module `mn`, teeth `z`, profile shift `x`, pressure angle `alpha_n`, helix `beta`, face width `b`, surface finish, and axial `position`. Exposes reference/base/tip geometry and the attribute interface consumed by the meshing classes.

**`internal_gear.py`**

**`InternalGear`** — ring (internal) gear model following the conventional KHK positive-z definition. Provides mesh-pair interference checks (`validate_mesh`) used by `InternalGearMeshing`.

---

## core/mechanical_system — Systems

**`shaft_system.py`**

**`GearElement`** — thin wrapper binding a gear geometry object to a kinematic role on a shaft; carries no geometry of its own.
- Constructor: `gear`, `role` ("driver"/"driven"), `rotation_dir` (±1, set only for the source gear), `label`.
- `position` — delegates to the underlying gear.
- `validate()` / `validate_or_raise()`.

**`ShaftSystem`** — autonomous single-shaft container of bearings, gears and loads. Has no knowledge of other shafts; its `shaft_position` (global y,z offset) is normally set by the gear system during resolve.
- `add_bearing(b)`, `add_gear(g)`, `add_load(ld)` — placement with axial-bounds checking (chainable).
- `set_gear_loads(loads)` — idempotent replacement of all `gear_mesh`-sourced loads (safe to re-resolve).
- Sorted accessors: `bearings`, `gears`, `loads`, `support_positions`.
- Type-filtered accessors: `radial_loads`, `axial_loads`, `torque_loads`, `external_moments`, `distributed_radial_loads`.
- `gear_extent(ge)`, `bearing_extent(b)` — axial [lo, hi] footprint (position is the centre of face/bearing width).
- `validate()` / `validate_or_raise()`.

**`SpurHelical_gear_system.py`**

**`SpurHelicalMeshLink`** — one directed mesh, shaft_a (driver) → shaft_b (driven), carrying a meshing model and the global line-of-centres angle `phi_deg`. Optional `torque_split` selects fan-out mode; `distribute_loads` toggles distributed vs point gear loads.
- `validate()` — self-mesh, torque_split range, and meshing-interface checks.

**`SpurHelicalGearSystem`** — single-source DAG of shafts connected by mesh links. Enforces exactly one source, no convergent merges, acyclicity, and fan-out torque_split consistency.
- `resolve(P, rpm, rotation_dir_source, source_position)` — propagates power from the source through the DAG in topological order, computing torque, rotation sense and shaft placement, then injects the resulting mesh loads onto every shaft. Torque propagates in N·m.
- `validate()` / `validate_or_raise()` — topology plus per-shaft validation.
- Internal graph helpers: incoming-count, source detection, Kahn topological order, driver grouping, axial-alignment checks.

---

## core/mechanical_system — Gear Meshing

**`spur_helical_gear_meshing.py`**

**`SpurHelicalGearMeshing`** — external spur/helical pair. Computes working centre distance and transverse working pressure angle, contact ratios (εα, εβ, εγ), interference checks, and mesh forces.
- Constructor accepts optional working centre distance `al`, profile-shift equalisation (`equalise_gs`), and addendum reduction.
- `forces(T_in, phi_deg, rotation_dir_in)` — returns Ft, Fr, Fa and per-side load angles, plus output torque and rotation sense.
- `al`, `u` — working centre distance and gear ratio, consumed by the gear system.
- `correction_for_gs_equilibrium()` — Henriot method for profile-shift balancing.

**`internal_gear_meshing.py`**

**`InternalGearMeshing`** — external + internal (ring) pair, using the difference form (z2−z1, x2−x1) throughout. Computes working geometry, path-of-contact points, Ohlendorf loss factor, contact ratios and mesh forces.
- `forces(T_in, phi_deg, rotation_dir_in)` — same interface as the external pair; note internal meshing does **not** reverse rotation sense.
- `gear_geometry(addendum_reduction)` — builds independent working-geometry copies of both gears.
- `validate()` / `validate_or_raise()`, `summary()`.

**`planetary_gear_meshing.py`**

**`PlanetaryGearMeshing`** — single-stage epicyclic train (sun + k planets + ring + carrier) by composition of one external pair (sun–planet) and one internal pair (planet–ring). Owns train-specific logic only.
- `planet_phi_deg(j)`, `planet_position(j)` — angular placement of planet j.
- Structural conditions (coaxiality, assembly, neighbouring), Willis kinematics (F=1 and F=2 modes), and ideal torque distribution over members and planets.

**`PlanetaryKinematics`** — frozen result of the Willis kinematic solve (ω per member in rad/s, operating-mode DoF, and Willis residual for numerical transparency).

**`PlanetaryMember`** / **`MeshTag`** — enums tagging train members (sun/planet/ring/carrier) and tooth contacts.

---

## core — Loads

**`loads.py`**

| Class | Purpose |
|-------|---------|
| `LoadPlane` | Enum of principal bending planes (XY, XZ) used as a decomposition key. |
| `Load` | Base class carrying `position`, `label`, `source` ("user"/"gear_mesh"/"bearing_reaction"). |
| `RadialLoad` | Transverse point force at angular position θ, decomposed into Fy/Fz. |
| `AxialLoad` | Force along the shaft axis (+X tensile). |
| `TorqueLoad` | Torque about the shaft axis [N·m]. |
| `ExternalMoment` | Applied bending moment at orientation θ, decomposed into My/Mz. |
| `DistributedRadialLoad` | Transverse load distributed over [x_lo, x_hi] with uniform or callable intensity/direction. |
| `LoadingProfile` | Fatigue cycle decomposition (stress ratio R → mean/amplitude factors). |

`RadialLoad`/`ExternalMoment` expose `Fy`/`Fz` (or `My`/`Mz`) and `component(plane)`. `DistributedRadialLoad` provides `resultant_component`, `centroid`, `bending_moment_contribution` (closed-form for uniform loads, quadrature otherwise), and `as_point_load` for the constant-θ case.

---

## core — Materials

**`materials.py`**

| Class | Purpose |
|-------|---------|
| `Material` | Shaft/structural material (Shigley-based): Sut, Sy, E, density, Poisson ratio, endurance limit. |
| `GearMaterial` | Gear material (ISO 6336-5): E, ν, ρ, thermal properties, σHlim, σFlim, quality class. |

`Material` exposes `endurance_limit` (Shigley §6-2: 0.5·Sut capped at 700 MPa) and `shear_yield_strength` (0.577·Sy). `GearMaterial` exposes `equivalent_modulus(other)` for the Hertzian reduced modulus of a pair.

Embedded shaft library: `S355`, `CrMo42` (42CrMo4), `AISI_1045`, `AISI_4340`. Embedded gear library: `GEAR_STEEL`, `GEAR_ADI`, `GEAR_POM`, `GEAR_PA66`.

Lookups: `get_material(id)`, `available_materials()`.