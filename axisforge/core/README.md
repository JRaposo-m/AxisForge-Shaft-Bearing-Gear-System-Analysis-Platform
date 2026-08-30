# axisforge/core — Module Reference

Domain model for the platform: shaft geometry, bearing catalogue + family-derived geometry, gear geometry, gear/planetary meshing physics, the multi-shaft system graph, applied loads, and materials. These classes carry data and geometry — the solvers that consume them live in [`solvers/README.md`](../solvers/README.md).

← back to [project root](../../README.md)

---

## Table of Contents

- [Package surface](#package-surface)
- [core/machine_elements — Shaft](#coremachine_elements--shaft)
- [core/machine_elements — Bearings](#coremachine_elements--bearings)
- [core/machine_elements — Gears](#coremachine_elements--gears)
- [core/mechanical_system — Systems](#coremechanical_system--systems)
- [core/mechanical_system — Schematic](#coremechanical_system--schematic)
- [core — Loads](#core--loads)
- [core — Materials](#core--materials)
- [Adjacent: axisforge/database](#adjacent-axisforgedatabase)

---

## Package surface

Directory names are lowercase snake_case throughout (`shaft/`, `bearings/`, `gears/parallel_axis/`, `mechanical_system/parallel_axis/`). Every level declares its exports in `__init__.py` using the lazy roll-up pattern described in the [root README](../../README.md#package-surface--the-__init__py-roll-up-system).

```
core/
├── loads.py
├── materials.py
├── machine_elements/
│   ├── shaft/          __init__ → Shaft, ShaftSection, Shoulder, Keyway, KeywayType
│   ├── bearings/       __init__ → Bearing, BearingCatalog, BearingFamily, BearingType
│   │   └── families/   __init__ → the concrete BearingFamily classes
│   └── gears/
│       └── parallel_axis/  __init__ → SpurHelicalGear, InternalGear,
│                           SpurHelicalGearMeshing, InternalGearMeshing,
│                           PlanetaryGearTrainMeshing
└── mechanical_system/
    └── parallel_axis/
        ├── spur_helical/   __init__ → SpurHelicalMeshLink, SpurHelicalGearSystem,
        │                              GearElement, ShaftSystem
        └── schematic.py
```

The two-level split inside `bearings/` is deliberate: `bearings/__init__.py` exposes only the four orchestration primitives, and `bearings/families/__init__.py` exposes the families. Keeping families out of the parent is what lets a new family be added without editing anything in `core/`.

---

## core/machine_elements — Shaft

**`shaft/shaft.py`** — geometric primitives in the (r, θ, z) convention. `z` is the absolute axial coordinate [mm]; θ [deg] is used only for localized features (keyways) — the shaft body itself is a solid of revolution.

| Class | Purpose |
|-------|---------|
| `KeywayType` | Enum — `PARALLEL` / Woodruff. |
| `Keyway` | Localized stress raiser at absolute `(z_position, theta)`. |
| `Shoulder` | Fillet transition geometry between two adjacent sections. |
| `ShaftSection` | Single uniform cylindrical segment with cross-section properties. |
| `Shaft` | Ordered sequence of `ShaftSection` objects forming a complete shaft. |

**`Keyway`** *(new)* — dataclass with `label`, `z_position`, `width`, `depth`, `length`, `theta=0.0`, `keyway_type`, plus optional tolerance-band fields (`depth_min/max`, `L_min/max`) and pre-resolved fatigue factors (`Kf_bending`, `Kf_torsion`).
- `Keyway.from_standard(label, shaft_diameter, z_position, theta=0.0, keyway_type=..., length=None, series=1)` *(classmethod)* — builds a keyway from the catalogue tables in `axisforge/database/shaft/keyway/` (DIN 6885 parallel, ISO 3912 Woodruff). Raises `NotImplementedError` for a type with no table wired.
- `validate()` — geometry errors.

**`Shoulder`** — dataclass carrying `fillet_radius`, `diameter_large`, `diameter_small`. Validates that the fillet is positive and does not exceed the step height.
- `r_over_d` — r/d ratio (primary Peterson interpolation).
- `D_over_d` — D/d ratio (secondary Peterson interpolation).
- `validate()`.

**`ShaftSection`** — dataclass with `length`, `diameter`, `inner_diameter=0.0`, `material_id="S355"`, `surface_finish_ra=0.8`, optional `shoulder_left`/`shoulder_right`, and `keyways: list[Keyway]`.
- `is_hollow`, `radius`, `area` — A [mm²].
- `second_moment_of_area` — I = π/64·(d⁴−dᵢ⁴) [mm⁴].
- `polar_moment` — J = π/32·(d⁴−dᵢ⁴) = 2I [mm⁴].
- `section_modulus` — W = I/(d/2) [mm³], for σ_b = M/W.
- `polar_section_modulus` — Wt = J/(d/2) [mm³], for τ = T/Wt.
- `validate()` — delegates to shoulders and keyways.

**`Shaft`** — datum z=0 is the left face of `sections[0]`; interior boundaries belong to the right section.
- `add_section(section)`, `total_length`, `n_sections`.
- `axial_start(index)`, `axial_end(index)` — absolute face positions [mm].
- `section_at(z)` → `(ShaftSection, index)`.
- `diameter_at(z)`, `I_at(z)`, `J_at(z)`, `W_at(z)`, `Wt_at(z)`.
- `shoulders()` — list of `(z_position, Shoulder)`; reports only the `shoulder_right` of each section, so a step is never counted twice.
- `validate()` / `validate_or_raise()` — full consistency check, including adjacent shoulder–diameter matching.

---

## core/machine_elements — Bearings

Every bearing is built through one orchestrator, `Bearing.assemble()`, and is immutable from that point on. There is no base-class-with-subclasses hierarchy and no registry/factory dispatch — a "subtype" is a `BearingFamily` instance, passed directly to `assemble()`. Adding a new bearing family means writing a new `BearingFamily` subclass anywhere (in `families/`, or ad hoc in a script) and handing it to `assemble()`; nothing in `core/` needs to change.

```
bearings/
├── __init__.py            ← Bearing, BearingCatalog, BearingFamily, BearingType
├── bearing.py             ← Bearing — the orchestrator (assemble(), immutability, validate())
├── bearing_types.py       ← BearingType enum
├── catalog.py             ← BearingCatalog — generic catalogue data (frozen dataclass)
├── family.py              ← BearingFamily (ABC) — the pluggable family contract
└── families/
    ├── __init__.py        ← roll-up of every family; the table capabilities/ reads
    ├── ball_bearing/
    │   ├── radial/
    │   │   ├── functions/
    │   │   │   ├── capacity.py           ← BearingCapacity, RollingElementCapacity (radial ball)
    │   │   │   └── contact_stiffness.py  ← contact_angle_and_clearance, gamma,
    │   │   │                                raceway_contact_radius, curvature_*,
    │   │   │                                hertz_spring_constant, SelfAligningContactStiffness
    │   │   └── subtypes/
    │   │       ├── deep_groove.py        ← DeepGrooveBallFamily
    │   │       ├── angular_contact.py    ← AngularContactFamily
    │   │       └── self_aligning.py      ← SelfAligningBallFamily
    │   └── thrust/
    │       ├── functions/
    │       │   ├── capacity.py           ← BearingCapacity, RollingElementCapacity (thrust ball)
    │       │   └── contact_stiffness.py
    │       └── subtypes/
    │           ├── thrust_single.py      ← SingleRowThrustBallFamily
    │           └── thrust_multirow.py    ← MultiRowThrustBallFamily
    └── roller_bearing/
        ├── radial/
        │   ├── functions/
        │   │   ├── capacity.py           ← BearingCapacity, RollingElementCapacity (radial roller)
        │   │   └── contact_stiffness.py  ← lamina_positions, gamma, line_contact_spring_constant
        │   └── subtypes/
        │       └── cylindrical_roller.py ← CylindricalRollerFamily
        └── thrust/
            ├── functions/
            │   ├── capacity.py           ← BearingCapacity, RollingElementCapacity (thrust roller)
            │   └── contact_stiffness.py
            └── subtypes/
                ├── cylindrical_single.py    ← ThrustCylindricalRollerFamily
                ├── cylindrical_multirow.py  ← MultiRowThrustCylindricalRollerFamily
                └── needle_single.py         ← ThrustNeedleRollerFamily
```

The ISO/TS 16281 *solvers* that consume `Bearing` instances live under `solvers/machine_elements/bearings/ISO_16281/` — see [`solvers/README.md`](../solvers/README.md#solvers--bearings-isots-16281).

> **Roll-up caveats.** `families/__init__.py`'s `__all__` omits `MultiRowThrustCylindricalRollerFamily`, and `roller_bearing/thrust/subtypes/__init__.py` lazily resolves that name to `.cylindrical_single` instead of `.cylindrical_multirow`. `ball_bearing/thrust/__init__.py` forwards to `.thrust_single`/`.thrust_multirow` instead of `.subtypes`. Fix these before `fixtures/capabilities/` reads the tables — see the [root README](../../README.md#known-inconsistencies-in-the-current-roll-ups).

**`bearing_types.py` — `BearingType`**

Enum of rolling bearing families: `DEEP_GROOVE_BALL`, `ANGULAR_CONTACT`, `SELF_ALIGNING_BALL`, `THRUST_BALL`, `CYLINDRICAL_ROLLER`, `TAPERED_ROLLER`, `SPHERICAL_ROLLER`, `THRUST_CYLINDRICAL_ROLLER`, `THRUST_NEEDLE_ROLLER`. Mirrored onto `Bearing.bearing_type` at assembly by each family's `BEARING_TYPE` class attribute. **Informational label only** — since the solver side moved to `ISO_16281/dispatch.py` (capability + required-attrs), nothing dispatches on this enum any more. It survives so reports and the GUI can group/filter bearings without importing every family class. Extend it for a genuinely new *family* (a new life exponent, a new ISO 281 `p`), not for every subtype variant.

**`catalog.py` — `BearingCatalog`**

Frozen dataclass carrying generic, family-agnostic catalogue data: `d`, `D`, `b=0.0`, `C=0.0`, `C0=0.0`, `designation=""`, `label=""`, `position=0.0`, `arrangement="locating"` (`"locating"` / `"floating"` / `"non-locating"`).
- `validate()` — catalogue-level checks only (`position >= 0`, `C >= 0`, `C0 >= 0`, `arrangement` in the allowed set); geometry guards live on the family.
- `validate_or_raise()` — called automatically, once, inside `Bearing.assemble()`; a `BearingCatalog` is never separately validated by the caller.

**`family.py` — `BearingFamily` (ABC)**

The contract every bearing family must implement to be pluggable into `Bearing.assemble()`. A family owns exactly two things: how to turn raw geometry inputs into the flat computed attributes solvers read (`assemble_geometry()`), and what it declares itself capable of (`CAPABILITIES`, `REQUIRED_FOR`) so `assemble()` can validate "do I have enough data for the analyses I said I'd run" at assembly time instead of failing deep inside a solver later. A family never runs solvers or computes load-dependent quantities (X, Y, life, ...) — those live entirely in `solvers/`.

- `CAPABILITIES: frozenset[str]` — analysis names this family supports (e.g. `{"point_contact"}`, `{"line_contact"}`, `{"multirow_capacity"}`).
- `REQUIRED_FOR: dict[str, frozenset[str]]` — analysis name → computed-geometry attribute names that must be non-`None` after `assemble_geometry()`.
- `BEARING_TYPE` — `BearingType` label, mirrored onto `Bearing.bearing_type`.
- `DUTY` — `"radial"` | `"thrust"`, mirrored onto `Bearing.duty`.
- `name` *(abstract property)* — short identifier, e.g. `"deep_groove_ball"`.
- `assemble_geometry(catalog, **geometry_kwargs)` *(abstract)* — pure function of its inputs; returns the `dict[str, Any]` `Bearing.assemble()` mirrors onto the instance via `setattr()`.

`CAPABILITIES` / `REQUIRED_FOR` are now load-bearing in two more places than before: `ISO_16281/dispatch.py` resolves *which solver* a bearing gets from exactly these declarations, and `fixtures/capabilities/` is specified to read them to build its analysis menu.

**`bearing.py` — `Bearing`**

The orchestrator. A rolling bearing = catalogue data + family-derived geometry, assembled ONCE via `Bearing.assemble()` and immutable from that point on — `__setattr__` raises `AttributeError` on any write once `_assembled` is `True`. This is the single source of truth every solver reads from; it never learns about load cases, X/Y factors, life, or any other analysis result — those belong entirely to whichever solver computes them, keyed by bearing label + load case.

- `Bearing.assemble(family, catalog, geometry, analyses=None)` *(classmethod)* — the ONLY way to build a `Bearing`. Runs `catalog.validate_or_raise()`, checks the requested `analyses` against `family.CAPABILITIES`, calls `family.assemble_geometry(catalog, **geometry)` and mirrors every returned attribute onto the instance, checks `family.REQUIRED_FOR` completeness for each enabled analysis, then seals the instance. Raises `ValueError` (catalogue), `NotImplementedError` (unsupported analysis), or `RuntimeError` (missing geometry field).
- Catalogue attributes mirrored at construction: `d`, `D`, `b`, `C`, `C0`, `designation`, `label`, `position`, `arrangement`, plus `dm = 0.5·(d + D)`.
- `family` *(property)* — the `BearingFamily` instance this bearing was assembled with; callers dispatch family-specific methods off the bearing itself (`bearing.family.per_element_dynamic_capacity(bearing)`) rather than re-importing the concrete subtype class.
- `is_enabled(analysis)`, `has_internal_geometry()`, `is_locating()`.
- `validate()` — call-site compatibility with `ShaftSystem.validate()`'s `for e in b.validate()` loop; defense-in-depth re-check of catalogue conditions plus `REQUIRED_FOR` completeness, not an unconditional `[]`.
- `summary()` / `__repr__()`.

**Multi-row bearings.** A multi-row family assembles `rows: list[dict]` and `i = len(rows)` instead of flat top-level geometry — the per-row fields only mean something per row. Consumers reach a specific row through `bearing.rows[j][...]`; solvers build lightweight `SimpleNamespace` views over them.

```python
from axisforge.core.machine_elements.bearings import Bearing, BearingCatalog
from axisforge.core.machine_elements.bearings.families import DeepGrooveBallFamily

bearing = Bearing.assemble(
    family=DeepGrooveBallFamily(),
    catalog=BearingCatalog(d=20, D=47, b=14, C=12700, C0=6550,
                           designation="6204", position=20.0, label="brg1a"),
    geometry=dict(Dw=7.94, Dpw=33.5, Z=8, E=206000, nu=0.3, s=0.010),
    analyses={"point_contact": True},
)
```

### families/ball_bearing/radial — point contact, radial duty

Three subtypes share the same point-contact math (`functions/contact_stiffness.py`: `contact_angle_and_clearance`, `gamma`, `raceway_contact_radius`, `curvature_sum_inner/outer`, `curvature_diff_inner/outer`, `hertz_spring_constant`) and the same eq.(19)-(20) capacity formula (`functions/capacity.py`), differing only in how each is idiomatically specified and in its own ISO 281:2007 Table 1 constants (declared per-subtype, not shared, so one table row can change without affecting another).

**`functions/capacity.py`**
- `BearingCapacity.dynamic(Z, Dw, alpha_0, ri, re, gamma, reduction_factor, i=1)` — Cr [N], Formula (13)/(14), `fc` from Formula (15). `_A1_0089_N` constant corrected to `98.0665` (was ported as `98_066.5`, three decimal places too high, giving `fc` ~1000× above literature — verified against the 6204 example: `fc ≈ 59.2`, in range against NASA/TP-2016-218937). **Not validated end-to-end** — see the `_fc()` docstring.
- `BearingCapacity.static(...)` — Ca0, ISO 76:2006 Sec 5. `NotImplementedError` — `f_0` formula text not provided yet.
- `RollingElementCapacity.radial(Z, alpha_0, ri, re, Dw, gamma, Cr, i=1)` *(classmethod)* — `(Q_ci, Q_ce)` [N], ISO/TS 16281 Sec 4.3.1.2 eq.(19)-(20). `Cr` is supplied by the caller, never resolved internally. Shares `_geometry_bracket()` with the thrust side's `alpha != 90°` branch.

**`subtypes/deep_groove.py` — `DeepGrooveBallFamily`**

Specified by clearance (`s`) only — `alpha_0_deg` is not a parameter. `ri`/`re` are never inputs, always `reference_raceway_radii(Dw) = (0.52·Dw, 0.52·Dw)` (ISO 281:2007 Table 1). `REDUCTION_FACTOR_BY_ROWS = {1: 0.95, 2: 0.90}`.
- `assemble_geometry(catalog, Dw, Dpw, Z, E, s, nu=0.3, i=1)` → `ri, re, Dw, Dpw, Z, s, E, nu, A, alpha_0, Ri, phi_j, gamma, cp, i, reduction_factor`.
- `per_element_dynamic_capacity(bearing, Cr=None)` — `(Q_ci, Q_ce)`; `Cr` defaults to `bearing.C`.
- `dynamic_capacity(bearing)` — Cr via `BearingCapacity.dynamic()`.

**`subtypes/angular_contact.py` — `AngularContactFamily`**

Specified by nominal contact angle (`alpha_0_deg`, bounded to `(0, 45]` — above 45° is thrust duty), not by clearance. Same `reference_raceway_radii` ratios as DGBB (`0.52·Dw`), but `REDUCTION_FACTOR_BY_ROWS = {1: 0.95, 2: 0.95}` (unlike DGBB, lambda doesn't drop for double row).
- `assemble_geometry(catalog, Dw, Dpw, Z, E, alpha_0_deg, nu=0.3, i=1)` — same output fields as DGBB.
- `per_element_dynamic_capacity(bearing, Cr=None)` / `dynamic_capacity(bearing)`.

**`subtypes/self_aligning.py` — `SelfAligningBallFamily`**

Specified by `alpha_0_deg` (bounded `(0, 45]`) for a structural reason, not just idiom consistency: `re` is *derived from `gamma`* — `re_from_gamma(Dw, gamma) = 0.5·(1/gamma + 1)·Dw` (ISO 281:2007 Table 1, self-aligning row, not a fixed ratio of `Dw` like every other ball subtype) — and `gamma` depends on `alpha_0`, so `alpha_0` must be known up front to avoid a circular dependency. `ri = 0.53·Dw`; `REDUCTION_FACTOR_BY_ROWS = {1: 1.0, 2: 1.0}`.
- `assemble_geometry(catalog, Dw, Dpw, Z, E, alpha_0_deg, nu=0.3, i=1)`. Uses `SelfAligningContactStiffness.hertz_spring_constant()` (not the generic one) — this subtype's outer race has `F_e(rho)=0` identically, which the general elliptical χ-solve can't handle.
- `per_element_dynamic_capacity(bearing, Cr=None)`. **No `dynamic_capacity()`** — it depends on `SelfAligningContactStiffness.outer_race_spring_term()`, still `NotImplementedError`.

### families/ball_bearing/thrust — point contact, thrust duty

Now split into a single-row and a multi-row family, because the ISO capacity formulas differ (Sec 6.3 vs Sec 6.4) and because a multi-row bearing has no meaningful flat top-level geometry.

**`functions/capacity.py`** — `BearingCapacity` here follows **ISO 1281-1:2021** (not the radial side's ISO 281 `fc` form):
- `dynamic_nonzero_alpha(Z, Dw, alpha_0, ri, re, gamma, lam, eta)` — Ca [N], Formula (18)/(19), `fc` from Formula (20).
- `dynamic_90deg(Z, Dw, ri, re, gamma, lam, eta)` — Ca [N], Formula (23)/(24), `fc` from Formula (25).
- `dynamic_multirow(Z_rows, Ca_rows)` — Formula (29): `Ca = (ΣZⱼ) · [Σ (Zⱼ/Caⱼ)^(9/2)]^(−2/9)`, the practical form of (26)-(28) after assuming a row's load is proportional to its ball count.
- `static(...)` — `NotImplementedError`.
- `RollingElementCapacity.thrust_nonzero_alpha(Z, alpha_0, ri, re, Dw, gamma, Ca)` — ISO/TS 16281 Sec 4.3.1.3 eq.(21)-(22).
- `RollingElementCapacity.thrust_90deg(Z, ri, re, Dw, Ca)` — Sec 4.3.1.4 eq.(23)-(24); at 90°, `gamma = 0` and the geometry bracket reduces to the groove radii ratio alone.

**`subtypes/thrust_single.py` — `SingleRowThrustBallFamily`**

Specified by `alpha_0_deg`, bounded `(45, 90]` (mirrors `AngularContactFamily`'s `(0, 45]` from the other side). ISO 281:2007 Table 1, Table No. 4: `ri = re = 0.535·Dw`, `REDUCTION_FACTOR = 0.90` (single scalar — no row-count column in this table row).
- `eta(alpha_0) = 1 − sin(alpha_0)/3` *(staticmethod)* — now actually wired into the capacity call, not just recorded.
- `assemble_geometry(catalog, Dw, Dpw, Z, E, alpha_0_deg, nu=0.3)` — note: **no `i`**; multiple rows go through the multi-row family.
- `per_element_dynamic_capacity(bearing, Ca=None)` — dispatches on `bearing.alpha_0`: exactly 90° → `thrust_90deg()`, otherwise → `thrust_nonzero_alpha()`.
- `dynamic_capacity_from_fields(Z, Dw, alpha_0, ri, re, gamma, reduction_factor=None)` *(classmethod)* — Ca from loose fields, so the multi-row family can call it per row without building a `Bearing` per row.
- `dynamic_capacity(bearing)` *(classmethod)* — thin wrapper pulling those fields off an assembled bearing.

**`subtypes/thrust_multirow.py` — `MultiRowThrustBallFamily`** *(new)*

One catalog part built from `i ≥ 2` rows of balls (ISO 1281-1:2021 Sec 6.4). `Bearing.__init__` takes a single `BearingCatalog`, so the per-row data lives in `rows`.
- `CAPABILITIES = {"multirow_capacity"} | SingleRowThrustBallFamily.CAPABILITIES` — it is still a point-contact bearing, so it advertises both; that is exactly what lets `dispatch.py` pick a single-row contact solver for the rows and a multi-row orchestrator for the bearing.
- `REQUIRED_FOR = {"multirow_capacity": {"rows", "i"}}`.
- `assemble_geometry(catalog, row_specs)` — `row_specs` is a list of kwargs dicts, each forwarded as-is to `SingleRowThrustBallFamily.assemble_geometry(catalog, **spec)`. Returns `bearing_type`, `rows`, `i` only — deliberately **not** `ri`/`re`/`Dw`/... at top level. Raises `ValueError` for `len(row_specs) < 2`.
- `dynamic_capacity(bearing)` — per-row Ca via `SingleRowThrustBallFamily.dynamic_capacity_from_fields()` (each row dispatched on its own `alpha_0`), combined by `BearingCapacity.dynamic_multirow()`.

### families/roller_bearing/radial — line contact, radial duty

**`functions/contact_stiffness.py`** — `lamina_positions(Lwe, n_s)` (Sec 5.2.2, midpoints strictly inside `(−Lwe/2, Lwe/2)`), `gamma(Dwe, Dpw, alpha_0)` (raises if outside `(0, 1)`), `line_contact_spring_constant(Lwe, n_s)` → `(cL, cs)` eq.(35)/(37).

**`functions/capacity.py`**
- `BearingCapacity.dynamic(Z, Dwe, Lwe, alpha_0, gamma, reduction_factor, i=1)` — Cr [N], ISO 281:2007 Formula (33), `f_c` from Formula (34). *(Now implemented — was `NotImplementedError` in the previous doc pass.)*
- `BearingCapacity.static(...)` — `NotImplementedError`, `f_0` table not provided.
- `RollingElementCapacity.radial(Z, alpha_0, gamma, Cr, lambda_v, i=1)` *(classmethod)* — `(Q_ci, Q_ce)` [N], whole-roller, ISO/TS 16281 Sec 5.3.1.2 eq.(47)-(48). `lambda_v` has no default — always the subtype's own table value.
- `RollingElementCapacity.per_lamina(Q_ci, Q_ce, n_s)` — `(q_ci, q_ce)` [N] per-lamina, Sec 5.3.2 eq.(56)-(57).

**`subtypes/cylindrical_roller.py` — `CylindricalRollerFamily`**

NU/N-type, zero nominal contact angle, no axial capacity. `LAMBDA_V_RADIAL = 0.83` (ISO 281:2007 Table 2, Table No. 7). Rejects `catalog.arrangement == "locating"` at assembly (no flange to react axial load) — only `"floating"`/`"non-locating"` are valid.
- `CAPABILITIES = {"line_contact"}`; `REQUIRED_FOR["line_contact"] = {Dwe, Lwe, Dpw, Z, s, n_s, alpha_0, x_k, phi_j, gamma, cL, cs, P_xk, i}`.
- `assemble_geometry(catalog, Dwe, Lwe, Dpw, Z, s, n_s=30, alpha_0_deg=0.0, i=1)` — `i` is now an input, mirroring `DeepGrooveBallFamily`. `n_s >= 30` enforced (Sec 5.2.2).
- `_reference_roller_profile(x_k, Dwe, Lwe)` — Sec 6.2 eq.(42)-(44); cached onto `bearing.P_xk` at assembly, not recomputed per solve iteration. Lives on the subtype (not `functions/`) because a future tapered/spherical family needs a different profile.
- `dynamic_capacity(bearing, reduction_factor)` — Cr via `BearingCapacity.dynamic()`.
- `per_element_dynamic_capacity(bearing, Cr=None, lambda_v=None)` — whole-roller `(Q_ci, Q_ce)`.
- `per_lamina_dynamic_capacity(bearing, Q_ci, Q_ce)` — `(q_ci, q_ce)`.

### families/roller_bearing/thrust — line contact, thrust duty

**`functions/capacity.py`** — ISO 281:2007 Sec 6.6:
- `dynamic_nonzero_alpha(Z, Dwe, Lwe, alpha_0, gamma, reduction_factor, eta)` — Formula (37), `f_c` from (38). Single row only.
- `dynamic_90deg(Z, Dwe, Lwe, gamma, reduction_factor, eta)` — Formula (41), `f_c` from (42).
- `dynamic_multirow(Z_rows, Lwe_rows, Ca_rows)` — Formula (46): `Ca = (Σ Zⱼ·Lweⱼ) · [Σ (Zⱼ·Lweⱼ/Caⱼ)^(9/2)]^(−2/9)`.
- `static(...)` — `NotImplementedError`.
- `RollingElementCapacity.thrust_nonzero_alpha(Z, alpha_0, gamma, Ca, lambda_v)`. **Flagged, not silently fixed:** this method's `base` term does NOT carry the leading `1.038` coefficient that the radial side's eq.(47)-(48) `base` does — ported exactly as given from the source draft; needs confirming against the actual ISO/TS 16281 clause before trusting it.
- `RollingElementCapacity.thrust_90deg(Z, Ca, lambda_v)` — `Q_ci == Q_ce` (the bracket collapses to a fixed `2^(2/9)` constant, no geometry dependence left).
- `RollingElementCapacity.per_lamina(Q_ci, Q_ce, n_s)` — same formula as the radial side, duplicated for self-containment.

**`subtypes/cylindrical_single.py` — `ThrustCylindricalRollerFamily`**

`LAMBDA_V_THRUST = 0.73` (ISO 281:2007 Table 2, Table No. 10, "Thrust roller bearings" — not split by roller shape). Unlike the previous revision, `alpha_0_deg` is now a **required input** to `assemble_geometry()` rather than fixed at 90°, and `eta_from_alpha_0(alpha_0)` implements the Table No. 10 `eta = 1 − 0.15·sin(alpha)` column.
- `assemble_geometry(catalog, Dwe, Lwe, Dpw, Z, s, alpha_0_deg, n_s=30)`.
- `_reference_roller_profile(...)` — identical to the radial side (same roller, same eq.(42)-(44)); duplicated rather than imported cross-duty.
- `dynamic_capacity(bearing, reduction_factor=None)` — dispatches on `alpha_0`: 90° → `dynamic_90deg()`, otherwise → `dynamic_nonzero_alpha()`.
- `per_element_dynamic_capacity(bearing, Ca=None, lambda_v=None)` *(classmethod)*, `per_lamina_dynamic_capacity(bearing, Q_ci, Q_ce)`.

**`subtypes/cylindrical_multirow.py` — `MultiRowThrustCylindricalRollerFamily`** *(new)*

Roller-side counterpart of `MultiRowThrustBallFamily`, same shape.
- `assemble_geometry(catalog, row_specs)` — each spec forwarded to `ThrustCylindricalRollerFamily.assemble_geometry()`; returns `rows`, `i`, `bearing_type`.
- `dynamic_capacity(bearing, reduction_factor, nu, eta)` — each argument may be a scalar (applied to every row) or a per-row sequence; per-row Ca via `BearingCapacity.dynamic_90deg()`, combined by `dynamic_multirow()` (Formula 46).

**`subtypes/needle_single.py` — `ThrustNeedleRollerFamily`**

`ALPHA_0_DEG = 90.0` fixed (flat race, pure axial — flagged assumption, not confirmed against a source for a variable-angle case). Same profile and per-lamina helpers as the cylindrical thrust family; kept as a separate class since needle-specific geometry constraints may diverge later.
- `assemble_geometry(catalog, Dwe, Lwe, Dpw, Z, s, n_s=30)`.
- `dynamic_capacity(bearing, reduction_factor, nu, eta)`, `per_element_dynamic_capacity(...)` *(classmethod, always `thrust_90deg()`)*, `per_lamina_dynamic_capacity(...)`.

**Solver coverage today.** `ISO16281BallSolver` serves any family declaring `point_contact` (DGBB, angular contact, and each row of a multi-row thrust ball bearing); `ISO16281RollerSolver` serves `line_contact` (`CylindricalRollerFamily`). Multi-row bearings are routed to the `MULTIROW_SOLVER` registered on the resolved row solver. The self-aligning ball and the thrust roller families have core-level capacity support but no internal load-distribution solver wired yet; `TAPERED_ROLLER` / `SPHERICAL_ROLLER` need an extra coordinate transform neither solver implements.

---

## core/machine_elements — Gears

Now under `gears/parallel_axis/`, split three ways: `gear_properties/` (single-gear geometry), `gear_meshing/` (pair physics), `planetary_gear/` (epicyclic train). The `parallel_axis/__init__.py` roll-up exposes all five public classes.

### gear_properties

**`spur_helical_gear.py` — `SpurHelicalGear`**

Geometry model for an external spur (β=0) or helical (β>0) gear, per ISO 53:2013 + ISO 21771:2007. Constructor: `mn`, `z`, `Ca=0.0`, `Cf=0.0`, `x=0.0`, `b=0.0`, `alpha_n_deg=20.0`, `beta_n_deg=0.0`, rack parameters `haP=1.0`, `cP=0.25`, `rfP=0.38`, surface `Ra`/`Rq`/`Rz`, `label`, `material_id`, `position`. Derived at construction: `mt`, `alphat`, `betab`, `rhoF`, `pb`, `pt`, `pbt`, `d`, `r`, `db`.
- `undercutting()` — `z_min = 2·cos(β)/sin²(αt)·(haP − x)`.
- `validate()` / `validate_or_raise()`.

**`internal_gear.py` — `InternalGear`**

Ring (internal) gear following the signed-z convention (`z < 0`), per ISO 21771 + KHK §4.2. Same rack/surface parameter set; derived `mt`, `mx`, `pn`, `pt`, `pz`, `px`, `d`, `r`, `beta_b`, `db`, `rb`, `alphat`.
- `values_at(dy)` — point-specific involute quantities at an arbitrary flank diameter.
- `tip_below_base_circle()`, `involute_interference(z1, x1)`, `trochoid_interference(z1, x1)`, `trimming_interference(z1)`.
- `validate_mesh(z1, x1=0.0)` — interference checks against a specific mating pinion, kept separate from `validate()` because they are inherently pair-dependent.
- `_solve_inv(inv_target, alpha0)` — Newton–Raphson on `inv(α) = tan α − α`.

### gear_meshing

**`spurhelical_meshing.py` — `SpurHelicalGearMeshing`** *(was `spur_helical_gear_meshing.py`)*

External spur/helical pair. Computes working centre distance `al` and transverse working pressure angle `alphatw`, contact ratios (εα, εβ, εγ), interference checks, and mesh forces.
- Constructor: `gear1`, `gear2`, `label=""`, `al=None`, `equalise_gs=False`, `addendum_reduction=False`, `driver="gear1"`. With `al=None` the centre distance is derived from the declared profile-shift sum (`_set_al_from_x_sum`, `brentq` on the involute equation); with `al` given it is back-calculated (`_set_alphatw_from_al`).
- `forces(T1, phi_deg=0.0, rotation_dir=1)` → dict with Ft, Fr, Fa, per-side load angles, output torque and rotation sense.
- `gear_geometry(addendum_reduction=True)` — builds independent working copies `gear1_w` / `gear2_w` and the contact ratios.
- `correction_for_gs_equilibrium()` — Henriot profile-shift balancing (equalises max specific sliding).
- `al`, `u`, `a`, `x_sum_from_al`, `x_sum_declared`; `validate()` / `validate_or_raise()` / `summary()`.

**`internal_meshing.py` — `InternalGearMeshing`** *(was `internal_gear_meshing.py`)*

External + internal (ring) pair, using the difference form (z2−z1, x2−x1) throughout: `a = (d2 − d1)/2`, `x_diff_from_al`, `x_diff_declared`, `u = z2/z1`.
- `forces(T_in, phi_deg, rotation_dir_in)` — same interface as the external pair; internal meshing does **not** reverse rotation sense.
- `gear_geometry(addendum_reduction=False)`, `correction_for_gs_equilibrium()`, `validate()` (delegates to `InternalGear.validate_mesh`), `summary()`.

### planetary_gear

**`planetary_gear_meshing.py`** — moved here from `mechanical_system/`, since a planetary train is a *meshing* problem, not a shaft-system assembly.

**`PlanetaryGearTrainMeshing`** *(renamed from `PlanetaryGearMeshing`)* — single-stage epicyclic train (sun + k planets + ring + carrier) by composition of one external pair (`meshing_12 = SpurHelicalGearMeshing(sun, planet)`) and one internal pair (`meshing_23 = InternalGearMeshing(planet, ring)`). Owns train-specific logic only.
- Constructor: `sun_gear`, `planet_gear`, `ring_gear`, `k=3`, `label`, `phi0_deg=0.0`, per-mesh `al_* / equalise_gs_* / addendum_reduction_*`, `load_sharing_factor=1.0`. Derives `al_12`, `al_23`, `r_carrier = al_12`, `i_0 = z_ring/z_sun`.
- `planet_phi_deg(j)`, `planet_position(j)` — angular and (y, z) placement of planet j.
- `willis_residual(omega_1, omega_3, omega_H)` — LHS of eq.(7.2), zero for any admissible motion.
- `solve_kinematics(prescribed)` → `PlanetaryKinematics`; handles F=1 and F=2 (two prescribed velocities).
- `speed_ratio(A, B, C)` / `speed_ratio_equation(A, B, C)` — closed-form `i_AB(C)` plus the book equation number backing it (traceability); `ratio(...)` is a backwards-compatible alias.
- `mode_of_work(i)` — `"reducer"` / `"multiplier"`. `is_pseudo_pgt(C)` — `True` when the carrier is fixed (the train degenerates to a fixed-axis train).
- `t` *(property)* — torque ratio (6.7), `t = T3/T1 = −i_0`.
- `torques(T_in, input_member)` → `PlanetaryTorques`.
- `coaxiality_error()`, `assembly_condition()`, `neighbouring_condition(Safety_Factor=1.1)`, `validate()` / `validate_or_raise()` / `summary()`.

**`PlanetaryKinematics`** — frozen result of the Willis solve: `omega_1/2/3/H`, the carrier-relative `omega_*_rel`, `i_0`, DoF `F`, `fixed`, `willis_residual`, `prescribed`, plus readable `omega_sun` / `omega_planet_abs` / `omega_planet_rel` / `omega_ring` / `omega_carrier` properties.

**`PlanetaryTorques`** — ideal external torques `T1`, `T3`, `TH` plus `t`, `Ft`, `F_H2`, `T_mesh_sun`, `T_mesh_planet`, `k`. Valid under three preconditions (stationary load, K_γ = 1, no losses). `satisfies_torque_ordering()` checks eq.(6.6) `|T1| < |T3| < |TH|` as a cheap sanity check — not enforced.

**`PlanetaryMember`** / **`MeshTag`** — enums tagging train members (sun/planet/ring/carrier) and tooth contacts. `MeshTag` is currently unused, kept as a building block.

---

## core/mechanical_system — Systems

Now under `mechanical_system/parallel_axis/spur_helical/`. Gear meshing has left this package — it lives with the gears (see above).

**`shaft_system.py`**

**`GearElement`** — thin wrapper binding a gear geometry object to a kinematic role on a shaft; carries no geometry of its own.
- Constructor: `gear`, `role` (`"driver"`/`"driven"`), `rotation_dir` (±1, set only for the source gear), `label`.
- `position` — delegates to the underlying gear.
- `validate()` / `validate_or_raise()` — delegates to the gear if it can validate itself.

**`ShaftSystem`** — autonomous single-shaft container of bearings, gears and loads. Has no knowledge of other shafts; `shaft_position` (global y,z offset) and `shaft_origin_x` are normally set by the gear system during `resolve()`.
- Constructor: `shaft`, `name="System_1"`, `speed_rpm=0.0`, `design_life_hours`, `shaft_position=(0,0)`, `shaft_origin_x=0.0`, `label`.
- `add_bearing(b)`, `add_gear(g)`, `add_load(ld)` — placement with axial-bounds checking (chainable); `DistributedRadialLoad` is bounds-checked on `[x_lo, x_hi]`.
- `set_gear_loads(loads)` — idempotent replacement of all `gear_mesh`-sourced loads (safe to re-resolve).
- Sorted accessors: `bearings`, `gears`, `loads`, `support_positions`.
- Type-filtered accessors: `radial_loads`, `axial_loads`, `torque_loads`, `external_moments`, `distributed_radial_loads`.
- `gear_extent(ge)`, `bearing_extent(b)` — axial `[lo, hi]` footprint (position is the centre of face/bearing width; `b == 0.0` → point).
- `validate()` / `validate_or_raise()` — geometry, per-element validation, pairwise gear/bearing overlap (`_overlap_errors`) and shoulder-coincidence clearance (`_shoulder_coincidence_errors`).
- `snapshot()` — freezes the current state into a plain dict (objects referenced, containers copied) for per-shaft downstream consumption.

**`gear_system.py`** *(was `SpurHelical_gear_system.py`)*

**`SpurHelicalMeshLink`** — one directed mesh, shaft_a (driver) → shaft_b (driven), carrying a meshing model and the global line-of-centres angle `phi_deg`. Optional `torque_split` selects fan-out mode; `distribute_loads` toggles distributed vs point gear loads; `meshing_load_factor` (default 1.0) is the hook the ISO 6336 `K_A`/`K_v` data layer multiplies into `Ft`.
- `validate()` — self-mesh, torque_split range, meshing-interface checks.

**`SpurHelicalGearSystem`** — single-source DAG of shafts connected by mesh links. Enforces exactly one source, no convergent merges, acyclicity, fan-out `torque_split` consistency, and axial alignment of every meshed gear pair.
- `resolve(P, rpm, rotation_dir_source, source_position=(0.0, 0.0))` — propagates power from the source through the DAG in topological order, computing torque, rotation sense and shaft placement, then injecting the resulting mesh loads onto every shaft. Torque propagates in N·m.
- Two link modes: `_resolve_link_mode_B` (sequential/reuse — the driver delivers the full available torque) and `_resolve_link_mode_A` (simultaneous fan-out — this mesh takes `torque_split` of the total).
- `_forces_to_loads(...)` → `_radial_as_point` / `_radial_as_distributed` (a `DistributedRadialLoad` over `[pos − b/2, pos + b/2]` when `distribute_loads=True`).
- `validate()` / `validate_or_raise()` — topology plus per-shaft validation. Internal helpers: `_incoming_count`, `_sources`, `_topological_order` (Kahn), `_driver_groups` (by `id(gear_a)` — object identity, which is what makes fan-out mode A work), `_axial_alignment_errors`.
- `summary()`.

---

## core/mechanical_system — Schematic

**`parallel_axis/schematic.py`** — line schematic of a real `ShaftSystem` / `SpurHelicalGearSystem`. Owns nothing about the assembly: it only reads what is already there (sections, `shaft_position`, `shaft_origin_x`, bearings, gears, links) and draws lines. Both axes plot real mm with `ax.set_aspect("equal")`, so proportions are directly comparable. Visual/topological check only — not a solver, and it draws no loads.
- `draw_gear_system(ax, gear_system, axis="y")` — every shaft as a 1D line plus dotted mesh connectors.
- `draw_shaft_system(ax, shaft_system, detail=False, axis="y", local=False)`.
- `draw_shaft_detail(ax, shaft_system, axis="y", local=False)` — 2D detail of one shaft: stepped profile, shoulder transitions, bearings/gears flush against the local shaft edge.
- `gear_anchor(shaft_system, gear, axis="y")`, `recommended_figsize(target, axis="y", height=4.0)`.

> Pick whichever of `axis="y"` / `axis="z"` actually separates your shafts — a link with `phi_deg=270` offsets `shaft_position` along z, and with `axis="y"` every shaft lands on y=0 and overlaps.

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

`RadialLoad`/`ExternalMoment` expose `Fy`/`Fz` (or `My`/`Mz`) and `component(plane)`. `DistributedRadialLoad` provides `resultant_component`, `centroid`, `component_intensity`, `bending_moment_contribution` (closed-form for uniform loads, quadrature otherwise), and `as_point_load` for the constant-θ case.

---

## core — Materials

**`materials.py`**

| Class | Purpose |
|-------|---------|
| `Material` | Shaft/structural material (Shigley-based): `Sut`, `Sy`, `E`, `density`, `poisson_ratio`, `Se_base`. |
| `GearMaterial` | Gear material (ISO 6336-5): `E`, `ν`, `ρ`, `cp`, `k_thermal`, `sigma_Hlim`, `sigma_Flim`, `material_class`. |

`Material` exposes `endurance_limit` (Shigley §6-2: 0.5·Sut, capped at 700 MPa) and `shear_yield_strength` (0.577·Sy). `GearMaterial` exposes `equivalent_modulus(other)` for the Hertzian reduced modulus of a pair.

Embedded shaft library: `S355`, `CrMo42` (42CrMo4), `AISI_1045`, `AISI_4340`. Embedded gear library: `GEAR_STEEL`, `GEAR_ADI`, `GEAR_POM`, `GEAR_PA66`.

Lookups: `get_material(id)`, `get_gear_material(id)`, `available_materials()`, `available_gear_materials()`.

---

## Adjacent: axisforge/database

Not part of `core/`, but every consumer of it is. `database/` holds **tabular standard data only** — no solving, no geometry classes.

**`shaft/keyway/Parallel/parallel_keyway.py`** — `lookup_parallel_keyway(shaft_diameter)` → DIN 6885 row (`d_min < d ≤ d_max`), with tolerance bands pre-resolved into `_min`/`_max` columns so the caller can pick the critical value. Consumed by `Keyway.from_standard()`.

**`shaft/keyway/Woodruff_key/iso3912.py`** — `lookup_woodruff_keyway(shaft_diameter, series=1)` → ISO 3912 row; two series with different diameter bands, same geometry columns.

**`gears/SpurHelicalGears/LoadCapacity_data/`** — ISO 6336-1 dynamic-factor data layer, consumed by the (still empty) gear load-capacity solver:
- `Kv_methodB.py` — `DynamicFactor(link, T1, rpm_driver, moment_inertia_1, moment_inertia_2, stiffness, KA, f_pb_eff, f_fa_eff, rotation_dir=1)`. Computes the Cv1–Cv7 velocity coefficients and Cay (Table 8), single tooth stiffness `c'` (eq. 82/92 with `q'`, `C_R`, `C_B`), the resonance ratio `N`, and dispatches `Kv` across the subcritical / main-resonance / intermediate / supercritical regimes. `applicability()` reports Method B's validity conditions.
- `Kv_methodC.py` — `DynamicFactorC(link, T1, rpm_driver, KA, gear_tolerance_class, rotation_dir=1)`. Table 11 `K1`/`K2`, speed term `(v·z1/100)·√(u²/(1+u²))`, `K3` per formulae (37)/(38).
- `csv_reader.py` — `lookup_KA(driving, driven)`, application factor from ISO 6336-1 Table 4.