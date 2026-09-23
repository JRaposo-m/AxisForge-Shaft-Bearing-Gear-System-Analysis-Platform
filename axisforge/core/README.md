# axisforge/core — Module Reference

Domain model: shaft geometry, bearings and their families, gears, gear and planetary meshing, the multi-shaft system graph, applied loads and materials. These objects carry geometry and data. The solvers that consume them are in [`solvers/README.md`](../solvers/README.md).

← back to [project root](../../README.md)

---

## Table of Contents

- [Scope](#scope)
- [Status](#status)
- [Position in the architecture](#position-in-the-architecture)
- [Import surface](#import-surface)
- [Conventions](#conventions)
- [Shaft](#shaft)
- [Bearings](#bearings)
- [Gears](#gears)
- [Mechanical system](#mechanical-system)
- [Schematic](#schematic)
- [Loads](#loads)
- [Materials](#materials)
- [Database](#database)
- [Design contracts](#design-contracts)
- [Extending this package](#extending-this-package)
- [Standards referenced](#standards-referenced)

---

## Scope

| In scope | Out of scope |
|---|---|
| Geometry, catalogue data, and everything derivable from them alone | Anything requiring a load case or a solve |
| A standard's *capacity* formulae, owned by the element they describe | A standard's *analysis* procedure |
| Validation at construction | Validation of a result |
| Topology of a multi-shaft system and the power flow through it | The FEM that solves any shaft in it |

A domain object is a complete, self-consistent description of a physical thing. It knows its own geometry, it can check itself, and it owns the standard formulae that depend only on that geometry. It never learns about load cases, X/Y factors, or life.

---

## Status

| Group | Module | Contents | Status |
|---|---|---|---|
| `machine_elements/shaft/` | `shaft.py` | `Shaft`, `ShaftSection`, `Shoulder`, `Keyway`, `KeywayType` | Implemented |
| `machine_elements/bearings/` | `base.py` (`BearingType`, `BearingCatalog`, `BearingFamily`), `bearing.py` (`Bearing`) | Orchestration primitives | Implemented — **restructured**: `catalog.py`/`bearing_types.py` no longer exist as separate files, folded into `base.py` |
| `machine_elements/bearings/families/` | `family.py` (all concrete families + `@register_family` registry), `capacity.py`, `iso16281_contact.py` | Eight concrete families | Implemented; static capacity `Ca0` outstanding for all — see [Families](#families) for what changed |
| `machine_elements/gears/parallel_axis/gear_properties/` | `spur_helical_gear.py`, `internal_gear.py` | Single-gear geometry | Implemented |
| `machine_elements/gears/parallel_axis/gear_meshing/` | `spurhelical_meshing.py`, `internal_meshing.py` | Pair meshing | Implemented |
| `machine_elements/gears/parallel_axis/planetary_gear/` | `planetary_gear_meshing.py` | Single-stage epicyclic train | Implemented |
| `mechanical_system/parallel_axis/spur_helical/` | `shaft_system.py`, `gear_system.py` | `ShaftSystem`, `GearElement`, `SpurHelicalMeshLink`, `SpurHelicalGearSystem` | Implemented |
| `mechanical_system/parallel_axis/` | `schematic.py` | matplotlib line schematic | **Removed** — the module no longer exists on disk (only a stale `.pyc` remains) and `parallel_axis/__init__.py` does not import it. See [Schematic](#schematic). |
| — | `loads.py` | Six load types plus `LoadingProfile` | Implemented |
| — | `materials.py` | `Material`, `GearMaterial`, embedded libraries | Implemented |
| `database/` | keyways, `K_A`, `K_v` | Standard tabular data | Implemented; the two `K_v` modules have broken imports — see [Database](#database) |

---

## Position in the architecture

```
   config
      │
      ├──────────────┐
      ▼              ▼
  database  ────▶  core                  this package
                     │
                     ├────────▶ mesh
                     └────────▶ solvers
```

`core/` depends on `config`, `database/`, NumPy and the standard library. It imports no solver, no mesh primitive, no result container and no interface layer. This is what makes every domain object usable standalone — in a script, in a test, or in a report — without dragging in a numerical stack.

> **Open item.** The direction `database → core` shown above is the intended one, and holds for the keyway tables. It does **not** hold for `Kv_methodB` and `Kv_methodC`, which import from `core/`. See [Database](#database).

---

## Import surface

| Import from | Names |
|---|---|
| `axisforge.core.machine_elements.shaft` | `Shaft`, `ShaftSection`, `Shoulder`, `Keyway`, `KeywayType` |
| `axisforge.core.machine_elements.bearings` | `Bearing`, `BearingCatalog`, `BearingFamily`, `BearingType` |
| `axisforge.core.machine_elements.bearings.families` | every concrete `*Family` class |
| `axisforge.core.machine_elements.gears.parallel_axis` | `SpurHelicalGear`, `InternalGear`, `SpurHelicalGearMeshing`, `InternalGearMeshing`, `PlanetaryGearTrainMeshing` |
| `axisforge.core.mechanical_system.parallel_axis.spur_helical` | `ShaftSystem`, `GearElement`, `SpurHelicalMeshLink`, `SpurHelicalGearSystem` |
| `axisforge.core.loads` | the load classes |
| `axisforge.core.materials` | `Material`, `GearMaterial`, the lookups |

`axisforge.core.mechanical_system.parallel_axis.schematic` **no longer exists** — see [Schematic](#schematic).

The concrete bearing families **are** re-exported eagerly, from `axisforge.core.machine_elements.bearings.families` and up through every rollup level to `axisforge` itself — a new family is a new class in `families/family.py` decorated `@register_family`, and its name appears in every `__all__` above it automatically, with nothing written by hand.

---

## Conventions

| Symbol | Meaning |
|---|---|
| `x` | Axial coordinate along a shaft system, increasing left → right |
| `z` | Absolute axial coordinate within a single `Shaft` — `z = 0` is the left face of the first section |
| `XZ` | Horizontal plane — tangential gear force `Wt` |
| `XY` | Vertical plane — radial gear force `Wr`, opposed to gravity |
| `theta` | Angular position, measured `+Y → +Z`, right-hand about `+X` |
| `M` (bending moment, `ExternalMoment`) | Right-hand-rule moment component about the plane's normal axis — `+Z` for `XY`-plane bending (`My` — see naming note below), `+Y` for `XZ`-plane bending (`Mz`) — the same convention as torque about `+X`. Positive `M` and positive nodal rotation `theta` (the FEM rotational DOF, defined on the element side in [`solvers/README.md`](../solvers/README.md#conventions)) point the same physical way — they are work-conjugate. |
| `T` (torque, `TorqueLoad`) | Right-hand rule about `+X`; positive = CCW viewed from `+x` |

Units, as stored: length in **mm**, force in **N**, moment in **N·mm**, torque in **N·m**, stress in **MPa**, angles in **degrees** at the constructor interface and **radians** internally, speed in **rpm** at the interface and **rad/s** internally.

Shaft geometry uses the (r, θ, z) convention. θ appears only for localised features such as keyways, since the shaft body itself is a solid of revolution.

A radial load contributes to **both** planes through its `theta`. Consumers must not pre-filter by plane.

> **Naming note.** `ExternalMoment.My`/`.Mz` are named after the axis they act *about* (right-hand rule), not the plane they bend. `My` (moment about `+Y`) is the component that drives `XY`-plane bending; `Mz` (about `+Z`) drives `XZ`-plane bending — the same `component(plane)` pairing `RadialLoad` already uses for `Fy`/`Fz`. This sign convention was fixed against a reference Abaqus case (single cantilever, one end moment) rather than derived in the abstract, precisely because getting `My`/`Mz` versus the element's rotational DOF backwards produces a solve that runs cleanly and returns a plausible, sign-flipped answer — see [`solvers/README.md`](../solvers/README.md#conventions) for where `theta` is defined on the element side and must be checked against this.

---

## Shaft

`machine_elements/shaft/shaft.py`.

| Class | Purpose |
|---|---|
| `KeywayType` | Enum of keyway standards (parallel, Woodruff). |
| `Keyway` | Localised stress raiser at an absolute `(z_position, theta)`. |
| `Shoulder` | Fillet transition between two adjacent sections. |
| `ShaftSection` | One uniform cylindrical segment with its cross-section properties. |
| `Shaft` | Ordered sequence of sections forming a complete shaft. |

### `Keyway`

Carries `label`, `z_position`, `width`, `depth`, `length`, `theta`, `keyway_type`, the tolerance bands `depth_min`/`depth_max`/`L_min`/`L_max`, and pre-resolved fatigue factors `Kf_bending` / `Kf_torsion`.

| Member | Purpose |
|---|---|
| `from_standard` | Builds a keyway from the catalogue tables in `database/shaft/keyway/`, given the shaft diameter. |
| `validate` | Geometry errors as a list of strings. |

### `Shoulder`

Carries `fillet_radius`, `diameter_large`, `diameter_small`.

| Member | Purpose |
|---|---|
| `r_over_d` | r/d ratio — primary Peterson interpolation variable. |
| `D_over_d` | D/d ratio — secondary interpolation variable. |
| `validate` | Checks the fillet is positive and does not exceed the step height. |

### `ShaftSection`

Carries `length`, `diameter`, `inner_diameter`, `material_id`, `surface_finish_ra`, a list of `keyways`, and a `label`.

| Member | Purpose |
|---|---|
| `is_hollow`, `radius` | Basic geometry queries. |
| `area` | Cross-sectional area A. |
| `second_moment_of_area` | I = π/64·(d⁴ − dᵢ⁴). |
| `polar_moment` | J = π/32·(d⁴ − dᵢ⁴) = 2I. |
| `section_modulus` | W = I/(d/2), for σ_b = M/W. |
| `polar_section_modulus` | Wt = J/(d/2), for τ = T/Wt. |
| `validate` | Section errors, delegating to keyways. |

### `Shaft`

`z = 0` is the left face of the first section; an interior boundary belongs to the section on its right.

| Member | Purpose |
|---|---|
| `add_section` | Append a section to the right end. |
| `total_length`, `n_sections` | Aggregate properties. |
| `axial_start`, `axial_end` | Absolute face positions of a section by index. |
| `section_at` | The section containing a given z, and its index. |
| `diameter_at`, `I_at`, `J_at`, `W_at`, `Wt_at` | Section properties at any axial position. |
| `transitions` | Steps between sections, each optionally carrying a `Shoulder`. |
| `shoulders` | All steps as `(z_position, Shoulder)` pairs. |
| `validate`, `validate_or_raise` | Full consistency check, including adjacent shoulder–diameter matching. |

Shoulders are stored on the **shaft's transitions**, not on individual sections. A fillet is a property of the step between two sections, and duplicating it on both would allow the two copies to disagree.

---

## Bearings

A bearing is **catalogue data plus family-derived geometry**, assembled once and immutable from then on. There is no subclass hierarchy and no factory dispatch: a "subtype" is a `BearingFamily` instance handed directly to `Bearing.assemble()`.

```
bearings/
├── base.py            BearingType, BearingCatalog, BearingFamily (ABC) — the three small,
│                      mutually-coupled primitives, kept together on purpose (see the
│                      module's own docstring): BearingFamily.assemble_geometry() reads
│                      BearingCatalog; BearingFamily.BEARING_TYPE is a BearingType.
├── bearing.py         Bearing — the orchestrator
└── families/
    ├── family.py           Every concrete BearingFamily, each decorated @register_family
    ├── capacity.py         Shared capacity mathematics (point and line contact, radial and thrust)
    ├── iso16281_contact.py Shared contact-stiffness mathematics (point and line contact)
    └── tests/
```

**Superseded structure.** An earlier pass of this package split families by contact type and duty into their own subtree (`ball_bearing/{radial,thrust}/{functions,subtypes}/`, `roller_bearing/{radial,thrust}/{functions,subtypes}/`), each subtype in its own file. That split is gone: every concrete family is now a class in the single `families/family.py`, self-registering via a `@register_family` decorator into a module-level `_FAMILY_REGISTRY: dict[str, type[BearingFamily]]`. `families/__init__.py` does `globals().update(_FAMILY_REGISTRY)` and builds `__all__` from `sorted(_FAMILY_REGISTRY)` — the export list is generated from the registry, not written by hand, so a new family can never be added to `family.py` and forgotten in the public surface. `capacity.py` and `iso16281_contact.py` hold the shared math for both contact types and both duties in one place each, rather than one copy per `ball_bearing/`/`roller_bearing/` subtree.

### Orchestration primitives

**`Bearing`** — the single source of truth every solver reads. It never learns about load cases, X/Y factors or life; those belong to whichever solver computes them.

| Member | Purpose |
|---|---|
| `Bearing.assemble` | The only way to build a bearing: validates the catalogue, checks the requested analyses against the family's capabilities, runs `family.assemble_geometry()`, mirrors every returned attribute onto the instance, checks completeness, then seals it. |
| `family` | The `BearingFamily` instance this bearing was assembled with — how callers reach family-specific methods. |
| `is_enabled` | Whether a given analysis was requested at assembly. |
| `has_internal_geometry` | Whether assembly has completed. |
| `is_locating` | Whether the mounting arrangement is locating. |
| `validate` | Defence-in-depth re-check of the catalogue conditions and required geometry. |
| `summary` | Human-readable dump of every assembled attribute. |

Catalogue attributes mirrored onto the instance: `d`, `D`, `b`, `C`, `C0`, `designation`, `label`, `position`, `arrangement`, and the mean diameter `dm`. **Writing to an assembled bearing raises `AttributeError`.**

**`BearingCatalog`** — frozen dataclass of generic, family-agnostic catalogue data: `d`, `D`, `b`, `C`, `C0`, `designation`, `label`, `position`, `arrangement` (`"locating"` / `"floating"` / `"non-locating"`).

| Member | Purpose |
|---|---|
| `validate` | Catalogue-level checks only — geometry guards belong to the family. |
| `validate_or_raise` | Called automatically inside `Bearing.assemble()`. |

**`BearingFamily`** — the contract a family implements to be pluggable.

| Member | Purpose |
|---|---|
| `CAPABILITIES` | Analysis names this family supports, e.g. `point_contact`, `line_contact`, `multirow_capacity`. |
| `REQUIRED_FOR` | Per analysis, the computed-geometry attributes that must exist after assembly. |
| `BEARING_TYPE`, `DUTY` | Labels mirrored onto the bearing. |
| `name` | Short identifier, e.g. `"deep_groove_ball"`. |
| `assemble_geometry` | Pure function from raw geometry inputs to the flat attribute dict the bearing mirrors. |
| `dynamic_capacity`, `per_element_dynamic_capacity` | Now declared as abstract on the contract itself (see the open item just below), not just a convention every family happens to follow. |

`CAPABILITIES` and `REQUIRED_FOR` are load-bearing beyond assembly: the ISO/TS 16281 dispatcher resolves which solver a bearing gets from exactly these declarations, and the fixture capability selector reads them to build its analysis menu. **Adding a family therefore requires no edit anywhere else.**

> **Open item — two different classes are both called `BearingFamily`.** `base.py` defines a `BearingFamily(ABC)` with `__init_subclass__` enforcement (checks that `REQUIRED_FOR`'s keys exactly match `CAPABILITIES`, and that `DUTY` agrees with `BEARING_TYPE.duty`) and re-exports it as the public `axisforge.core.machine_elements.bearings.BearingFamily`. But `families/family.py` does **not** import that class — it defines its own separate, module-local `class BearingFamily(ABC)` (same name, same four class-level attributes, plus `dynamic_capacity`/`per_element_dynamic_capacity` as extra abstract methods, but **no** `__init_subclass__` check), and every concrete family (`DeepGrooveBallFamily`, `AngularContactFamily`, …) subclasses *that* one. The practical effect: the consistency checks documented above as enforced "at class-definition time" never actually run for any real family, because none of them inherit from the class that has the check; and `isinstance(DeepGrooveBallFamily(), BearingFamily)` is `False` when `BearingFamily` is the one imported from the package's public surface (`base.py`'s), since they are two unrelated classes that happen to share a name. Flagged here rather than silently fixed — this needs a decision (probably: `family.py` should import and subclass `base.py`'s `BearingFamily` and drop its own copy) before more families are added on top of it.

**`BearingType`** — enum label only: `DEEP_GROOVE_BALL`, `ANGULAR_CONTACT`, `SELF_ALIGNING_BALL`, `THRUST_BALL`, `CYLINDRICAL_ROLLER`, `TAPERED_ROLLER`, `SPHERICAL_ROLLER`, `THRUST_CYLINDRICAL_ROLLER`, `THRUST_NEEDLE_ROLLER`. Nothing dispatches on it; it exists so reports can group bearings without importing every family class. Extend it for a genuinely new family — a new life exponent or ISO 281 `p` — not for every subtype variant.

### Assembling a bearing

```python
from axisforge.core.machine_elements.bearings import Bearing, BearingCatalog
from axisforge.core.machine_elements.bearings.families import DeepGrooveBallFamily

bearing = Bearing.assemble(
    family=DeepGrooveBallFamily(),
    catalog=BearingCatalog(d=20, D=47, b=14, C=12_700, C0=6_550,
                           designation="6204", position=20.0, label="brg1a"),
    geometry=dict(Dw=7.94, Dpw=33.5, Z=8, E=206_000, nu=0.3, s=0.010),
    analyses={"point_contact": True},
)
```

`assemble()` raises `ValueError` on bad catalogue data, `NotImplementedError` for an analysis the family does not support, and `RuntimeError` if a required geometry field is missing.

### Families

Eight concrete families today, all in `families/family.py`, each self-registering via `@register_family`:

| Family | Duty / contact | Specified by | Capacity notes |
|---|---|---|---|
| `DeepGrooveBallFamily` | radial, point | diametral clearance `s` | `ri = re = 0.52·Dw`; reduction factor 0.95 (1 row) / 0.90 (2 rows) |
| `AngularContactFamily` | radial, point | nominal contact angle in (0°, 45°] | same raceway ratios as DGBB; reduction factor 0.95 for both row counts |
| `SelfAligningBallFamily` | radial, point | nominal contact angle in (0°, 45°] | `ri = 0.53·Dw`; `re` derived from γ, not a fixed ratio of `Dw`; reduction factor 1.0 for both row counts, overall `Cr` **is** now computed (the previous "overall Cr not available" note no longer applies). The family's own docstring flags that its contact stiffness model may not be properly derived for this bearing type yet — treat capacity numbers from this family as provisional until that is resolved. |
| `ThrustBallSingleRowFamily` (renamed from `SingleRowThrustBallFamily`) | thrust, point | nominal contact angle in (45°, 90°] | `ri = re = 0.535·Dw`; `lam` (single-row) = 0.90; η = 1 − sin α₀/3 |
| `ThrustBallMultiRowFamily` (renamed from `MultiRowThrustBallFamily`) | thrust, point | a list of per-row specifications | per-row Ca combined by the multi-row formula |
| `CylindricalRollerFamily` | radial, line | clearance `s` and lamina count `n_s` | λ_v = 0.83; rejects a locating arrangement — an NU/N-type bearing has no flange to react axial load |
| `ThrustCylindricalRollerFamily` | thrust, line | nominal contact angle | λ_v = 0.73; η = 1 − 0.5·sin α — **corrected from a previous pass of this document**, which stated η = 1 − 0.15·sin α; the source's coefficient is 0.5, confirm against the standard if that number was carried from somewhere specific |
| `RollerThrustMultiRowFamily` (renamed from `MultiRowThrustCylindricalRollerFamily`) | thrust, line | a list of per-row specifications | per-row Ca combined by the multi-row formula; the combination itself is flagged in the source as not yet derived/confirmed (exponents 9/2 and 2/9 carried over by analogy with the ball case) |

> **Open item — `ThrustNeedleRollerFamily` no longer exists.** The previous pass of this document listed a ninth family, flat-race thrust needle roller with contact angle fixed at 90° and λ_v = 0.73. It is not present in the current `family.py` — dropped during the restructuring, or not yet ported back. If it is still needed, it has to be re-added rather than assumed to still be there under a different name.

Every family exposes the same capacity entry points, called through the bearing:

| Method | Returns |
|---|---|
| `per_element_dynamic_capacity` | `(Q_ci, Q_ce)` — per rolling element, from a known Cr or Ca. Thrust families dispatch internally on whether α₀ is 90°. |
| `per_lamina_dynamic_capacity` | `(q_ci, q_ce)` — per lamina, roller families only. |
| `dynamic_capacity` | The overall bearing Cr or Ca from geometry alone. |

**Radial versus thrust multi-row.** These are different things and are handled differently. A *radial* family's `i` (two rows of a deep-groove ball or cylindrical roller bearing) is a plain capacity-rating multiplier on one raceway per ISO 281:2007 Table 1 — solved as a single ring displacement, no rows list, no load split. A *thrust* multi-row family (`ThrustBallMultiRowFamily`, `RollerThrustMultiRowFamily`) assembles `rows` (a list of per-row attribute dicts, built by calling the matching single-row family's `assemble_geometry()` once per row — composition, not a re-implementation) and combines them via each capacity calculator's own `combine_multirow()`. Callers reach a row through `bearing.rows[j]`. A separate `MultiRowCylindricalRollerFamily` was tried for the radial case and retired; do not reintroduce it.

### Shared mathematics

Flattened since the previous pass of this document: shared math for **both** contact types and **both** duties now lives in two files, not one pair per contact-type subtree.

| Module | Contents |
|---|---|
| `families/iso16281_contact.py` | Contact angle and clearance resolution, `PointContactStiffness` and its `SelfAligningPointContactStiffness` variant (γ, raceway contact radius, curvature sums/differences, Hertzian spring constant `c_p`), and `LineContactStiffness` (lamina midpoint positions, γ, the line-contact spring constants `cL`/`cs`). |
| `families/capacity.py` | `PointContactCapacityRadial`, `PointContactCapacityThrust_90deg`/`_Non_90deg`, `LineContactCapacityRadial`, `LineContactCapacityThrust_90deg`/`_Non_90deg`, and the shared `ThrustCapacityCalculator` base with its `combine_multirow()`. |

The reference roller profile (`_reference_roller_profile`, ISO/TS 16281 Sec 6.2 eq. 42–44) is still duplicated between `CylindricalRollerFamily` and `ThrustCylindricalRollerFamily` rather than centralised in `capacity.py`/`iso16281_contact.py` — the two copies are identical today, which is itself a signal that a future tapered/spherical profile should either share this one or make the duplication deliberate.

**Open items.** Static capacity (`Ca0`) is not implemented for any family — the source tables are not in place. `ThrustNeedleRollerFamily` no longer exists (see [Families](#families)). `SelfAligningBallFamily`'s own docstring flags that its contact-stiffness treatment may not be properly elaborated for this bearing type. `RollerThrustMultiRowFamily.combine_multirow()` carries exponents taken by analogy with the ball case, not yet independently derived or confirmed. The `BearingFamily` duplicate-class issue is a separate, structural open item — see the note under [Orchestration primitives](#orchestration-primitives).

**Solver coverage.** Point-contact families are served by the ISO/TS 16281 ball solver, `CylindricalRollerFamily` by the roller solver, and multi-row thrust bearings by the multi-row solver in the same `contact_solver.py` module (see [`solvers/README.md`](../solvers/README.md#bearings--isots-16281)). The self-aligning ball family has capacity support but no internal load-distribution solver. Tapered and spherical roller bearings need a coordinate transform neither solver implements.

---

## Gears

`machine_elements/gears/parallel_axis/`, split into single-gear geometry, pair meshing, and epicyclic trains.

### gear_properties

**`SpurHelicalGear`** — external spur (β = 0) or helical (β > 0) gear per ISO 53 and ISO 21771. Carries module, teeth count, profile shift, face width, pressure and helix angles, the basic rack parameters, surface roughness, material and axial position, and derives the transverse module, transverse and base helix angles, pitches, and reference and base diameters.

| Member | Purpose |
|---|---|
| `undercutting` | Whether the gear is undercut at its declared profile shift. |
| `validate`, `validate_or_raise` | Geometry errors. |

**`InternalGear`** — ring gear following the signed-`z` convention, per ISO 21771 and the KHK reference.

| Member | Purpose |
|---|---|
| `values_at` | Point-specific involute quantities at an arbitrary flank diameter. |
| `tip_below_base_circle` | Whether the tip circle falls inside the base circle. |
| `involute_interference` | Involute interference against a mating pinion. |
| `trochoid_interference` | Flank (trochoid) interference against a mating pinion. |
| `trimming_interference` | Trimming interference from too small a tooth-count difference. |
| `validate_mesh` | All three interference checks against one specific pinion. |
| `validate`, `validate_or_raise` | Standalone geometry errors. |

### gear_meshing

**`SpurHelicalGearMeshing`** — external spur or helical pair. Given a working centre distance it back-calculates the transverse working pressure angle; given none, it derives both from the declared profile-shift sum.

| Member | Purpose |
|---|---|
| `gear_geometry` | Builds independent working copies of both gears and the contact ratios εα, εβ, εγ. |
| `forces` | Ft, Fr, Fa, the per-side load angles, output torque and rotation sense, from the driving torque. |
| `correction_for_gs_equilibrium` | Henriot profile-shift balancing — equalises maximum specific sliding at both gears. |
| `al`, `u`, `a` | Working centre distance, ratio, reference centre distance. |
| `validate`, `validate_or_raise`, `summary` | |

**`InternalGearMeshing`** — external plus internal pair, using the difference form (z₂ − z₁, x₂ − x₁) throughout. Same interface as the external pair, with one behavioural difference: internal meshing does not reverse rotation sense.

### planetary_gear

**`PlanetaryGearTrainMeshing`** — a single-stage epicyclic train (sun, k planets, ring, carrier) built by composition: one external pair for sun–planet, one internal pair for planet–ring. It owns only train-specific logic.

| Member | Purpose |
|---|---|
| `planet_phi_deg`, `planet_position` | Angular and (y, z) placement of planet j. |
| `willis_residual` | Left-hand side of the Willis equation — zero for any admissible motion. |
| `solve_kinematics` | Solves for the unknown central velocities; handles one or two prescribed velocities. |
| `speed_ratio` | i_AB(C) — input A, output B, fixed C. |
| `speed_ratio_equation` | The reference equation number backing that ratio. |
| `mode_of_work` | Reducer or multiplier. |
| `is_pseudo_pgt` | True when the carrier is fixed and the train degenerates to a fixed-axis train. |
| `t` | Torque ratio T₃/T₁. |
| `torques` | Ideal external torques on the three coaxial members from one known input torque. |
| `coaxiality_error` | Difference between the two working centre distances. |
| `assembly_condition` | Whether equally spaced planets are assemblable. |
| `neighbouring_condition` | Whether adjacent planets clear each other. |
| `validate`, `validate_or_raise`, `summary` | |

| Companion | Purpose |
|---|---|
| `PlanetaryKinematics` | Frozen result of the kinematic solve: absolute and carrier-relative angular velocities, the basic ratio, the degrees of freedom, the fixed member, and the Willis residual for numerical transparency. |
| `PlanetaryTorques` | Ideal external torques on sun, ring and carrier plus the mesh torques and tangential force. Valid under stationary load, equal load sharing and no losses. `satisfies_torque_ordering` is a cheap sanity check, not an enforced constraint. |
| `PlanetaryMember`, `MeshTag` | Enums tagging train members and tooth contacts. |

---

## Mechanical system

`mechanical_system/parallel_axis/spur_helical/`.

### `GearElement`

Binds a gear geometry object to a kinematic role on a shaft. It duplicates no geometry — `position` delegates to the underlying gear.

| Member | Purpose |
|---|---|
| `position` | Delegated axial position. |
| `validate`, `validate_or_raise` | Role and rotation-direction checks, delegating to the gear if it can validate itself. |

Constructed with the gear, a role (`"driver"` / `"driven"`), an optional rotation direction (set only on the kinematic source gear — the system propagates it everywhere else), and a label.

### `ShaftSystem`

Autonomous single-shaft container of bearings, gears and loads. It has no knowledge of other shafts; its global position and axial origin are normally written by the gear system during `resolve()`.

| Member | Purpose |
|---|---|
| `add_bearing`, `add_gear`, `add_load` | Chainable placement with axial-bounds checking. |
| `set_gear_loads` | Idempotent replacement of every mesh-sourced load, preserving user loads. Safe to call on each re-resolve. |
| `bearings`, `gears`, `loads`, `support_positions` | Sorted accessors. |
| `radial_loads`, `axial_loads`, `torque_loads`, `external_moments`, `distributed_radial_loads` | Type-filtered accessors. |
| `gear_extent`, `bearing_extent` | Axial footprint `[lo, hi]`; position is the centre of the face or bearing width, and a zero width is treated as a point. |
| `snapshot` | Freezes the current state into a plain dict for per-shaft downstream consumption. |
| `validate`, `validate_or_raise` | Geometry, per-element validation, gear/bearing overlap, and shoulder clearance. |
| `summary` | |

`bearing_extent` is what makes seat misalignment meaningful: a bearing of finite width and a knife-edge support give different shaft slopes at the same position, and the FEM results reader reads this extent to compute `psi`.

### `SpurHelicalMeshLink`

One directed mesh, driver shaft to driven shaft, carrying a meshing model and the global line-of-centres angle. Optional `torque_split` selects fan-out mode; `distribute_loads` turns the mesh forces into loads distributed over the face width; `meshing_load_factor` is the multiplier the ISO 6336 application and dynamic factors feed into Ft.

| Member | Purpose |
|---|---|
| `validate` | Self-mesh, torque-split range, and meshing-interface checks. |

### `SpurHelicalGearSystem`

A single-source DAG of shafts connected by mesh links.

| Member | Purpose |
|---|---|
| `resolve` | Propagates power from the unique source through the DAG in topological order, computing torque, rotation sense and shaft placement, then injecting the resulting mesh loads onto every shaft. |
| `validate`, `validate_or_raise` | Topology plus per-shaft validation. |
| `summary` | Source, topological order and link list. |

Enforced topology:

- exactly one source shaft (zero incoming links);
- no shaft with more than one incoming link — no convergent merge;
- no cycles;
- every meshed gear pair axially aligned within tolerance;
- for a fan-out (one driver gear meshing several driven gears simultaneously), one `GearElement` object reused across the links, each carrying a `torque_split`, summing to 1.

Fan-out is grouped by the **identity** of the driver `GearElement`, not by geometric equality, which is what makes the shared-object pattern work. The torque split itself is an input: computing it belongs to a dedicated pre-solver, not to this class. Non-linear topologies are modelled as several independent systems; planetary arrangements use `PlanetaryGearTrainMeshing` instead.

---

## Schematic

> **Removed.** The previous pass of this document described `mechanical_system/parallel_axis/schematic.py` — a matplotlib line schematic (`draw_gear_system`, `draw_shaft_system`, `draw_shaft_detail`, `gear_anchor`, `recommended_figsize`) drawing a real assembly from sections, shaft positions, bearings, gears and links. That module **no longer exists on disk** — only a stale `.pyc` remains under `mechanical_system/parallel_axis/__pycache__/` — and the current `parallel_axis/__init__.py` does not import anything named `schematic`. Whether this was a deliberate removal pending a rewrite, or a casualty of the `mechanical_system` restructuring, has not been confirmed. If a visual/topological check of an assembly is still wanted, it needs to be re-added; nothing in the current tree replaces it.

---

## Loads

`core/loads.py`.

| Class | Purpose |
|---|---|
| `LoadPlane` | Enum of the principal bending planes (XY, XZ), used as a decomposition key. |
| `Load` | Base class carrying position, label and source (`"user"`, `"gear_mesh"`, `"bearing_reaction"`). |
| `RadialLoad` | Transverse point force at an angular position, decomposed into Fy and Fz. |
| `AxialLoad` | Force along the shaft axis, positive in tension. |
| `TorqueLoad` | Torque about the shaft axis. |
| `ExternalMoment` | Applied bending moment at an orientation, decomposed into My (about `+Y`, drives `XY`-plane bending) and Mz (about `+Z`, drives `XZ`-plane bending) — see the sign-convention note under [Conventions](#conventions). |
| `DistributedRadialLoad` | Transverse load over an interval, with uniform or callable intensity and direction. |
| `LoadingProfile` | Fatigue cycle decomposition — stress ratio into mean and amplitude factors. |

`RadialLoad` and `ExternalMoment` expose their plane components and a `component(plane)` accessor. `DistributedRadialLoad` adds the resultant component, the centroid, the component intensity as a function of x, the bending-moment contribution (closed form when uniform, quadrature otherwise), and conversion to an equivalent point load when the direction is constant.

The `source` field is what makes re-resolving safe: `ShaftSystem.set_gear_loads()` replaces every load tagged `"gear_mesh"` and leaves user loads untouched. A design script can therefore add its own loads, re-resolve the power flow, and keep them.

**`ExternalMoment` sign, and why it matters downstream.** `My`/`Mz` are vector components under the right-hand rule about `+Y`/`+Z` — mechanically identical to how `TorqueLoad` is defined about `+X`. This module makes no assumption about beam elements or DOFs; the decomposition is pure coordinate geometry. What it does commit to, for whoever assembles a load vector downstream (`solvers/.../fem_solvers/assembly/`), is that a positive `My`/`Mz` and a positive nodal rotation `theta` must be defined to point the same way — they are work-conjugate, by construction of the FEM stiffness formulation. Getting this backwards between `loads.py` and the element formulation does not raise an error: the solve completes and returns a plausible, sign-flipped result. This was found by comparison against a reference Abaqus case, not by static analysis of the code, and is why the DOF-side convention now lives explicitly in [`solvers/README.md`](../solvers/README.md#conventions) next to the element stiffness definition, rather than only here.

---

## Materials

`core/materials.py`.

| Class | Purpose |
|---|---|
| `Material` | Shaft and structural material: ultimate and yield strength, modulus, density, Poisson ratio, optional specimen endurance limit. |
| `GearMaterial` | Gear material per ISO 6336-5: modulus, Poisson ratio, density, specific heat, thermal conductivity, σ_Hlim, σ_Flim, quality class. |

| Member | Purpose |
|---|---|
| `Material.endurance_limit` | Specimen endurance limit — the stored value, or 0.5·Sut capped at 700 MPa. |
| `Material.shear_yield_strength` | 0.577·Sy, von Mises. |
| `GearMaterial.equivalent_modulus` | Hertzian reduced modulus for a pair. |

| Lookup | Purpose |
|---|---|
| `get_material`, `available_materials` | Embedded shaft library: `S355`, `CrMo42`, `AISI_1045`, `AISI_4340`. |
| `get_gear_material`, `available_gear_materials` | Embedded gear library: `GEAR_STEEL`, `GEAR_ADI`, `GEAR_POM`, `GEAR_PA66`. |

Marin correction from the specimen endurance limit to a component value is applied by the shaft solver utilities, not here. The distinction is the layer rule: a specimen property belongs to the material, a component property depends on the component's surface, size and loading and therefore belongs to the solver.

---

## Database

`axisforge/database/` is not part of `core/`, but its intended consumers are. It holds standard tabular data — no solving, no geometry classes.

| Module | Provides |
|---|---|
| `shaft/keyway/Parallel/parallel_keyway.py` | DIN 6885 parallel keyway dimensions by shaft diameter, with tolerance bands pre-resolved into min/max columns so the caller can pick the critical value. Consumed by `Keyway.from_standard()`. |
| `shaft/keyway/Woodruff_key/iso3912.py` | ISO 3912 Woodruff keyway dimensions by shaft diameter, in two series. |
| `gears/.../LoadCapacity_data/csv_reader.py` | `lookup_KA` — application factor from ISO 6336-1 Table 4. |
| `gears/.../LoadCapacity_data/Kv_methodB.py` | `DynamicFactor` — ISO 6336-1 Method B: velocity coefficients and mesh stiffness factor, single tooth stiffness, the resonance ratio, and K_v across the subcritical, main-resonance, intermediate and supercritical regimes, with an `applicability()` check of the method's validity conditions. |
| `gears/.../LoadCapacity_data/Kv_methodC.py` | `DynamicFactorC` — ISO 6336-1 Method C: tolerance-class coefficients, the speed term, and K_v, with its own `applicability()`. |

These feed the gear load-capacity solver, whose module is reserved and not yet written.

> **Open item — broken imports.** `Kv_methodB` and `Kv_methodC` both import from
> `axisforge.core.mechanical_system.Parallel_Axis_systems.gear_meshing.spur_helical_gear_meshing` and
> `axisforge.core.mechanical_system.Parallel_Axis_systems.systems.spur_helicoidal_system.SpurHelical_gear_system`.
> Neither path exists: the package is `core/mechanical_system/parallel_axis/spur_helical/`. Both modules therefore cannot import today, and must be repaired before the gear load-capacity solver can consume them.
>
> Repairing them also raises a layering question. `Kv_methodB` additionally imports `core.materials`, so `database` depends on `core` while `core.machine_elements.shaft` depends on `database` — a cycle across the two packages. The `K_v` calculation needs geometry and speed, which suggests it belongs in the gear solver with the database supplying only the tables it reads. Worth deciding before the solver is written rather than after.

---

## Design contracts

- **Geometry and data only.** No load case, no solve, no result.
- **Validation at construction.** `validate()` returns a list of error strings; `validate_or_raise()` raises. Every domain object implements the pair. Invalid geometry is never silently accepted.
- **Capacity belongs to the element.** A standard's capacity formula lives with the thing it describes, and every family exposes the same entry points so a solver never carries its own copy.
- **Immutable once assembled.** A `Bearing` refuses every write after `assemble()`. Other domain objects are treated as write-once by convention.
- **Composition over inheritance.** A bearing composes a family; a planetary train composes two pair-meshing objects. There is no subclass hierarchy of bearings or of gears.
- **Declared capability, not hard-coded dispatch.** A family declares `CAPABILITIES` and `REQUIRED_FOR`; the solver dispatcher and the fixture capability selector both read those declarations. Adding a family edits no table anywhere else.
- **Duplicate nothing.** A shoulder lives on a transition, not on both adjacent sections. A `GearElement` delegates its position rather than storing a copy.
- **Flag, do not silently fix.** The thrust-roller coefficient discrepancy is marked in the source and named in this document rather than quietly corrected.
- **Sign conventions are cross-package contracts, not local choices.** `ExternalMoment`'s `My`/`Mz` (here) and the FEM rotational DOF `theta` (in `solvers/`) must agree on which physical sense is positive. Neither module can enforce this on its own — it is checked once, against a reference case, and then documented on both sides rather than re-derived from scratch by whoever writes the element formulation next.

---

## Extending this package

**Adding a bearing family.** One new class in `families/family.py`, decorated `@register_family`. Declare `CAPABILITIES`, `REQUIRED_FOR`, `BEARING_TYPE`, `DUTY` and `name`; implement `assemble_geometry()` as a pure function returning a flat attribute dict, plus the capacity entry points (`dynamic_capacity`, `per_element_dynamic_capacity`, and `per_lamina_dynamic_capacity` for line contact). Put shared mathematics in `iso16281_contact.py`/`capacity.py`, and anything genuinely specific to the family — a profile formula, a raceway ratio — as a class constant or method on the family itself. Extend `BearingType` only if this is a genuinely new family in the ISO 281 sense. The registration decorator means the class appears in the public surface automatically; nothing else needs editing by hand. **Until the `BearingFamily` duplicate-class issue above is resolved, subclass the local `BearingFamily` defined in `family.py` itself, to stay consistent with every existing family** — not `base.py`'s, which no concrete family currently uses.

**Adding a gear type.** Geometry in `gear_properties/`, pair behaviour in `gear_meshing/`. A train that composes pairs goes in its own subpackage, as `planetary_gear/` does, and owns only train-specific logic.

**Adding a load type.** Subclass `Load`, expose plane components through `component(plane)` if it is directional, and tag its `source` so `set_gear_loads()` knows whether to replace it. If the new type is directional and moment-like (not force-like), state explicitly which axis it is a right-hand-rule component about, and cross-reference the DOF it must stay work-conjugate with once a solver consumes it.

**Adding standard tabular data.** It goes in `database/`, as data only, with no import from `core/`.

---

## Standards referenced

| Standard | Applies to |
|---|---|
| ISO 281 | Bearing dynamic load ratings, multi-row reduction factors, life exponent `p` |
| ISO 76 | Bearing static load ratings — not yet implemented for any family |
| ISO/TS 16281 | Per-element and per-lamina capacity entry points |
| ISO 21771 | Cylindrical involute gear geometry and nomenclature |
| ISO 53 | Standard basic rack tooth profile |
| ISO 6336-1 | Application factor `K_A`, dynamic factor `K_v` (Methods B and C) |
| ISO 6336-5 | Gear material strength values |
| DIN 6885 | Parallel keys and keyways |
| ISO 3912 | Woodruff keys and keyways |