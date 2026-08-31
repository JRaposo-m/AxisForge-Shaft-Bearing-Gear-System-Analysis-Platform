# axisforge/core — Module Reference

Domain model: shaft geometry, bearings, gears, gear and planetary meshing, the multi-shaft system graph, applied loads and materials. These objects carry geometry and data. The solvers that consume them are in [`solvers/README.md`](../solvers/README.md).

← back to [project root](../../README.md)

---

## Table of Contents

- [Import surface](#import-surface)
- [Shaft](#shaft)
- [Bearings](#bearings)
- [Gears](#gears)
- [Mechanical system](#mechanical-system)
- [Schematic](#schematic)
- [Loads](#loads)
- [Materials](#materials)
- [Database](#database)

---

## Import surface

| Import from | Names |
|---|---|
| `axisforge.core.machine_elements.shaft` | `Shaft`, `ShaftSection`, `Shoulder`, `Keyway`, `KeywayType` |
| `axisforge.core.machine_elements.bearings` | `Bearing`, `BearingCatalog`, `BearingFamily`, `BearingType` |
| `axisforge.core.machine_elements.bearings.families` | every concrete `*Family` class |
| `axisforge.core.machine_elements.gears.parallel_axis` | `SpurHelicalGear`, `InternalGear`, `SpurHelicalGearMeshing`, `InternalGearMeshing`, `PlanetaryGearTrainMeshing` |
| `axisforge.core.mechanical_system.parallel_axis.spur_helical` | `ShaftSystem`, `GearElement`, `SpurHelicalMeshLink`, `SpurHelicalGearSystem` |
| `axisforge.core.mechanical_system.parallel_axis.schematic` | `draw_gear_system`, `draw_shaft_system`, `draw_shaft_detail`, `gear_anchor`, `recommended_figsize` |
| `axisforge.core.loads` | the load classes |
| `axisforge.core.materials` | `Material`, `GearMaterial`, the lookups |

The concrete bearing families are not re-exported by `bearings` on purpose: a new family is a new file under `families/`, and nothing else in `core/` changes.

---

## Shaft

`machine_elements/shaft/shaft.py` — geometric primitives in the (r, θ, z) convention. `z` is the absolute axial coordinate; θ is used only for localized features, since the shaft body itself is a solid of revolution.

| Class | Purpose |
|---|---|
| `KeywayType` | Enum of keyway standards (parallel, Woodruff). |
| `Keyway` | Localized stress raiser at an absolute `(z_position, theta)`. |
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

Carries `length`, `diameter`, `inner_diameter`, `material_id`, `surface_finish_ra`, optional `shoulder_left` / `shoulder_right`, a list of `keyways`, and a `label`.

| Member | Purpose |
|---|---|
| `is_hollow`, `radius` | Basic geometry queries. |
| `area` | Cross-sectional area A. |
| `second_moment_of_area` | I = π/64·(d⁴ − dᵢ⁴). |
| `polar_moment` | J = π/32·(d⁴ − dᵢ⁴) = 2I. |
| `section_modulus` | W = I/(d/2), for σ_b = M/W. |
| `polar_section_modulus` | Wt = J/(d/2), for τ = T/Wt. |
| `validate` | Section errors, delegating to shoulders and keyways. |

### `Shaft`

`z = 0` is the left face of the first section; an interior boundary belongs to the section on its right.

| Member | Purpose |
|---|---|
| `add_section` | Append a section to the right end. |
| `total_length`, `n_sections` | Aggregate properties. |
| `axial_start`, `axial_end` | Absolute face positions of a section by index. |
| `section_at` | The section containing a given z, and its index. |
| `diameter_at`, `I_at`, `J_at`, `W_at`, `Wt_at` | Section properties at any axial position. |
| `shoulders` | All steps as `(z_position, Shoulder)` pairs. |
| `validate`, `validate_or_raise` | Full consistency check, including adjacent shoulder–diameter matching. |

---

## Bearings

A bearing is **catalogue data plus family-derived geometry**, assembled once and immutable from then on. There is no subclass hierarchy and no factory dispatch: a "subtype" is a `BearingFamily` instance handed directly to `Bearing.assemble()`.

```
bearings/
├── bearing.py         Bearing — the orchestrator
├── bearing_types.py   BearingType
├── catalog.py         BearingCatalog
├── family.py          BearingFamily (ABC)
└── families/
    ├── ball_bearing/{radial,thrust}/{functions,subtypes}/
    └── roller_bearing/{radial,thrust}/{functions,subtypes}/
```

Within a family directory, `functions/` holds the shared contact and capacity mathematics for that contact type and duty; `subtypes/` holds the family classes, each owning its own standard-table constants so one table row can change without affecting another.

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

Catalogue attributes mirrored onto the instance: `d`, `D`, `b`, `C`, `C0`, `designation`, `label`, `position`, `arrangement`, and the mean diameter `dm`. Writing to an assembled bearing raises `AttributeError`.

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

`CAPABILITIES` and `REQUIRED_FOR` are load-bearing beyond assembly: the ISO/TS 16281 dispatcher resolves which solver a bearing gets from exactly these declarations, and the fixture capability selector reads them to build its analysis menu.

**`BearingType`** — enum label only: `DEEP_GROOVE_BALL`, `ANGULAR_CONTACT`, `SELF_ALIGNING_BALL`, `THRUST_BALL`, `CYLINDRICAL_ROLLER`, `TAPERED_ROLLER`, `SPHERICAL_ROLLER`, `THRUST_CYLINDRICAL_ROLLER`, `THRUST_NEEDLE_ROLLER`. Nothing dispatches on it; it exists so reports and the GUI can group bearings without importing every family class. Extend it for a genuinely new family — a new life exponent or ISO 281 `p` — not for every subtype variant.

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

| Family | Duty / contact | Specified by | Capacity notes |
|---|---|---|---|
| `DeepGrooveBallFamily` | radial, point | diametral clearance `s` | `ri = re = 0.52·Dw`; reduction factor 0.95 (1 row) / 0.90 (2 rows) |
| `AngularContactFamily` | radial, point | nominal contact angle in (0°, 45°] | same raceway ratios as DGBB; reduction factor 0.95 for both row counts |
| `SelfAligningBallFamily` | radial, point | nominal contact angle in (0°, 45°] | `ri = 0.53·Dw`; `re` derived from γ, not a fixed ratio of `Dw`; overall Cr not available |
| `SingleRowThrustBallFamily` | thrust, point | nominal contact angle in (45°, 90°] | `ri = re = 0.535·Dw`; reduction factor 0.90; η = 1 − sin α₀/3 |
| `MultiRowThrustBallFamily` | thrust, point | a list of per-row specifications | per-row Ca combined by the multi-row formula |
| `CylindricalRollerFamily` | radial, line | clearance `s` and lamina count `n_s` | λ_v = 0.83; rejects a locating arrangement — an NU/N-type bearing has no flange to react axial load |
| `ThrustCylindricalRollerFamily` | thrust, line | nominal contact angle | λ_v = 0.73; η = 1 − 0.15·sin α |
| `MultiRowThrustCylindricalRollerFamily` | thrust, line | a list of per-row specifications | per-row Ca combined by the multi-row formula |
| `ThrustNeedleRollerFamily` | thrust, line | flat race, contact angle fixed at 90° | λ_v = 0.73 |

Every family exposes the same three capacity entry points, called through the bearing:

| Method | Returns |
|---|---|
| `per_element_dynamic_capacity` | `(Q_ci, Q_ce)` — per rolling element, from a known Cr or Ca. Thrust families dispatch internally on whether α₀ is 90°. |
| `per_lamina_dynamic_capacity` | `(q_ci, q_ce)` — per lamina, roller families only. |
| `dynamic_capacity` | The overall bearing Cr or Ca from geometry alone. |

A multi-row family assembles `rows` (a list of per-row attribute dicts) and `i` instead of flat top-level geometry, because the per-row fields only mean something per row. Callers reach a row through `bearing.rows[j]`.

### Shared mathematics

| Module | Contents |
|---|---|
| `ball_bearing/*/functions/contact_stiffness.py` | Contact angle from clearance, γ, raceway contact radius, curvature sums and differences, Hertzian spring constant `c_p`, plus a self-aligning variant for the degenerate outer-race case. |
| `ball_bearing/*/functions/capacity.py` | Overall Cr / Ca and per-element `(Q_ci, Q_ce)` for point contact, radial and thrust duty, including the multi-row combination. |
| `roller_bearing/*/functions/contact_stiffness.py` | Lamina midpoint positions, γ, and the line-contact spring constants `(cL, cs)`. |
| `roller_bearing/*/functions/capacity.py` | Overall Cr / Ca, whole-roller `(Q_ci, Q_ce)`, per-lamina `(q_ci, q_ce)`, and the multi-row combination. |

The reference roller profile lives on each roller subtype rather than in `functions/`, because a future tapered or spherical family needs a different profile formula. It is computed once at assembly and cached on the bearing, so no solver recomputes it per iteration.

**Open items.** Static capacity (`Ca0`) is not implemented for any family — the source tables are not in place. The thrust-roller per-element formula omits a leading coefficient that its radial counterpart carries; this is ported as received and marked in the source, pending confirmation against the standard.

**Solver coverage.** Point-contact families are served by the ISO/TS 16281 ball solver, `CylindricalRollerFamily` by the roller solver, and multi-row bearings by the multi-row solver registered on their row solver. The self-aligning ball and thrust roller families have capacity support but no internal load-distribution solver. Tapered and spherical roller bearings need a coordinate transform neither solver implements.

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

Fan-out is grouped by the *identity* of the driver `GearElement`, not by geometric equality, which is what makes the shared-object pattern work. The torque split itself is an input: computing it belongs to a dedicated pre-solver, not to this class. Non-linear topologies are modelled as several independent systems; planetary arrangements use `PlanetaryGearTrainMeshing` instead.

---

## Schematic

`mechanical_system/parallel_axis/schematic.py` — a matplotlib line schematic of a real assembly. It owns nothing: it reads sections, shaft positions, bearings, gears and links, and draws. Both axes plot real millimetres with equal aspect, so a bearing block, a shaft diameter and the distance between two shafts are directly comparable. It is a visual and topological check, not a solver, and it draws no loads.

| Function | Purpose |
|---|---|
| `draw_gear_system` | Every shaft as a 1D line, plus dotted connectors marking which gears mesh. |
| `draw_shaft_system` | One shaft as a line, optionally with element detail. |
| `draw_shaft_detail` | 2D detail of one shaft: stepped profile, shoulder transitions, bearings and gears flush against the shaft edge. |
| `gear_anchor` | Real coordinates of a gear's symbol centre, for connector lines. |
| `recommended_figsize` | A width/height pair clamped to a sensible ratio for the given assembly. |

Choose the plotting axis that actually separates your shafts: a mesh at 270° offsets the shaft position along z, and plotting against y would stack every shaft at zero.

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
| `ExternalMoment` | Applied bending moment at an orientation, decomposed into My and Mz. |
| `DistributedRadialLoad` | Transverse load over an interval, with uniform or callable intensity and direction. |
| `LoadingProfile` | Fatigue cycle decomposition — stress ratio into mean and amplitude factors. |

`RadialLoad` and `ExternalMoment` expose their plane components and a `component(plane)` accessor. `DistributedRadialLoad` adds the resultant component, the centroid, the component intensity as a function of x, the bending-moment contribution (closed form when uniform, quadrature otherwise), and conversion to an equivalent point load when the direction is constant.

The `source` field is what makes re-resolving safe: `ShaftSystem.set_gear_loads()` replaces every load tagged `"gear_mesh"` and leaves user loads untouched.

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

Marin correction from the specimen endurance limit to a component value is applied by the shaft solver utilities, not here.

---

## Database

`axisforge/database/` is not part of `core/`, but its only consumers are. It holds standard tabular data — no solving, no geometry classes.

| Module | Provides |
|---|---|
| `shaft/keyway/Parallel/parallel_keyway.py` | DIN 6885 parallel keyway dimensions by shaft diameter, with tolerance bands pre-resolved into min/max columns so the caller can pick the critical value. Consumed by `Keyway.from_standard()`. |
| `shaft/keyway/Woodruff_key/iso3912.py` | ISO 3912 Woodruff keyway dimensions by shaft diameter, in two series. |
| `gears/.../LoadCapacity_data/csv_reader.py` | `lookup_KA` — application factor from ISO 6336-1 Table 4. |
| `gears/.../LoadCapacity_data/Kv_methodB.py` | `DynamicFactor` — ISO 6336-1 Method B: velocity coefficients and mesh stiffness factor, single tooth stiffness, the resonance ratio, and K_v across the subcritical, main-resonance, intermediate and supercritical regimes, with an `applicability()` check of the method's validity conditions. |
| `gears/.../LoadCapacity_data/Kv_methodC.py` | `DynamicFactorC` — ISO 6336-1 Method C: tolerance-class coefficients, the speed term, and K_v, with its own `applicability()`. |

These feed the gear load-capacity solver; the solver module itself is reserved and not yet written.