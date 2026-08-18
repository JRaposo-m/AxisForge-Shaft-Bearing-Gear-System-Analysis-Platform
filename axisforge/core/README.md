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

The bearing package is structured in three layers: a catalogue base class, type-specific geometry classes, and concrete subtypes — one file per concern.

```
Bearings/
├── bearing.py          ← Bearing base: catalogue data, ISO 281, mounting
├── bearing_types.py    ← BearingType enum
├── bearing_factory.py  ← make_bearing() — type-dispatch factory
├── geometry/
│   ├── base.py         ← BearingGeometry (ABC)
│   ├── ball_geometry.py    ← BallBearingGeometry  (point contact, ISO/TS 16281)
│   └── roller_geometry.py  ← RollerBearingGeometry (line contact, ISO/TS 16281)
└── subtypes/
    ├── deep_groove_ball.py     ← DeepGrooveBallBearing
    ├── angular_contact.py      ← AngularContactBallBearing
    ├── cylindrical_roller.py   ← CylindricalRollerBearing
    └── ...                     ← tapered/spherical roller (Phase 2)
```

The ISO/TS 16281 *solvers* that consume these objects live in a mirrored contact-type split under `solvers/machine_elements/bearings/ISO_16281/` — see [`solvers/README.md`](../solvers/README.md#solvers--bearings-isots-16281).

**`bearing_types.py`**

| Class | Purpose |
|-------|---------|
| `BearingType` | Enum of rolling bearing families (`DEEP_GROOVE_BALL`, `ANGULAR_CONTACT`, `CYLINDRICAL_ROLLER`, `TAPERED_ROLLER`, `SPHERICAL_ROLLER`). Determines life exponent p in ISO 281 (p=3 ball, p=10/3 roller). |

**`bearing.py` — `Bearing`**

Catalogue-level base class. Carries ISO 281 rating data, mounting arrangement, and geometry attribute slots (initialised to `None`; populated by subclass `setup_internal_geometry`). Solvers always access geometry via `bearing.Dw`, `bearing.ri`, `bearing.cp`, `bearing.alpha_0`, `bearing.gamma`, etc. — the interface is uniform regardless of subtype.

Contact angle is **not** a constructor input here — `alpha_0` (the real, geometry-derived contact angle, radians) is owned entirely by each subtype's `setup_internal_geometry()`, mirrored from its `Geometry` object. An earlier `contact_angle_deg` constructor argument was removed: it was never consumed by any solver or geometry calculation (everything reads `bearing.alpha_0`), so keeping it around only risked silently disagreeing with the real, computed angle.

- Constructor: bore `d`, outer `D`, width `b`, ratings `C`/`C0`, factors `X`/`Y`, `arrangement` (`"locating"` / `"floating"` / `"non-locating"`), `label`, `position`.
- `is_locating()` — True if the bearing restrains axial displacement.
- `equivalent_dynamic_load(Fr, Fa)` — P = X·Fr + Y·Fa (ISO 281).
- `has_internal_geometry()` — True if `setup_internal_geometry()` has been called.
- `validate()` / `validate_or_raise()` — catalogue-level geometry and factor checks.
- `summary()` / `__repr__()`.

**`bearing_factory.py` — `make_bearing(bearing_type, **kwargs)`**

Instantiates the correct subclass for a given `BearingType`. Useful when the type comes from a database, config file, or GUI — i.e. when the subclass is not known at write time. Currently registered: `DEEP_GROOVE_BALL` → `DeepGrooveBallBearing`, `CYLINDRICAL_ROLLER` → `CylindricalRollerBearing`; any other `BearingType` raises `NotImplementedError` (loudly, not a silent skip) until its subtype is implemented.

> **Open item:** `AngularContactBallBearing` now exists (see below) but hasn't been confirmed registered here yet — check `bearing_factory.py` and add `ANGULAR_CONTACT → AngularContactBallBearing` if it's still missing.

```python
from axisforge.core.machine_elements.Bearings import make_bearing, BearingType

b = make_bearing(BearingType.DEEP_GROOVE_BALL, d=20, D=47, C=12700, ...)
```

**`geometry/base.py` — `BearingGeometry` (ABC)**

Abstract contract for internal geometry classes. Defines `setup(**kwargs)`, `hertz_spring_constant()`, and `load_deflection_exponent` (3/2 for ball, 10/9 for roller).

**`geometry/ball_geometry.py` — `BallBearingGeometry`**

Point contact geometry for ball bearings. ISO/TS 16281 eq.(2)–(11).

- `setup(ri, re, Dw, Dpw, Z, E, nu, *, s=None, alpha_0_deg=None)` — exactly one clearance input required. Computes A = ri+re−Dw, then either `alpha_0 = arccos(1 − s/2A)` or `s = 2A·(1 − cos α₀)`, then Ri and φ_j. Invalidates the cached `gamma` on every call.
- `gamma` *(property)* — γ = Dw·cos(α₀)/Dpw, the ball-to-pitch-diameter ratio feeding the curvature formulas below. Lazily computed and cached on first access (purely geometric — nothing here changes after `setup()`). At α₀ = 90° (pure thrust) the cos(α) term is dropped and γ reduces to Dw/Dpw directly, rather than trusting floating-point `cos(π/2)` (~6e-17, sign not guaranteed) — same convention as `RollerBearingGeometry.gamma`.
- `curvature_sum_inner()` / `curvature_sum_outer()` — Σρ, eq.(5)/(6), reads `self.gamma`.
- `curvature_diff_inner()` / `curvature_diff_outer()` — F(ρ), eq.(7)/(8), reads `self.gamma`.
- `hertz_spring_constant()` — c_p [N/mm^(3/2)], eq.(9)–(11) via elliptic integrals and `brentq`.

**`geometry/roller_geometry.py` — `RollerBearingGeometry`**

Line contact geometry for cylindrical (and, in future, tapered/spherical) roller bearings. ISO/TS 16281 eq.(34)–(35), lamina positions per §5.2.2. Deliberately knows only generic line-contact math — no catalog data, no family-specific reference relations (§6 profile geometry is out of scope here; see `CylindricalRollerBearing.P_xk` below for why it lives on the subtype instead).

- `setup(Dwe, Lwe, Dpw, Z, s, n_s=30, alpha_0_deg=0.0)` — caches geometry; computes lamina midpoints `x_k` (§5.2.2, strictly inside (−Lwe/2, Lwe/2)) and roller angular positions `phi_j` (one per rolling element, not per lamina). `alpha_0_deg` is accepted for future tapered/spherical subclasses but stays 0.0 for a radial cylindrical roller bearing (NU/N-type). Invalidates the cached `gamma`.
- `gamma` *(property)* — γ = Dwe·cos(α₀)/Dpw, reusable by the ISO/TS 16281 capacity formulas (`RollerElementCapacity`) and by ISO 281. Lazily computed and cached; same α₀=90° → Dwe/Dpw convention as the ball side.
- `hertz_spring_constant()` — line-contact spring constant c_L [N/mm^(10/9)], eq.(35), plus the per-lamina spring constant c_s = c_L/n_s, eq.(37). Requires `setup()` first; caches `self.cL`, `self.cs`.

**`subtypes/deep_groove_ball.py` — `DeepGrooveBallBearing`**

Concrete DGBB subtype. Delegates all internal geometry to `BallBearingGeometry` and mirrors computed attributes onto `self` for uniform solver access.

A DGBB is specified by its catalog clearance class, not by a free contact angle (that's the angular-contact idiom) — so unlike the generic `BallBearingGeometry.setup()` underneath it, `setup_internal_geometry()` here accepts **only** `s`. Passing `alpha_0_deg` raises `TypeError` (it's simply not a parameter). Groove radii `ri`/`re` are likewise not inputs — always derived from `reference_raceway_radii(Dw)` (ISO/TS 16281 §6.3 eq.72–73), so a `DeepGrooveBallBearing` can't be built with geometry that contradicts what actually makes it this subtype.

- `reference_raceway_radii(Dw)` *(static)* — `(ri, re) = (0.52·Dw, 0.53·Dw)`, ISO/TS 16281 §6.3 eq.(72)–(73).
- `setup_internal_geometry(Dw, Dpw, Z, E, s, nu=0.3)` — populates `self.ri`, `self.re`, `self.Dw`, `self.Dpw`, `self.Z`, `self.s`, `self.E`, `self.nu`, `self.A`, `self.alpha_0`, `self.Ri`, `self.phi_j`, `self.gamma`.
- `compute_hertz_point_contact()` — returns and caches `self.cp` [N/mm^(3/2)].
- `validate()` — extends base validation with geometry guards (ri > Dw/2, re > Dw/2, s ≥ 0).

```python
from axisforge.core.machine_elements.Bearings import DeepGrooveBallBearing

b = DeepGrooveBallBearing(d=20, D=47, b=14, C=12700, C0=6550,
                           designation="6204", position=20.0,
                           arrangement="locating", label="brg1a")

b.setup_internal_geometry(Dw=7.94, Dpw=33.5, Z=8, E=206000, s=0.010)

cp = b.compute_hertz_point_contact()
```

**`subtypes/angular_contact.py` — `AngularContactBallBearing`**

Point contact, same family of formulas as `DeepGrooveBallBearing` (both built on `BallBearingGeometry`) — the difference is purely in how the bearing is idiomatically specified. An ACB is specified by its nominal (free) contact angle (a catalog property, e.g. "7208B" → 40°); clearance is the secondary, derived quantity. Mirrors DGBB's split exactly, flipped: `setup_internal_geometry()` accepts **only** `alpha_0_deg` (not `s`), and `ri`/`re` are likewise not inputs — always `reference_raceway_radii(Dw)` (same §6.3 eq.72–73 formula, deliberately duplicated rather than imported from the DGBB file — each subtype owns its own reference geometry).

Carries axial load (unlike DGBB in pure radial service) — commonly the locating bearing on a shaft, alone or in a paired arrangement (DB/DF/DT). Pairing arrangements are out of scope; this class models a single ring set.

- `reference_raceway_radii(Dw)` *(static)* — same eq.(72)–(73) as DGBB.
- `setup_internal_geometry(Dw, Dpw, Z, E, alpha_0_deg, nu=0.3)` — populates the same attribute set as DGBB (`ri`, `re`, `Dw`, `Dpw`, `Z`, `s`, `E`, `nu`, `A`, `alpha_0`, `Ri`, `phi_j`, `gamma`), just entered from the contact-angle side.
- `compute_hertz_point_contact()` — returns and caches `self.cp`.
- `validate()` — geometry guards as DGBB, plus `alpha_0 > 0` (checked once `has_internal_geometry()` is true — `alpha_0` doesn't exist before `setup_internal_geometry()` runs).

```python
from axisforge.core.machine_elements.Bearings import AngularContactBallBearing

b = AngularContactBallBearing(d=40, D=80, b=18, C=35000, C0=26000,
                               designation="7208B", position=50.0)

b.setup_internal_geometry(Dw=11.5, Dpw=60.0, Z=13, E=206000, alpha_0_deg=40.0)

cp = b.compute_hertz_point_contact()
```

**`subtypes/cylindrical_roller.py` — `CylindricalRollerBearing`**

NU/N-type cylindrical roller bearing — line contact, zero nominal contact angle, no axial capacity. Delegates all internal geometry to `RollerBearingGeometry` and mirrors computed attributes onto `self`, mirroring `DeepGrooveBallBearing`'s split with `BallBearingGeometry`. Constructed with `arrangement="floating"` by default, and also accepts `arrangement="non-locating"` — an NU/N-type bearing has no flange to react axial load, so the only thing it can never be is `"locating"`; passing that raises rather than silently building an invalid configuration that would only surface later as a `warn_if_floating_loaded()` warning.

- `setup_internal_geometry(Dwe, Lwe, Dpw, Z, s, n_s=30, alpha_0_deg=0.0)` — populates `self.Dwe`, `self.Lwe`, `self.Dpw`, `self.Z`, `self.s`, `self.n_s`, `self.alpha_0`, `self.x_k`, `self.phi_j`, `self.gamma`.
- `compute_line_contact_spring_constant()` — returns and caches `self.cL` [N/mm^(10/9)] (and `self.cs`, per-lamina).
- `P_xk` *(property)* — the ISO/TS 16281 §6.2 reference roller profile P(x_k), eq.(42)–(44) (crowning depth, subtracted as 2·P(x_k) from the raw lamina deflection so a purely cylindrical roller's theoretical edge-stress singularity doesn't appear in the load-distribution model). Lazily computed and cached on first access; purely geometric (`x_k`, `Dwe`, `Lwe` don't change after `setup_internal_geometry()`). Lives on the subtype rather than on `RollerBearingGeometry` because it's family-specific (tapered/spherical roller profiles use a different formula, eq.77-style), consistent with the geometry base class's "no §6 reference relations" scope.
- `has_internal_geometry()` — sentinel: `Dwe is not None`.

```python
from axisforge.core.machine_elements.Bearings import CylindricalRollerBearing

b = CylindricalRollerBearing(d=20, D=47, b=14, C=28_500.0, C0=22_000.0,
                              designation="NU204", position=100.0)
b.setup_internal_geometry(Dwe=6.5, Lwe=6.0, Dpw=33.5, Z=12, s=0.015, n_s=40)

cL = b.compute_line_contact_spring_constant()
```

Consumed by `ISO16281RollerSolver` (`solvers/machine_elements/bearings/ISO_16281/Roller_Bearing/roller_bearing.py`) via `BearingType.CYLINDRICAL_ROLLER`, dispatched through `RollingBearingSolver` alongside any `DEEP_GROOVE_BALL`/`ANGULAR_CONTACT` bearings on the same shaft. `_elements()` there now reads the cached `bearing.P_xk` directly instead of recomputing the profile every root-solve iteration.

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