# axisforge.core

Domain model of AxisForge: shaft geometry, bearings and their families, gears and gear meshing,
single-shaft and multi-shaft systems, applied loads and materials. A domain object describes a
physical thing, can validate itself and owns the standard formulae that depend only on its own
geometry (for example the basic dynamic load rating of a bearing). The package runs no solver,
stores no solver result and draws nothing; a `ShaftSystem` holds the loads applied to it and
`Transmission.resolve` returns loaded copies of its shafts with the mesh loads. It imports nothing from
`mesh`, `solvers`, `results` or `outputs`. Detail of every class is in its docstring.

## Contents

```
core/
  loads.py                      load primitives and the LoadPlane decomposition key
  materials/
    base.py                     Material, property blocks, process-local registry
  machine_elements/
    shaft/
      shaft.py                  stepped shaft, sections, shoulders, keyways, section properties
    bearings/
      base.py                   BearingCatalog (catalogue data), BearingFamily (contract)
      bearing.py                Bearing (catalogue + family geometry, immutable once assembled)
      contact_models/           ISO/TS 16281 contact spring constants, ContactAnalysis switch
      families/
        family.py               concrete families, @register_family and classification decorators
        geometry.py             raceway and rolling-element curvature geometry
        capacity.py             basic dynamic load rating calculators (point and line contact)
    gears/
      base.py                   GearFamily (contract), @register_gear_family, Gear, GearPair (immutable once assembled),
                                MeshForces (forces of one mesh; mesh frame and sign table)
      cylindrical.py            CylindricalGearData, CylindricalFamily: external and internal involute gears
                                (z < 0), their pair, checks and tooth forces (`forces`, `mesh_forces`), in the
                                order of ISO 21771
      cylindrical_capacity.py   placeholder (ISO 6336 element factors): contracts only, nothing computed
      parallel_axis/planetary_gear/   single-stage epicyclic train built from two pairs
  mechanical_system/
    shaft_system.py             GearElement, ShaftSystem (bearings, gears, loads on one shaft)
    transmission.py             mesh_forces_to_shaft_loads, PairStage, Transmission, ResolvedTransmission
                                (written power path)
```

Tests of the bearing code are in `bearings/tests/` and `bearings/families/tests/`; those of the
gears in `gears/tests/` and of the systems in `mechanical_system/tests/`.

## Entry points

All names below are re-exported through `axisforge.core` and `axisforge`, except the family
registry helper `available_families` (import it from `...bearings.families`) and the planetary
result types (import them from the module `planetary_gear_meshing`).

| Task | Call | Returns |
|---|---|---|
| Describe a shaft | `Shaft(label)`, `ShaftSection(length, diameter, ...)`, `Shaft.add_section(section)`, `Shaft.set_transition(index, Shoulder(...))` | `Shaft`; `validate()` returns a list of messages |
| Keyway from tables | `Keyway.from_standard(label, shaft_diameter, z_position, ...)` | `Keyway` |
| Section properties | `shaft.section_at(z)`, `diameter_at(z)`, `I_at(z)`, `J_at(z)`, `W_at(z)`, `Wt_at(z)` | section and index; floats |
| Register a material | `register(Material(...))`, `get_material(material_id)` | `Material` |
| Catalogue data of a bearing | `BearingCatalog(d, D, b, designation, label, position, arrangement)` | `BearingCatalog` |
| Assemble a bearing | `Bearing.assemble(family, catalog, geometry)` | immutable `Bearing`; `.C` [N] |
| Capacity of an assembled bearing | `bearing.family.dynamic_capacity(bearing)`, `per_element_dynamic_capacity(bearing, Cr=None)` (keyword `Cr` or `Ca`, by family) | `(Q_ci, Q_ce)` [N]; overall rating [N] |
| Switch on ISO/TS 16281 contact data | `geometry` with `contact=ContactAnalysis.ISO16281`, `e1`, `e2`, `nu1`, `nu2` | bearing with contact spring constants |
| Gear geometry | `Gear.assemble(CylindricalFamily(), CylindricalGearData(mn, z, x=..., b=..., ...))`; `z < 0` for an internal gear | immutable `Gear` |
| Gear pair | `GearPair.assemble(family, gear1=..., gear2=..., al=None, addendum_reduction=False, label="")` | immutable `GearPair` (no shaft, driver or torque) |
| Tooth forces of a pair | `pair.family.forces(pair, T, phi_deg, rotation_dir, driver="gear1")`, `T` in N m | dict of forces [N], output torque [N m] and force angles [deg]; the shaft loads are built from it |
| Forces of a mesh on its two gears | `pair.family.mesh_forces(pair, T1_Nm, rot1, hand1=1)` | `MeshForces`: `F_tw_N`, `F_r_N`, `F_a_N`, `F_bt_N`, `F_bn_N`, `F_t_nominal_N` [N], signed `on_1`, `on_2` in the mesh frame, `rho_mm` |
| Planetary train | `PlanetaryGearTrainMeshing(sun_gear, planet_gear, ring_gear, k, ...)` | train object |
| Loads | `RadialLoad`, `AxialLoad`, `TorqueLoad`, `ExternalMoment`, `DistributedRadialLoad` | load objects; `component(LoadPlane)` |
| One shaft with its elements | `ShaftSystem(shaft, ...)`, `.add_bearing`, `.add_gear`, `.add_load`; `GearElement(gear, role, ...)` | `ShaftSystem` |
| Several shafts joined by meshes | `PairStage(pair=..., shaft_a=..., gear_a=..., shaft_b=..., gear_b=..., phi_deg=...)`, `Transmission(shafts, path, source, sinks=())`, `.resolve(P_W, n_rpm, rotation)` | `ResolvedTransmission`: `shafts` (loaded `ShaftSystem` copies with positions and mesh loads, in the order given), `torque_Nm`, `rotation`, `speed_rpm`, `meshes`, `checks`, `warnings`; the given shafts are not modified |

Concrete bearing families (`DeepGrooveBallFamily`, `AngularContactFamily`, `SelfAligningBallFamily`,
`ThrustBallSingleRowFamily`, `ThrustBallMultiRowFamily`, `CylindricalRollerFamily`,
`ThrustCylindricalRollerFamily`, `RollerThrustMultiRowFamily`) are listed by `available_families()`.
The keys of the `geometry` dictionary are the keyword arguments of the family's
`assemble_geometry`; see the family docstrings.

```python
from axisforge.core.machine_elements.bearings import (
    Bearing, BearingCatalog, DeepGrooveBallFamily)

catalog = BearingCatalog(d=30.0, D=62.0, b=16.0, designation="6206", label="B1",
                         position=100.0, arrangement="locating")
bearing = Bearing.assemble(DeepGrooveBallFamily(), catalog,
                           dict(Dw=8.0, Dpw=40.0, Z=12, s=0.02, i=1))
print(bearing.C)   # basic dynamic radial load rating [N]
```

## Conventions

Axes, planes and signs follow the section Conventions of the root `README.md`; the
load decomposition is stated in the module docstring of `loads.py`.

- Length and position [mm], force [N], stress and modulus [MPa], density [kg/m^3].
- Angles are in degrees at the interface and carry the suffix `_deg` (`theta_deg`, `phi_deg`,
  `alpha_n_deg`, `beta_n_deg`); speed is in rpm at the interface and rad/s inside `resolve()`;
  power is in W.
- Torque in the gear path (`forces()`, `resolve()`) is in N m. The units of `TorqueLoad.magnitude`
  and `ExternalMoment.magnitude` are discussed in the Warnings of the `loads.py` docstring.
- The axial coordinate runs from the left end of the shaft to the right: `x` in `ShaftSystem` and
  in loads, `z` in `Shaft`. The `position` of a gear is the centre of its face width; that of a bearing is its axial coordinate along the shaft.
- `theta` is measured from +Y toward +Z, right-hand about +X; `Fy = F cos(theta)`,
  `Fz = F sin(theta)`; plane XY takes `Fy`, plane XZ takes `Fz`.
- An internal gear has a negative tooth number `z`; its diameters are stored as magnitudes.
- `ShaftSection`, `Shoulder`, `Keyway`, `BearingCatalog` and the material blocks raise
  `ValueError` at construction. Gears, shafts, loads and systems expose `validate() -> list[str]`
  and, for most, `validate_or_raise()`.
- A `Bearing`, a `Gear` and a `GearPair` refuse attribute writes after `assemble()`.
  `BearingCatalog`, `CylindricalGearData` and the material blocks are frozen dataclasses.
- `Gear.assemble` and `GearPair.assemble` do not run `validate()`; `validate()` of an internal
  pair is never empty (open questions of the gears refactor).
- Bearing families register themselves with `@register_family`; `@point_contact`,
  `@line_contact` and `@thrust` classify them. A new family is a new registered class; no
  table is edited.
- `Load.source` is `"user"`, `"gear_mesh"` or `"bearing_reaction"`; `ShaftSystem.with_loads`
  returns a copy with the given loads added after the loads of the shaft.
- The power path of a `Transmission` is used as written (ordered stages, one source); a stage
  takes `share` of the torque entering its input shaft (default 1.0).
- `ResolvedTransmission.speed_rpm` holds the speed of every shaft; `resolve` does not write it on
  `ShaftSystem.speed_rpm` of the loaded copies. The stored mesh loads are not `MeshForces`: the
  differences are listed in the docstring of `mesh_forces_to_shaft_loads`, and
  `ResolvedTransmission.checks` holds the closure checks they pass and fail.
- The material registry is process-local and starts empty: `get_material` finds only what the
  caller has registered.
- `core` imports `axisforge.config`, `axisforge.database`, NumPy and SciPy.

## Standards

Library keys as written in the docstrings (D-0023); clauses and equations are in the docstrings.

| Key | Used for |
|---|---|
| `ISO_TS_16281_2008` | Contact spring constants and load-rating calculators of the bearing families |
| `ISO_TR_1281` | Basic dynamic load rating, multi-row combination |
| `ISO_21771_2007`, `ISO_53_1998` | Involute gear geometry and meshing, basic rack (`gears/cylindrical.py`) |
| `ISO_6336-1_2019` | Nominal tangential load, virtual number of teeth; planetary train and gear-system docstrings |
| `Harris2007_vol1` | Rolling bearing geometry and load rating |
| `Harris2007_vol2` | Shear modulus from the elastic constants (`materials/base.py`) |
| `Onate2013_vol2` | Section properties of the shaft; elastic constants |
| `Cao2004` | Moment components and rotation convention of the load classes |
