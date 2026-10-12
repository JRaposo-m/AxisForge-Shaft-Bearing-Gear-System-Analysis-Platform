# axisforge.core

Domain model of AxisForge: shaft geometry, bearings and their families, gears and gear meshing,
single-shaft and multi-shaft systems, applied loads and materials. A domain object describes a
physical thing, can validate itself and owns the standard formulae that depend only on its own
geometry (for example the basic dynamic load rating of a bearing). The package runs no solver,
stores no solver result and draws nothing; a `ShaftSystem` holds the loads applied to it and
`SpurHelicalGearSystem.resolve` writes the mesh loads onto its shafts. It imports nothing from
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
    gears/parallel_axis/
      gear_properties/          geometry of one external or internal involute gear
      gear_meshing/             external-external and external-internal pair, tooth forces
      planetary_gear/           single-stage epicyclic train built from two pairs
  mechanical_system/parallel_axis/
    spur_helical/
      shaft_system.py           GearElement, ShaftSystem (bearings, gears, loads on one shaft)
      gear_system.py            SpurHelicalMeshLink, SpurHelicalGearSystem (power-flow graph)
```

Tests of the bearing code are in `bearings/tests/` and `bearings/families/tests/`.

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
| Gear geometry | `SpurHelicalGear(mn, z, ...)`, `InternalGear(mn, z, ...)` | gear object |
| Gear pair | `SpurHelicalGearMeshing(gear1, gear2, ...)`, `InternalGearMeshing(...)`, `.forces(T_in)` | pair object; dict of forces |
| Planetary train | `PlanetaryGearTrainMeshing(sun_gear, planet_gear, ring_gear, k, ...)` | train object |
| Loads | `RadialLoad`, `AxialLoad`, `TorqueLoad`, `ExternalMoment`, `DistributedRadialLoad` | load objects; `component(LoadPlane)` |
| One shaft with its elements | `ShaftSystem(shaft, ...)`, `.add_bearing`, `.add_gear`, `.add_load`; `GearElement(gear, role, ...)` | `ShaftSystem` |
| Several shafts joined by meshes | `SpurHelicalMeshLink(...)`, `SpurHelicalGearSystem(shafts, links)`, `.resolve(P, rpm, rotation_dir_source)` | `None`; shaft positions and mesh loads are set on every `ShaftSystem` |

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
- A `Bearing` refuses attribute writes after `assemble()`. `BearingCatalog` and the material
  blocks are frozen dataclasses.
- Bearing families register themselves with `@register_family`; `@point_contact`,
  `@line_contact` and `@thrust` classify them. A new family is a new registered class; no
  table is edited.
- `Load.source` is `"user"`, `"gear_mesh"` or `"bearing_reaction"`; `ShaftSystem.set_gear_loads`
  replaces the `"gear_mesh"` loads and keeps the others.
- Fan-out in a gear system (one driver gear, several meshes) is detected by the identity of the
  `GearElement` object.
- The material registry is process-local and starts empty: `get_material` finds only what the
  caller has registered.
- `core` imports `axisforge.config`, `axisforge.database`, NumPy and SciPy.

## Standards

Library keys as written in the docstrings (D-0023); clauses and equations are in the docstrings.

| Key | Used for |
|---|---|
| `ISO_TS_16281_2008` | Contact spring constants and load-rating calculators of the bearing families |
| `ISO_TR_1281` | Basic dynamic load rating, multi-row combination |
| `ISO_6336-1_2019` | Involute gear geometry, meshing, gear-system forces |
| `Harris2007_vol1` | Rolling bearing geometry and load rating |
| `Harris2007_vol2` | Shear modulus from the elastic constants (`materials/base.py`) |
| `Onate2013_vol2` | Section properties of the shaft; elastic constants |
| `Cao2004` | Moment components and rotation convention of the load classes |
