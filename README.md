# AxisForge

AxisForge is a deterministic, standards-based engine for the analysis of shaft-bearing-gear
systems. It computes stepped-shaft deflections and internal forces by the finite element method
(Timoshenko or Euler-Bernoulli beams, bending in two planes, axial load and torsion), the internal
load distribution and basic reference rating life of ball and roller bearings (ISO/TS 16281),
stress concentration at shoulders and keyways, mesh convergence, gear geometry and meshing forces,
and planetary train kinematics. Solvers run without an interface and return plain result
containers. Records, text and CSV of objects and results are produced by `outputs`; plots and
studies are outside this repository.

## Installation

Python 3.12 or later. From the repository root:

```
pip install -e .
python -m pytest axisforge
```

Tests are in `tests/` folders next to the code.

## Minimal example

One shaft on two deep-groove ball bearings under one transverse load: shaft FEM, then the ISO/TS
16281 bearing solver. Run from any directory after installation.

```python
from axisforge.core import (Material, IsotropicElastic, register, Shaft, ShaftSection,
                            ShaftSystem, BearingCatalog, Bearing, RadialLoad)
from axisforge.core.machine_elements.bearings.families import DeepGrooveBallFamily
from axisforge.core.machine_elements.bearings.contact_models import ContactAnalysis
from axisforge.mesh import BeamModelSettings
from axisforge.solvers import RigidSupportFEMSolver, ISO16281BallSolver
from axisforge.outputs.solvers.bearings.bearing_analysis_result import analysis_record

register(Material(material_id="steel", density=7850.0,
                  elastic=IsotropicElastic(E=210000.0, poisson_ratio=0.3)))

shaft = Shaft("S1")
shaft.add_section(ShaftSection(length=300.0, diameter=40.0, material_id="steel"))
system = ShaftSystem(shaft, name="demo", speed_rpm=1000.0)
for label, x, arrangement in (("A", 50.0, "locating"), ("B", 250.0, "floating")):
    catalog = BearingCatalog(d=40.0, D=80.0, b=18.0, designation="6208",
                             label=label, position=x, arrangement=arrangement)
    geometry = dict(Dw=12.0, Dpw=60.0, Z=9, s=0.01, contact=ContactAnalysis.ISO16281,
                    e1=210000.0, e2=210000.0, nu1=0.3, nu2=0.3)
    system.add_bearing(Bearing.assemble(DeepGrooveBallFamily(), catalog, geometry))
system.add_load(RadialLoad(position=150.0, magnitude=2000.0, theta_deg=0.0, label="F"))

settings = BeamModelSettings(beam_theory="timoshenko", shear_theory="cowper",
                             integration_method="single_point")
shaft_results = RigidSupportFEMSolver(settings).solve(system)
print("mesh nodes:", len(shaft_results.x_nodes))

bearings = {b.label: b for b in system.bearings}
analyses = ISO16281BallSolver(postprocess=True).solve(system, bearings, shaft_results)
for label, analysis in analyses.items():
    rec = analysis_record(analysis)
    print(label, "Fr = %.1f N, delta_r = %.4f mm, L10r = %.0f Mrev, ok = %s"
          % (rec["Fr_N"], rec["delta_r_mm"], rec["L10r_Mrev"], rec["ok"]))
```

Output:

```
mesh nodes: 9
A Fr = 1000.0 N, delta_r = 0.0180 mm, L10r = 10230 Mrev, ok = True
B Fr = 1000.0 N, delta_r = 0.0180 mm, L10r = 10230 Mrev, ok = True
```

The solver takes the bearings as a dictionary keyed by label. A material is registered before
a section refers to it.

## Conventions

* Length mm, force N, moment N mm, stress MPa, section modulus mm^3, stiffness N/mm, density
  kg/m^3.
* Angles are degrees at the interface (suffix `_deg`) and radians inside the solvers and in every
  result container. Speed is rpm at the interface and rad/s inside.
* Torque in the gear path, in `TorsionSolver` and in `ShaftResults.T` is N m. The units of the
  torque and moment loads in `core/loads.py` are described in that module.
* `x` runs along the shaft axis from left to right. The plane XY takes the force
  component Fy; the plane XZ takes Fz.
* A radial load at angle `theta_deg` is measured from +Y toward +Z, right-hand about +X, with
  Fy = F cos(theta) and Fz = F sin(theta).
* A field suffixed `_xz` or `_xy` is the component of a quantity in that plane; the unsuffixed
  name is the resultant.
* A positive torque is counter-clockwise seen from +X, and torsion accumulates from left to
  right.
* Record keys carry the unit (`Fr_N`, `delta_r_mm`, `L10r_Mrev`); the exception is `T_Nmm` in
  the shaft FEM node records, whose value is in N m.
* Report files are written in UTF-8.

The conventions of each package are listed in its README. Units and signs are shared by all
packages and are not changed locally.

## Package map

All packages are under `axisforge/`.

| Package | Content | Link |
|---|---|---|
| `config` | Tolerances, solver defaults, geometry limits (a module) | `axisforge/config.py` |
| `core` | Domain model: shaft, bearings and families, gears, planetary train, shaft and gear systems, loads, materials | `axisforge/core/README.md` |
| `database` | Standard tables: DIN 6885 and ISO 3912 keyways, ISO 6336 factor data for gears | `axisforge/database/` |
| `mesh` | Node grid, grading, beam elements, `BeamModelSettings` | `axisforge/mesh/README.md` |
| `results` | Result containers filled by the solvers; no computation | `axisforge/results/README.md` |
| `solvers` | Shaft FEM, torsion, sub-model, stress concentration, ISO/TS 16281 bearing solvers, mesh convergence | `axisforge/solvers/README.md` |
| `outputs` | Records, text and CSV of core objects and results | `axisforge/outputs/README.md` |

`from axisforge import Name` exposes the public names of `core`, `mesh`, `results` and `solvers`;
`config` and `outputs` stay modules. Code inside the package imports from the subpackage that
defines the name.

The repository `AxisForge-Design-Studies` imports `axisforge`; `axisforge` does not import it.

## Layer rule

Imports are one-way. A package imports only the packages in its row.

| Package | Imports from axisforge at runtime |
|---|---|
| `config` | nothing |
| `database` | nothing; `Kv_methodB.py` and `Kv_methodC.py` import `axisforge.core` and are not importable as they stand |
| `core` | `config`, `database` |
| `mesh` | `core`, `config` |
| `results` | nothing (annotations only refer to `solvers`) |
| `solvers` | `core`, `mesh`, `results`, `config` |
| `outputs` | nothing; objects are read by attribute |

No solver imports another solver's internals; solvers share only the result containers. A new
edge between packages is a design decision.

## References

Library keys (D-0023, proposed); clauses and equations are in the docstrings.

* ISO_TS_16281_2008, ISO_TS_16281_2008_Cor1_2009
* ISO_TR_1281
* ISO_6336-1_2019
* Harris2007_vol1, Harris2007_vol2
* Onate2013_vol2
* Cao2004
* Zaretsky2016
* Hernot2000
