# AxisForge

**Modular CAE-style platform for mechanical transmission system analysis.**

AxisForge is a deterministic, solver-centric engineering platform for the analysis of shaft–bearing–gear systems. It is built around physical first principles and traceable standards — every quantity is explainable, every result linked to an equation and a reference. Solvers are fully independent of any GUI layer, expose all intermediate quantities for inspection, and are validated against textbook and standard reference cases.

---

## Table of Contents

- [What it does](#what-it-does)
- [Stack and units](#stack-and-units)
- [Repository layout](#repository-layout)
- [Importing](#importing)
- [Analysis pipeline](#analysis-pipeline)
- [Documentation](#documentation)
- [Design principles](#design-principles)
- [Roadmap](#roadmap)
- [References](#references)

---

## What it does

| Capability | Standard / method | Where |
|---|---|---|
| Multi-shaft power flow and mesh load injection | ISO 21771 force resolution over a single-source DAG | `core/mechanical_system/` |
| 1D FEM shaft deflection and internal forces, two planes | Timoshenko beam, selective integration | `mesh/`, `solvers/.../shaft/` |
| Mesh convergence assessment | Richardson extrapolation + Grid Convergence Index | `solvers/mesh/` |
| Stress concentration at shoulders and keyways | Peterson / Shigley / Neuber | `solvers/.../static/shaft_post_processor.py` |
| Internal rolling element load distribution | ISO/TS 16281 §4 (point contact), §5 (line contact) | `solvers/.../bearings/ISO_16281/` |
| Per-element and per-lamina capacity, equivalent load, L10r | ISO 281, ISO 1281-1, ISO/TS 16281 | `core/.../bearings/families/`, `solvers/.../ISO_16281/` |
| Basic rating life and static safety | ISO 281 | `solvers/.../bearings/life.py` |
| Spur, helical and internal gear geometry and mesh forces | ISO 21771, ISO 53, KHK, MAAG | `core/.../gears/`, `solvers/.../gears/` |
| Planetary train kinematics and ideal torque split | Willis equation, Arnaudov & Karaivanov | `core/.../gears/parallel_axis/planetary_gear/` |
| Dynamic factor K_v data layer | ISO 6336-1 Method B and Method C | `database/gears/` |
| Keyway geometry lookup | DIN 6885, ISO 3912 | `database/shaft/keyway/` |
| Assembly visualisation | matplotlib line schematic | `core/mechanical_system/.../schematic.py` |
| Interactive script runner and exporters | PySide6 | `ui/` |

Fatigue, lubrication and failure susceptibility are future phases — see [Roadmap](#roadmap).

---

## Stack and units

**Python 3.11+** · NumPy · SciPy · matplotlib · PySide6 · pytest. SQLite is planned for the data layer.

Units are consistent internally:

| Quantity | Unit |
|---|---|
| Length, position, deflection | mm |
| Force | N |
| Bending moment | N·mm |
| Torque through the gear system | N·m (converted at the shaft boundary) |
| Stress | MPa |
| Angles | degrees at the interface, radians internally |
| Speed | rpm at the interface, rad/s internally |

Console output in scripts and fixtures is **ASCII only** (Windows PowerShell, cp1252).

---

## Repository layout

```
axisforge/
├── config.py                                   Tolerances, solver defaults, global constants
├── core/                                       Domain model — geometry and data, no solving
│   ├── loads.py
│   ├── materials.py
│   ├── machine_elements/
│   │   ├── shaft/                              Shaft, ShaftSection, Shoulder, Keyway
│   │   ├── bearings/                           Bearing, BearingCatalog, BearingFamily
│   │   │   └── families/                       ball_bearing/ · roller_bearing/
│   │   └── gears/parallel_axis/                gear_properties/ · gear_meshing/ · planetary_gear/
│   └── mechanical_system/parallel_axis/
│       ├── spur_helical/                       ShaftSystem, GearElement,
│       │                                       SpurHelicalMeshLink, SpurHelicalGearSystem
│       └── schematic.py
├── database/                                   Standard tabular data — no solving
│   ├── shaft/keyway/                           DIN 6885, ISO 3912
│   └── gears/SpurHelicalGears/LoadCapacity_data/   ISO 6336-1 K_A, K_v
├── mesh/shaft/                                 1D mesh generation and beam elements
│   ├── mesh_generation/                        Mesh1D, Grader
│   └── element_type/                           Elem, TimoshenkoBeam
├── solvers/
│   ├── machine_elements/
│   │   ├── shaft/oneD_analysis/                FEM, results library, post-processing
│   │   ├── bearings/ISO_16281/                 Ball_Bearing/ · Roller_Bearing/
│   │   ├── bearings/life.py
│   │   └── gears/                              GearSolver, geometry helpers
│   ├── mesh/                                   MeshConvergenceStudy, RichardsonGCI
│   └── lubrification/                          Reserved
├── models/                                     Result containers
├── ui/                                         PySide6 application
├── fixtures/                                   Reusable analysis-script building blocks
└── tests/                                      Unit / regression / validation tests
```

---

## Importing

Every package publishes its public surface in its own `__init__.py`. Import from the package, not from the file inside it:

```python
from axisforge.core.machine_elements.shaft            import Shaft, ShaftSection, Shoulder
from axisforge.core.machine_elements.bearings         import Bearing, BearingCatalog
from axisforge.core.machine_elements.bearings.families import DeepGrooveBallFamily
from axisforge.core.machine_elements.gears.parallel_axis import (
    SpurHelicalGear, SpurHelicalGearMeshing,
)
from axisforge.core.mechanical_system.parallel_axis.spur_helical import (
    ShaftSystem, GearElement, SpurHelicalMeshLink, SpurHelicalGearSystem,
)
from axisforge.mesh.shaft.mesh_generation            import Mesh1D, Grader
from axisforge.solvers.machine_elements.shaft.oneD_analysis.FEM_solvers import SimpleFEMSolver
from axisforge.solvers.machine_elements.shaft.oneD_analysis.static      import (
    ShaftResultsReader, SimpleFEMResultsLibrary,
)
from axisforge.solvers.machine_elements.bearings.ISO_16281 import (
    RollingBearingSolver, BearingResultsLibrary,
)
from axisforge.solvers.mesh import MeshConvergenceStudy
```

Three properties of this surface are worth knowing:

- **Imports are lazy.** A package's `__init__.py` executes nothing heavy until a name is actually used, so importing a package you only partly need costs nothing.
- **Roll-ups cascade.** A parent package forwards to its child package, never to a leaf module, so a module rename never propagates past one file.
- **The surface is introspectable.** `dir(package)` and `package.__all__` list what a package offers without importing it. This is what lets `fixtures/capabilities/` build an import set from a declaration — see [`fixtures/README.md`](axisforge/fixtures/README.md#capabilities).

Two surfaces are deliberately narrow:

- `bearings` exports only `Bearing`, `BearingCatalog`, `BearingFamily`, `BearingType`. The concrete families live in `bearings.families` so that adding a family changes nothing in `core/`.
- `ISO_16281` exports only the orchestration layer. `Ball_Bearing` and `Roller_Bearing` are imported explicitly, because point and line contact have different result types and mixing them in one namespace hides which model a name belongs to.

---

## Analysis pipeline

The canonical solve sequence for one shaft:

```python
# 1 — Assemble the multi-shaft gear system and resolve power flow
gearbox = SpurHelicalGearSystem(shafts, links, label="drivetrain")
gearbox.resolve(P_W, rpm_in, rotation_dir_source=1)

# 2 — Mesh and solve the shaft
fem = SimpleFEMSolver()
fem.solve(shaft_system, extra_mandatory=gear_grade_nodes)

# 3 — Post-process the FEM into the results library
library = SimpleFEMResultsLibrary()
ShaftResultsReader(fem, shaft_system).read(library)

# 4 — Internal bearing load distribution (ISO/TS 16281)
solver    = RollingBearingSolver()
load_dist = solver.solve(shaft_system, bearings, library)

# 5 — Capacity, equivalent load and stiffness, recorded per bearing label
results = solver.postprocess_and_record(
    shaft_system, bearings, library,
    catalog={"brg1a": {"capacity": {"Cr": 29_600.0},
                       "dynamic_equivalent_load": {"inner_rotating": True}}},
    load_distribution=load_dist,
)
bundle = results.get("brg1a")

# 6 — Stress concentration on the shaft
ppr = ShaftPostProcessor(shaft_system, library.get(shaft_system.name)).process()
```

Two contracts govern this sequence:

- **One solver instance per shaft.** `SimpleFEMSolver.solve()` publishes its results as public attributes; a second call overwrites them.
- **Capacity belongs to the bearing, not to the solver.** Every family owns its own ISO formulas and is called through the bearing: `bearing.family.per_element_dynamic_capacity(bearing, Cr=...)`, and for roller families `bearing.family.per_lamina_dynamic_capacity(bearing, Q_ci, Q_ce)`.

A shaft may carry a mixed bearing set — different contact types and different row counts — in the same call. `RollingBearingSolver` resolves each bearing to a solver from the capabilities its family declares.

---

## Documentation

Each top-level package has its own README with the module-by-module reference. This document is the entry point; the detail lives next to the code it describes.

| Package | Covers |
|---|---|
| [`axisforge/core/README.md`](axisforge/core/README.md) | Shaft · Bearings and families · Gears · Systems · Meshing · Loads · Materials · Database |
| [`axisforge/solvers/README.md`](axisforge/solvers/README.md) | Shaft FEM · Post-processing · Bearings (ISO/TS 16281, ISO 281 life) · Gears · Mesh convergence |
| [`axisforge/mesh/README.md`](axisforge/mesh/README.md) | Node generation, grading, beam elements |
| [`axisforge/models/README.md`](axisforge/models/README.md) | Result containers |
| [`axisforge/ui/README.md`](axisforge/ui/README.md) | PySide6 application and exporters |
| [`axisforge/fixtures/README.md`](axisforge/fixtures/README.md) | Analysis-script building blocks · capability selection |

---

## Design principles

- **Low coupling, high cohesion.** Solvers depend only on explicit result containers, never on each other's internals.
- **Declared surface, lazy import.** A package states what it offers and imports nothing until asked.
- **No hidden state.** Every intermediate quantity is a public attribute; a solver exposes its full working for inspection.
- **GUI-independent solvers.** The analysis core runs headless; the UI is a thin consumer.
- **Deterministic and explainable.** No black-box methods, no probabilistic life prediction, no machine learning.
- **Composition over inheritance.** A planetary train composes two pair-meshing objects; a bearing composes a family rather than subclassing one.
- **Capacity belongs to the element.** A solver never carries its own copy of a standard's formula.
- **Fail fast on geometry.** Validation happens at construction; invalid geometry is never silently accepted.
- **One file per concern.** Each subtype, function group, solver and result shape lives in its own module.
- **Flag, do not silently fix.** A suspected discrepancy against a standard is documented in place, never quietly corrected.
- **`validate()` returns, `validate_or_raise()` raises.** Every domain object follows this pair.

---

## Roadmap

| Phase | Focus | Status |
|---|---|---|
| 1 | Shaft FEM · mesh convergence · gear force integration | Complete |
| 2 | Bearing families, capability dispatch, ISO/TS 16281 load distribution and L10r | Complete for radial ball and radial roller; thrust roller load distribution outstanding |
| 3 | ISO 6336 gear load capacity — data layer in place, solver outstanding | In progress |
| 4 | Capability selection and fixture library completion | In progress |
| 5 | Fatigue — Goodman / Morrow / Miner | Planned |
| 6 | Lubrication — EHD film thickness, grease | Planned |
| 7 | Failure susceptibility scoring · SQLite data layer | Planned |
| 8 | GUI consolidation | Ongoing |

---

## References

- ISO/TS 16281:2008 — *Rolling bearings: Methods for calculating the modified reference rating life for universally loaded bearings*
- ISO 281:2007 — *Rolling bearings: Dynamic load ratings and rating life*
- ISO 1281-1:2021 — *Rolling bearings: Explanatory notes on ISO 281*
- ISO 76:2006 — *Rolling bearings: Static load ratings*
- ISO 21771:2007 — *Gears: Cylindrical involute gears and gear pairs*
- ISO 53:2013 — *Cylindrical gears for general engineering: standard basic rack tooth profile*
- ISO 6336-1/-2/-3/-5 — *Calculation of load capacity of spur and helical gears*
- DIN 6885 — *Parallel keys and keyways* · ISO 3912 — *Woodruff keys and keyways*
- Harris & Kotzalas, *Rolling Bearing Analysis*, 5th ed., Wiley
- Palmgren, *Grundlagen der Wälzlagertechnik*, 3rd ed., Franckh
- Shigley, *Mechanical Engineering Design*, 10th ed.
- Peterson, *Stress Concentration Factors*
- MAAG Gear Book, 2nd ed. · KHK Gear Technical Reference
- Henriot, *Traité théorique et pratique des engrenages*
- Arnaudov & Karaivanov, *Planetary Gear Trains*