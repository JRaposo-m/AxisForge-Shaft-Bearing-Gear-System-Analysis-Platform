# AxisForge

**Modular CAE-style platform for mechanical transmission system analysis.**

AxisForge is a deterministic, solver-centric engineering platform for the analysis of shaft–bearing–gear systems. It is built on physical first principles and traceable standards: every quantity is explainable, every result linked to an equation and a reference. Solvers run headless, expose all intermediate quantities for inspection, and are validated against textbook and standard reference cases.

---

## Table of Contents

- [What it does](#what-it-does)
- [Status](#status)
- [Stack and units](#stack-and-units)
- [Conventions](#conventions)
- [Architecture](#architecture)
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
| 1D FEM shaft deflection and internal forces, two planes | Timoshenko **or Euler-Bernoulli** beam, selected once per shaft via `BeamModelSettings` | `mesh/`, `solvers/.../shaft/fem_solvers/` |
| Shaft torsion (torque, shear stress, twist angle) | Pure statics, run independently of the bending/axial solve | `solvers/.../shaft/static_solvers/torsion.py` |
| Submodel refinement with prescribed cut-node state | Lagrange multipliers | `solvers/.../shaft/fem_solvers/constraints/submodel_extraction.py` |
| Mesh convergence assessment | Richardson extrapolation + Grid Convergence Index | `solvers/mesh/convergence_solver.py` |
| Stress concentration at shoulders and keyways | Peterson / Shigley / Neuber | `solvers/.../shaft/static_solvers/postprocessing.py` |
| Internal rolling element load distribution, single row | ISO/TS 16281 §4 (point contact), §5 (line contact) | `solvers/.../bearings/load_distribution/single_row/` |
| Internal load distribution, multi-row thrust bearings | ISO/TS 16281, shared rigid-ring displacement | `solvers/.../bearings/load_distribution/multi_row/` |
| Per-element and per-lamina capacity, equivalent load, L10r | ISO 281, ISO 1281-1, ISO/TS 16281 | `core/.../bearings/families/`, the bearing postprocessing modules |
| Spur, helical and internal gear geometry and mesh forces | ISO 21771, ISO 53, KHK, MAAG | `core/.../gears/`, `solvers/.../gears/` |
| Planetary train kinematics and ideal torque split | Willis equation, Arnaudov & Karaivanov | `core/.../gears/parallel_axis/planetary_gear/` |
| Application and dynamic factor data layer | ISO 6336-1 Method B and Method C | `database/gears/` |
| Keyway geometry lookup | DIN 6885, ISO 3912 | `database/shaft/keyway/` |
| Assembly visualisation | matplotlib line schematic | `core/mechanical_system/.../schematic.py` |
| Construction and resolution text reports | fixed-width ASCII writers | `fixtures/*/text_report.py` |

Gear load capacity, fatigue, lubrication and failure susceptibility are future phases — see [Roadmap](#roadmap).

---

## Status

| Package | Contents | Status |
|---|---|---|
| `config` | Tolerances, solver defaults, global constants | Implemented |
| `core/` | Shaft, bearings and families, gears, meshing, planetary trains, systems, loads, materials | Implemented |
| `database/` | DIN 6885, ISO 3912, ISO 6336-1 K_A and K_v | Implemented — see the note under [Architecture](#architecture) |
| `mesh/` | 1D node generation, grading, Timoshenko **and Euler-Bernoulli** beam elements, `BeamModelSettings` | Implemented — Euler-Bernoulli is **no longer reserved**; see the open items in [`mesh/README.md`](axisforge/mesh/README.md#status) |
| `results/` | Result containers: shaft FEM (now including axial displacement, bending rotation and twist angle), single-row bearing load distribution | Implemented |
| `solvers/.../shaft/` | Stiffness assembly, rigid-*support* FEM (`RigidSupportFEMSolver`, renamed from `RigidBearingFEMSolver`), separate torsion solve, submodel, results reader, stress concentration | Implemented; `static_failure` reserved — see the open items in [`solvers/README.md`](axisforge/solvers/README.md#status) |
| `solvers/.../bearings/` | ISO/TS 16281 single-row and multi-row thrust, dispatch, postprocessing | Implemented |
| `solvers/.../gears/` | Geometry and mesh forces | Implemented; ISO 6336 load capacity reserved |
| `solvers/mesh/` | Mesh convergence, Richardson GCI (`convergence_solver.py`, renamed from `mesh_convergence_study.py`) | Implemented |
| `fixtures/construction/` | Object-building fixtures and construction reports | Implemented |
| `fixtures/studies/` | Shaft FEM study, mesh-convergence study, bearing load-distribution study, result libraries, combined studies report | Implemented; the mesh-convergence fixture currently calls a pre-rename solver API and does not import — see [`fixtures/README.md`](axisforge/fixtures/README.md#status) |

---

## Stack and units

**Python 3.11+** · NumPy · SciPy · matplotlib · pytest. SQLite is planned for the data layer.

| Quantity | Unit |
|---|---|
| Length, position, deflection, displacement | mm |
| Force | N |
| Bending moment, moment reaction | N·mm |
| Torque through the gear system | N·m (converted at the shaft boundary) |
| Stress | MPa |
| Section modulus | mm³ |
| Stiffness | N/mm |
| Angles | degrees at the interface, radians internally |
| Speed | rpm at the interface, rad/s internally |

Console and report output is **ASCII only** (Windows PowerShell, cp1252). Report files are written with an explicit UTF-8 encoding.

---

## Conventions

These are load-bearing across the whole platform. Changing one is a breaking change everywhere, not a local edit.

| Symbol | Meaning |
|---|---|
| `x` | Axial coordinate, increasing left → right |
| `XZ` | Horizontal plane — carries the tangential gear force `Wt` |
| `XY` | Vertical plane — carries the radial gear force `Wr`, opposed to gravity |
| Torsion | Accumulates left → right; positive is counter-clockwise viewed from `+x` |

A radial load at angle `theta_deg` (measured `+Y → +Z`, right-hand about `+X`, magnitude ≥ 0) contributes to **both** planes: `component(XY)` gives `Fy` (vertical, `Wr`) and `component(XZ)` gives `Fz` (horizontal, `Wt`). Solvers must not pre-filter by plane.

Any field suffixed `_xz` or `_xy` is one plane's component of a quantity whose resultant is stored unsuffixed.

---

## Architecture

Five layers, with a strictly one-way dependency:

```
                config
                   │
     ┌─────────────┼─────────────┐
     ▼             ▼             ▼
  database  ─→   core   ←─     mesh          domain: geometry and data, no solving
                   │             │
                   └──────┬──────┘
                          ▼
                      solvers                 computes
                          │  writes
                          ▼
                      results                 result shapes only, no computation
                          │  reads
                          ▼
                     fixtures                 studies, libraries, reports
```

Rules, in force:

- `core/` depends on `config`, `database/`, NumPy and the standard library. It never imports a solver.
- `results/` imports NumPy and the standard library only — nothing from `axisforge` at all.
- `solvers/` depends on `core/`, `mesh/`, `config` and `results/`. **No solver imports another solver's internals** — only the shared result containers.
- `fixtures/` depends on everything below it. **Nothing in `axisforge` imports from `fixtures/`.**

`results/` is the narrow waist: a solver depends on the container it writes, a consumer on the container it reads, and neither depends on the other.

> **Open item.** `database/gears/.../Kv_methodB` and `Kv_methodC` import from `core/`, inverting the `database → core` direction stated above, and they do so through a package path (`core.mechanical_system.Parallel_Axis_systems`) that no longer exists. Both modules need their imports repaired before the gear load-capacity solver can consume them.

---

## Repository layout

```
axisforge/
├── config.py                                   Tolerances, solver defaults, global constants
│
├── core/                                       Domain model — geometry and data, no solving
│   ├── loads.py                                Load, RadialLoad, AxialLoad, TorqueLoad,
│   │                                           ExternalMoment, DistributedRadialLoad, LoadingProfile
│   ├── materials.py                            Material, GearMaterial
│   ├── machine_elements/
│   │   ├── shaft/shaft.py                      Shaft, ShaftSection, Shoulder, Keyway, KeywayType
│   │   ├── bearings/                           Bearing, BearingCatalog, BearingFamily, BearingType
│   │   │   └── families/                       ball_bearing/ · roller_bearing/, each radial/ + thrust/
│   │   └── gears/parallel_axis/                gear_properties/ · gear_meshing/ · planetary_gear/
│   └── mechanical_system/parallel_axis/
│       ├── spur_helical/                       ShaftSystem, GearElement,
│       │                                       SpurHelicalMeshLink, SpurHelicalGearSystem
│       └── schematic.py                        matplotlib line schematic
│
├── database/                                   Standard tabular data — no solving
│   ├── shaft/keyway/                           DIN 6885 (Parallel/) · ISO 3912 (Woodruff_key/)
│   └── gears/SpurHelicalGears/LoadCapacity_data/   csv_reader · Kv_methodB · Kv_methodC
│
├── mesh/shaft/                                 1D mesh generation and beam elements
│   ├── beam_model_settings.py                  BeamModelSettings — theory/shear/integration, once per shaft
│   ├── mesh_generation/                        mesh_1D.py (Mesh1D) · mesh_grade.py (Grader)
│   └── element_type/                           elem.py (Elem) · frame_element.py (FrameElement)
│                                               euler_bernoulli/two_noded.py (EulerBernoulliBeam)
│                                               timoshenko/two_noded.py (TimoshenkoBeam)
│                                               timoshenko/three_noded.py (QuadraticTimoshenkoBeam, not verified)
│
├── results/                                    Result containers — shapes only, no computation
│   ├── fem_results/shaft_results.py            ShaftResults (now incl. u, theta_xz/xy, phi/phi_total/phi_max), BearingNodeData
│   └── bearings/load_distribution/single_row/  ball_bearing_results.py · roller_bearing_results.py
│
├── solvers/
│   ├── machine_elements/
│   │   ├── shaft/
│   │   │   ├── fem_solvers/                    rigid_support.py (RigidSupportFEMSolver, renamed)
│   │   │   │   ├── assembly/                   build_stiffness_matrix.py (StiffnessMatrixBuilder)
│   │   │   │   │                               load_assembly/ · numerics/
│   │   │   │   ├── constraints/                boundary_conditions.py · submodel_extraction.py
│   │   │   │   └── element_theories/           timoshenko/ · euler_bernoulli/  postprocessing.py
│   │   │   ├── static_solvers/                 torsion.py (TorsionSolver, split out — see open item)
│   │   │   │                                   results_reader.py (ShaftResultsReader)
│   │   │   │                                   postprocessing.py (ShaftPostProcessor)
│   │   │   │                                   static_failure.py (reserved)
│   │   │   └── utils.py                        Marin factors, Peterson Kt, Neuber q, Kf
│   │   ├── bearings/load_distribution/
│   │   │   ├── single_row/iso_16281/           dispatch.py · numerics.py · validation.py
│   │   │   │                                   ball_bearing/ · roller_bearing/
│   │   │   └── multi_row/thrust_bearings/iso_16281/
│   │   │                                       numerics.py · validation.py
│   │   │                                       ball_bearings/ · roller_bearings/
│   │   └── gears/                              geometry.py (GearSolver) · utils.py
│   │       └── SpurHelicalGears/LoadCapacity_solver/   load_capacity.py (reserved)
│   └── mesh/convergence_solver.py              MeshConvergenceStudy, RichardsonGCI (renamed)
│
└── fixtures/                                   Analysis-script building blocks
    ├── capabilities/catalogue.py               Capability metadata, no axisforge imports
    ├── construction/                           Build objects — nothing is solved
    │   ├── construction_capabilities.py        ConstructionCapabilities, CapabilityError
    │   ├── shafts/ · bearings/ · gears/ · systems/     fixtures + per-domain report writers
    │   └── outputs/text_report.py              write_construction_report
    └── studies/                                Solve and record
        ├── study_capabilities.py               StudyCapabilities
        ├── outputs/text_report.py              write_studies_report (renamed, now 3 optional sections)
        ├── shafts/
        │   ├── fem_studies/                    fem_simple.py · results_library.py (RigidSupportFEMResultsLibrary)
        │   │                                   outputs/ (resolution_report.py · comparison_report.py · plots.py)
        │   └── convergence_studies/            convergence_library.py · convergence_study.py (see open item)
        └── bearings/load_distribution/no_lubrication/
                                                results_library.py · rolling_bearing_study.py
```

---

## Importing

Every package publishes its public surface in its own `__init__.py`. Import from the package, not from the file inside it:

```python
from axisforge.core.machine_elements.shaft              import Shaft, ShaftSection, Shoulder
from axisforge.core.machine_elements.bearings           import Bearing, BearingCatalog
from axisforge.core.machine_elements.bearings.families  import DeepGrooveBallFamily
from axisforge.core.machine_elements.gears.parallel_axis import (
    SpurHelicalGear, SpurHelicalGearMeshing,
)
from axisforge.core.mechanical_system.parallel_axis.spur_helical import (
    ShaftSystem, GearElement, SpurHelicalMeshLink, SpurHelicalGearSystem,
)
from axisforge.mesh.shaft.mesh_generation               import Mesh1D, Grader
from axisforge.results.fem_results.shaft_results        import ShaftResults, BearingNodeData
from axisforge.solvers.machine_elements.shaft.fem_solvers.rigid_bearing import (
    RigidBearingFEMSolver,
)
from axisforge.solvers.machine_elements.shaft.static.results_reader import ShaftResultsReader
from axisforge.solvers.mesh                             import MeshConvergenceStudy
```

Three properties of this surface are worth knowing:

- **Imports are lazy.** A package's `__init__.py` executes nothing heavy until a name is used, so importing a package you only partly need costs nothing.
- **Roll-ups cascade.** A parent package forwards to its child package, never to a leaf module, so a module rename never propagates past one file.
- **The surface is introspectable.** `dir(package)` and `package.__all__` list what a package offers without importing it. This is what lets `fixtures/` resolve an import set from a declaration — see [`fixtures/README.md`](axisforge/fixtures/README.md#capabilities).

Two surfaces are deliberately narrow:

- `bearings` exports only `Bearing`, `BearingCatalog`, `BearingFamily`, `BearingType`. Concrete families live in `bearings.families`, so adding a family changes nothing in `core/`.
- Ball and roller solver packages are imported explicitly and never flattened into one namespace: point and line contact produce different result types, and a shared namespace would hide which contact model a name belongs to.

---

## Analysis pipeline

The canonical solve sequence, expressed through the fixtures layer:

```python
# 1 — Declare what the script builds, and build it
construction = ConstructionCapabilities(
    shaft   = ("shafts.stepped_3section",),
    system  = ("systems.parallel_axis_linear",),
)
construction.validate_or_raise()
objs   = construction.resolve()
system = build_linear_system(...)          # SpurHelicalGearSystem, already resolved

# 2 — Solve every shaft, into the study's results library
library = solve_system(system, construction, theory="timoshenko",
                        shear_theory="cowper", integration_method="single_point")  # RigidSupportFEMResultsLibrary
result  = library.get("input_shaft")                # ShaftResults

# 3 — Internal bearing load distribution (ISO/TS 16281)
solver    = RollingBearingSolver()
load_dist = solver.solve(system, bearings, library)

# 4 — Capacity, equivalent load and stiffness, recorded per bearing label
results = solver.postprocess_and_record(
    system, bearings, library,
    catalog={"brg1a": {"capacity": {"Cr": 29_600.0},
                       "dynamic_equivalent_load": {"inner_rotating": True}}},
    load_distribution=load_dist,
)
bundle = results.get("brg1a")                       # BearingResultBundle

# 5 — Stress concentration on the shaft
ppr = ShaftPostProcessor(shaft_system, result).process()

# 6 — Reports
write_construction_report(system, "construction.txt", "Drivetrain")
write_studies_report(system, "resolution.txt", "Drivetrain", shaft_fem_library=library)
```

Three contracts govern this sequence:

- **One solver instance per shaft.** `RigidSupportFEMSolver` (renamed from `RigidBearingFEMSolver`) publishes its results as public attributes; a second `solve()` overwrites them. `solve_system()` creates a fresh solver per shaft for exactly this reason.
- **Capacity belongs to the bearing, not to the solver.** Every family owns its own ISO formulas and is called through the bearing: `bearing.family.per_element_dynamic_capacity(bearing, Cr=...)`, and for roller families `bearing.family.per_lamina_dynamic_capacity(bearing, Q_ci, Q_ce)`.
- **The library is the hand-off.** Downstream solvers read `RigidSupportFEMResultsLibrary`; they never reach back into the FEM solver's attributes.

Torsion (`T`, `tau`, `phi`) is solved separately from step 2 above, via `TorsionSolver` — `ShaftResultsReader.read()` calls it internally, so `result` already carries the torsion fields; a caller only invokes `TorsionSolver` directly when it wants torsion without a full bending/axial solve.

A shaft may carry a mixed bearing set — different contact types and different row counts — in the same call. `RollingBearingSolver` resolves each bearing to a solver from the capabilities its family declares, so adding a family requires no edit to any dispatch table.

---

## Documentation

Each top-level package has its own README with the module-by-module reference. This document is the entry point; the detail lives next to the code it describes.

| Package | Covers |
|---|---|
| [`axisforge/core/README.md`](axisforge/core/README.md) | Shaft · Bearings and families · Gears · Planetary trains · Systems · Loads · Materials · Database |
| [`axisforge/mesh/README.md`](axisforge/mesh/README.md) | Node generation, grading, beam elements |
| [`axisforge/results/README.md`](axisforge/results/README.md) | Result containers · units and invariants · how to read a result |
| [`axisforge/solvers/README.md`](axisforge/solvers/README.md) | Shaft FEM · Post-processing · Bearings (ISO/TS 16281) · Gears · Mesh convergence |
| [`axisforge/fixtures/README.md`](axisforge/fixtures/README.md) | Capability declaration · construction fixtures · studies · report writers |

---

## Design principles

- **Low coupling, high cohesion.** Solvers depend only on explicit result containers, never on each other's internals.
- **Declared surface, lazy import.** A package states what it offers and imports nothing until asked.
- **No hidden state.** Every intermediate quantity is a public attribute; a solver exposes its full working for inspection.
- **Headless solvers.** The analysis stack runs without any interface layer; presentation is always a consumer, never a dependency.
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
| 1 | Shaft FEM (Timoshenko **and now Euler-Bernoulli**, via `BeamModelSettings`) · torsion split into its own solver · mesh convergence · gear force integration | Complete for Timoshenko; Euler-Bernoulli implemented but its own quadratic (three-node) element is unverified — see [`mesh/README.md`](axisforge/mesh/README.md#status) |
| 2 | Bearing families, capability dispatch, ISO/TS 16281 load distribution | Complete for radial ball, radial roller and multi-row thrust; no multi-row roller family in `core/` to exercise the roller multi-row solver against |
| 3 | Result containers extracted into `results/` | Single-row complete; multi-row containers still local to their solvers |
| 4 | Capability declaration and fixture library — Construction and Studies stages | In progress. Mesh-convergence is now wired into the fixtures layer (`fixtures/studies/shafts/convergence_studies/`), but its `run_convergence()` entry point currently calls a pre-rename solver API and does not import — see [`fixtures/README.md`](axisforge/fixtures/README.md#status) |
| 5 | ISO 6336 gear load capacity — data layer in place, solver module reserved | In progress |
| 6 | ISO 281 rating life as a standalone solver | Planned |
| 7 | Fatigue — Goodman / Morrow / Miner | Planned |
| 8 | Lubrication — EHD film thickness, grease | Planned |
| 9 | Static failure criteria · failure susceptibility scoring · SQLite data layer | Planned |

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