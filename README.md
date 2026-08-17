# AxisForge

**Modular CAE-style platform for mechanical transmission system analysis.**

AxisForge is a deterministic, solver-centric engineering platform for the analysis of shaft–bearing–gear systems. It is built around physical first principles and traceable standards — every quantity is explainable, every result linked to an equation and a reference. Solvers are fully independent of any GUI layer, expose all intermediate quantities for inspection, and are validated against textbook and standard reference cases.

---

## Table of Contents

- [Overview](#overview)
- [Stack](#stack)
- [Repository Layout](#repository-layout)
- [Analysis Pipeline](#analysis-pipeline)
- [Documentation](#documentation)
- [Design Principles](#design-principles)
- [Roadmap](#roadmap)
- [References](#references)

---

## Overview

The platform targets the complete analysis pipeline of a multi-shaft parallel-axis transmission:

- Static load distribution across multi-shaft gear trains
- 1D FEM shaft deflection and internal force recovery (two-plane)
- Internal rolling element load distribution in ball and cylindrical roller bearings (ISO/TS 16281 point and line contact)
- Bearing rating life inputs (per-element and per-lamina capacities and equivalent loads)
- Spur / helical / internal gear geometry and mesh force integration
- Planetary (epicyclic) train kinematics and ideal torque distribution
- Fatigue analysis and failure susceptibility assessment *(future phases)*

---

## Stack

- **Python 3.11+**
- NumPy · SciPy · matplotlib
- PySide6 *(GUI — future)*
- SQLite *(data persistence — future)*
- pytest

Units are SI-consistent internally: **mm** for lengths, **N** for forces, **N·mm** for moments (torque propagates in **N·m** through the gear system and is converted at the boundary), **MPa** for stresses, **degrees** for input/output angles (**radians** internally).

---

## Repository Layout

```
axisforge/
├── core/
│   ├── loads.py
│   ├── materials.py
│   └── machine_elements/
│       ├── Shaft/                          # Shoulder, ShaftSection, Shaft
│       ├── Bearings/                       # Bearing, BearingType, make_bearing(),
│       │                                   #   geometry/, subtypes/
│       └── Gears/Parallel_Axis_gears/      # SpurHelicalGear, InternalGear
│   └── mechanical_system/Parallel_Axis_systems/
│       ├── systems/spur_helicoidal_system/ # GearElement, ShaftSystem,
│       │                                   #   SpurHelicalMeshLink, SpurHelicalGearSystem
│       ├── gear_meshing/                   # SpurHelicalGearMeshing, InternalGearMeshing,
│       │                                   #   PlanetaryGearMeshing
│       └── schematic.py                    # assembly visualisation
├── mesh/oneD/shaft/                        # Mesh1D, Grader, Elem, TimoshenkoBeam,
│                                            #   RichardsonGCI, MeshConvergenceStudy
├── solvers/machine_elements/
│   ├── shaft/oneD_analysis/                # StiffnessMatrixBuilder, SimpleFEMSolver,
│   │                                       #   SubmodelSolver, static_analysis
│   ├── bearings/ISO_16281/                 # RollingBearingSolver, ISO16281BallSolver,
│   │                                       #   ISO16281RollerSolver, capacity/postprocessing
│   └── gears/                              # GearSolver, geometry helpers
├── models/                                 # GearGeometryResult, GearForceResult,
│                                            #   CriticalSection, StressResult
├── ui/                                     # PySide6 interactive section runner
├── fixtures/                               # Modular analysis templates (in construction)
│                                            #   see fixtures/README.md
└── tests/                                  # unit / regression / validation tests
```

> **Note on legacy modules:** an earlier flat layout (`core/shaft.py`, `core/components.py`, `core/system.py`) coexists with the current `core/machine_elements/` package. New development targets the `machine_elements` structure; the flat modules are retained for backward compatibility during migration.

Full class-by-class detail for each package lives in that package's own README — see [Documentation](#documentation) below.

---

## Analysis Pipeline

The canonical solve sequence for one shaft:

```python
# 1. Assemble the multi-shaft gear system and resolve power flow
gearbox = SpurHelicalGearSystem(shafts, links, label="drivetrain")
gearbox.resolve(P_W, rpm_in, rotation_dir_source=1)

# 2. Mesh the shaft (mandatory nodes + optional grading at gears)
fem = SimpleFEMSolver()
fem.solve(shaft_system, extra_mandatory=gear_grade_nodes)

# 3. Post-process FEM into a results library
ShaftResultsReader(fem, shaft_system).read(library)

# 4. Solve internal bearing load distribution (ISO/TS 16281) — dispatches
#    per BearingType, so a shaft's bearing set may mix contact types
#    (e.g. a locating DGBB plus a floating cylindrical roller bearing)
solver = RollingBearingSolver()
load_dist = solver.solve(shaft_system, bearings, library)

# 5. Compute per-element capacities and equivalent loads (contact-type-specific)
cap   = RollingElementCapacity.radial(bearing, Cr=C_rating)                       # ball
derel = DynamicEquivalentRollingElementLoad.from_distribution(bearing, result)    # ball

cap_r  = RollerElementCapacity.radial(roller_bearing, Cr=C_rating)                # roller
derel_r = LaminaDynamicEquivalentLoad.from_distribution(roller_bearing, result)    # roller
```

See [`solvers/README.md`](axisforge/solvers/README.md) for the full solver-by-solver reference and [`core/README.md`](axisforge/core/README.md) for the objects built in steps 1–2.

---

## Documentation

Each top-level package has its own README with the full module-by-module reference (classes, methods, usage snippets). This root document stays the entry point — pitch, architecture, pipeline, principles, roadmap; the detail lives next to the code it describes so it stays in sync as each package evolves independently.

| Package | Covers |
|---|---|
| [`axisforge/core/README.md`](axisforge/core/README.md) | Shaft, Bearings, Gears (`machine_elements`) · Systems, Gear Meshing (`mechanical_system`) · Loads · Materials |
| [`axisforge/solvers/README.md`](axisforge/solvers/README.md) | Shaft FEM · Bearings (ISO/TS 16281) · Gears |
| [`axisforge/mesh/README.md`](axisforge/mesh/README.md) | 1D Shaft Mesh generation, elements, convergence study |
| [`axisforge/models/README.md`](axisforge/models/README.md) | Result containers (gear geometry/force, stress) |
| [`axisforge/ui/README.md`](axisforge/ui/README.md) | PySide6 interactive section runner |
| [`axisforge/fixtures/README.md`](axisforge/fixtures/README.md) | Modular analysis-script templates (in construction) |

---

## Design Principles

- **Low coupling, high cohesion** — solvers depend only on explicit result containers, never on each other's internals.
- **No hidden state** — every intermediate quantity is a public attribute; solvers expose their full working for inspection.
- **GUI-independent solvers** — the analysis core runs headless; the UI is a thin consumer.
- **Deterministic and explainable** — no black-box methods. Failure assessment (future) uses measurable physical drivers and traceable indices, not machine learning or probabilistic life prediction.
- **Composition over inheritance** — e.g. the planetary train composes two pair-meshing objects rather than subclassing them; bearing geometry is a component of the bearing subtype, not its identity.
- **Fail fast on geometry** — validation happens at construction; invalid geometry is never silently accepted.
- **One file per concern** — each bearing subtype, geometry class, and solver lives in its own module; imports are explicit and traceable.

---

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Shaft FEM · ISO/TS 16281 bearing load distribution · gear force integration | **Active** |
| 2 | Angular contact / tapered / spherical roller bearing subtypes · ISO 6336 gear strength | In progress — cylindrical roller bearing (ISO/TS 16281 line contact) implemented |
| 3 | Fatigue analysis — Goodman / Morrow / Miner | Planned |
| 4 | Lubrication assessment — EHD film, grease | Planned |
| 5 | Failure susceptibility scoring · SQLite data layer | Planned |
| 6 | PySide6 GUI | Planned |

---

## References

- ISO/TS 16281:2008 — *Rolling bearings: Methods for calculating the modified reference rating life for universally loaded bearings*
- ISO 281:2007 — *Rolling bearings: Dynamic load ratings and rating life*
- ISO 21771:2007 — *Gears: Cylindrical involute gears and gear pairs*
- ISO 6336 — *Calculation of load capacity of spur and helical gears*
- ISO 6336-5 — *Strength and quality of materials*
- Harris & Kotzalas, *Rolling Bearing Analysis*, 5th ed., Wiley
- Palmgren, *Grundlagen der Wälzlagertechnik*, 3rd ed., Franckh
- Shigley, *Mechanical Engineering Design*, 10th ed.
- MAAG Gear Book, 2nd ed.
- Arnaudov & Karaivanov, *Planetary Gear Trains*
