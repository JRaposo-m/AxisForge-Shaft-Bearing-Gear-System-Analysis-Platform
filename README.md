# AxisForge

**Modular CAE-style platform for mechanical transmission system analysis.**

AxisForge is a deterministic, solver-centric engineering platform for the analysis of shaft–bearing–gear systems. It is built around physical first principles and traceable standards — every quantity is explainable, every result linked to an equation and a reference. Solvers are fully independent of any GUI layer, expose all intermediate quantities for inspection, and are validated against textbook and standard reference cases.

---

## Table of Contents

- [Overview](#overview)
- [Stack](#stack)
- [Repository Layout](#repository-layout)
- [Package Surface — the `__init__.py` roll-up system](#package-surface--the-__init__py-roll-up-system)
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
- Stress-concentration post-processing (shoulders, keyways) on the recovered stresses
- Internal rolling element load distribution in ball and cylindrical roller bearings (ISO/TS 16281 point and line contact), single-row and multi-row
- Bearing rating life inputs (per-element / per-lamina capacities, equivalent loads, basic reference rating life L10r)
- Spur / helical / internal gear geometry and mesh force integration
- ISO 6336 dynamic factor (Method B / Method C) data layer
- Planetary (epicyclic) train kinematics and ideal torque distribution
- Fatigue analysis, lubrication and failure susceptibility assessment *(future phases)*

---

## Stack

- **Python 3.11+**
- NumPy · SciPy · matplotlib
- PySide6 *(GUI — `ui/`, running)*
- SQLite *(data persistence — future)*
- pytest

Units are SI-consistent internally: **mm** for lengths, **N** for forces, **N·mm** for moments (torque propagates in **N·m** through the gear system and is converted at the boundary), **MPa** for stresses, **degrees** for input/output angles (**radians** internally).

---

## Repository Layout

Reorganised: package names are now lowercase and snake_case throughout, one concern per directory, and every directory carries an `__init__.py` that declares its public surface (see the next section).

```
axisforge/
├── config.py                                   # tolerances, solver defaults, global constants
├── core/
│   ├── loads.py                                # LoadPlane, Load, RadialLoad, AxialLoad, TorqueLoad,
│   │                                           #   ExternalMoment, DistributedRadialLoad, LoadingProfile
│   ├── materials.py                            # Material, GearMaterial + embedded libraries
│   ├── machine_elements/
│   │   ├── shaft/
│   │   │   └── shaft.py                        # KeywayType, Keyway, Shoulder, ShaftSection, Shaft
│   │   ├── bearings/
│   │   │   ├── bearing.py                      # Bearing — the orchestrator (assemble(), immutable)
│   │   │   ├── bearing_types.py                # BearingType enum (label only)
│   │   │   ├── catalog.py                      # BearingCatalog (frozen dataclass)
│   │   │   ├── family.py                       # BearingFamily (ABC) — pluggable family contract
│   │   │   └── families/
│   │   │       ├── ball_bearing/{radial,thrust}/{functions,subtypes}/
│   │   │       └── roller_bearing/{radial,thrust}/{functions,subtypes}/
│   │   └── gears/
│   │       └── parallel_axis/
│   │           ├── gear_properties/            # SpurHelicalGear, InternalGear
│   │           ├── gear_meshing/               # SpurHelicalGearMeshing, InternalGearMeshing
│   │           └── planetary_gear/             # PlanetaryGearTrainMeshing, PlanetaryKinematics,
│   │                                           #   PlanetaryTorques, PlanetaryMember, MeshTag
│   └── mechanical_system/
│       └── parallel_axis/
│           ├── spur_helical/
│           │   ├── shaft_system.py             # GearElement, ShaftSystem
│           │   └── gear_system.py              # SpurHelicalMeshLink, SpurHelicalGearSystem
│           └── schematic.py                    # draw_gear_system, draw_shaft_detail, ...
├── database/                                   # catalogue / standard tabular data (no solving)
│   ├── shaft/keyway/Parallel/parallel_keyway.py        # DIN 6885 lookup
│   ├── shaft/keyway/Woodruff_key/iso3912.py            # ISO 3912 lookup
│   └── gears/SpurHelicalGears/LoadCapacity_data/       # DynamicFactor (Kv method B),
│                                                       #   DynamicFactorC (Kv method C), lookup_KA
├── mesh/
│   └── shaft/
│       ├── mesh_generation/                    # Mesh1D, Grader
│       └── element_type/                       # Elem, TimoshenkoBeam
├── solvers/
│   ├── machine_elements/
│   │   ├── shaft/
│   │   │   ├── oneD_analysis/
│   │   │   │   ├── build_stiffness_matrix.py   # StiffnessMatrixBuilder
│   │   │   │   ├── FEM_solvers/                # SimpleFEMSolver, SubmodelSolver
│   │   │   │   └── static/                     # static_analysis.py, shaft_post_processor.py
│   │   │   └── utils.py                        # Marin factors, Kt/Kf helpers
│   │   ├── bearings/
│   │   │   ├── ISO_16281/                      # dispatch.py, library.py,
│   │   │   │                                   #   rolling_bearing_solver.py,
│   │   │   │                                   #   Ball_Bearing/, Roller_Bearing/
│   │   │   └── life.py                         # BearingLifeSolver (ISO 281 L10)
│   │   └── gears/                              # geometry.py (GearSolver), utils.py,
│   │                                           #   SpurHelicalGears/LoadCapacity_solver/
│   ├── mesh/
│   │   └── mesh_convergence_study.py           # RichardsonGCI, MeshConvergenceStudy
│   └── lubrification/                          # placeholder, Phase 4+
├── models/                                     # GearGeometryResult, GearForceResult, ...
├── ui/                                         # PySide6 app: main_window, script_editor,
│                                               #   runner, project_explorer, workspace_panel,
│                                               #   output_console, project_wizard, exporters/
├── fixtures/                                   # modular analysis templates + capabilities/
└── tests/                                      # unit / regression / validation tests
```


## Package Surface — the `__init__.py` roll-up system

Every package now carries an `__init__.py` that declares **what that directory offers**, and nothing else. The pattern is uniform:

```python
__all__ = ["DeepGrooveBallFamily", "AngularContactFamily", "SelfAligningBallFamily"]

_LAZY = {
    "DeepGrooveBallFamily": ".subtypes",
    "AngularContactFamily": ".subtypes",
    "SelfAligningBallFamily": ".subtypes",
}

def __getattr__(name):          # PEP 562 — resolved on first access, then cached
    ...
def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY.keys()))

if TYPE_CHECKING:               # static analysers / IDEs see the real symbols
    from .subtypes import DeepGrooveBallFamily, AngularContactFamily, SelfAligningBallFamily
```

Three properties follow, and they are the whole point:

1. **Lazy.** Importing `axisforge.core.machine_elements.bearings.families` costs nothing — no NumPy-heavy subtype module is executed until a name is actually touched. A script that only uses a DGBB never imports the roller side.
2. **Cascading.** A roll-up never points at a leaf `.py`; it always forwards to the next `__init__.py` down (`families/` → `ball_bearing/` → `radial/` → `subtypes/` → `deep_groove.py`). Renaming a leaf module touches exactly one `__init__.py`.
3. **Introspectable.** `__all__` and `_LAZY` are plain data. `dir(package)` lists everything the package offers *without importing it*. This is what `fixtures/capabilities/` reads to build its selection menu — see below.

### Direction of dependency

```
fixtures/capabilities/   →  reads __all__ / _LAZY of the package roll-ups
        ↓                     (never the reverse: no __init__.py imports capabilities)
package __init__.py      →  forwards to the child __init__.py
        ↓
leaf module (.py)        →  the actual class
```

`families/__init__.py` states it explicitly: *"É a partir daqui que `bearing.py` é capaz de listar/selecionar famílias, e mais tarde de onde um `capabilities.py` para bearings vai ler — nunca ao contrário."*

### Deliberate exclusions

- `bearings/__init__.py` exports only `Bearing`, `BearingCatalog`, `BearingFamily`, `BearingType`. The concrete families are **not** re-exported there — otherwise every new family would force an edit of that file, which is exactly what the "adding a family changes nothing in `core/`" principle forbids. Import them from `bearings.families`.
- `families/**/functions/` is **not** rolled up into its parent. Those are the Hertz/capacity primitives, internal to the subtypes; the public surface at that level is the family classes.
- `ISO_16281/__init__.py` exports only the orchestration layer (`RollingBearingSolver`, `BearingResultsLibrary`, `resolve_solver_cls*`, `SolverDispatchError`). `Ball_Bearing/` and `Roller_Bearing/` are **not** flattened into it — point and line contact have different result types, and mixing them in one namespace would hide which contact model a name belongs to.
- `register_contact_solver` is not re-exported: it is the decorator `single_row_solver.py` applies to itself at import time; nothing outside the package registers a contact solver.


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

# 4. Solve internal bearing load distribution (ISO/TS 16281).
#    Dispatch is by CAPABILITY + required attributes (ISO_16281/dispatch.py),
#    not by BearingType, so a shaft's bearing set may freely mix contact
#    types (a locating DGBB plus a floating cylindrical roller bearing) and
#    row counts (a multi-row thrust ball bearing alongside single-row ones).
solver   = RollingBearingSolver()
load_dist = solver.solve(shaft_system, bearings, library)   # {label: [row results]}

# 5. Capacity, equivalent load and stiffness in one pass, recorded per label
results = solver.postprocess_and_record(
    shaft_system, bearings, library,
    catalog={"brg1a": {"capacity": {"Cr": 29_600.0},
                       "dynamic_equivalent_load": {"inner_rotating": True}}},
    load_distribution=load_dist,
)
bundle = results.get("brg1a")     # .load_distribution / .capacity /
                                  # .dynamic_equivalent_load / .stiffness / .extra

# 6. Stress concentration post-processing on the shaft
ppr = ShaftPostProcessor(shaft_system, library.get(shaft_system.name)).process()
```

Capacity is **never** computed inside `solvers/`. Every `BearingFamily` owns its own ISO 281 / ISO/TS 16281 formulas and is called uniformly:

```python
Q_ci, Q_ce = bearing.family.per_element_dynamic_capacity(bearing, Cr=...)   # or Ca=... for thrust
q_ci, q_ce = bearing.family.per_lamina_dynamic_capacity(bearing, Q_ci, Q_ce) # roller only
```

See [`solvers/README.md`](axisforge/solvers/README.md) for the full solver-by-solver reference and [`core/README.md`](axisforge/core/README.md) for the objects built in steps 1–2.

---

## Documentation

Each top-level package has its own README with the full module-by-module reference (classes, methods, usage snippets). This root document stays the entry point — pitch, architecture, pipeline, principles, roadmap; the detail lives next to the code it describes so it stays in sync as each package evolves independently.

| Package | Covers |
|---|---|
| [`axisforge/core/README.md`](axisforge/core/README.md) | Shaft · Bearings (families) · Gears · Systems · Gear/Planetary meshing · Loads · Materials |
| [`axisforge/solvers/README.md`](axisforge/solvers/README.md) | Shaft FEM · post-processing · Bearings (ISO/TS 16281 + life) · Gears · Mesh convergence |
| [`axisforge/mesh/README.md`](axisforge/mesh/README.md) | 1D shaft mesh generation, grading, beam elements |
| [`axisforge/models/README.md`](axisforge/models/README.md) | Result containers (gear geometry/force, stress) |
| [`axisforge/ui/README.md`](axisforge/ui/README.md) | PySide6 application: script editor, section runner, exporters |
| [`axisforge/fixtures/README.md`](axisforge/fixtures/README.md) | Modular analysis-script templates · `capabilities/` selector |

`axisforge/database/` has no README of its own yet — it holds only tabular lookups (DIN 6885 / ISO 3912 keyways, ISO 6336-1 K_A and K_v data) and is documented from `core/README.md`, which is where its only consumer (`Keyway.from_standard()`) lives.

---

## Design Principles

- **Low coupling, high cohesion** — solvers depend only on explicit result containers, never on each other's internals.
- **Declared surface, lazy import** — a package's `__init__.py` states what it offers and imports nothing until asked. This is what makes capability selection possible without a registry.
- **No hidden state** — every intermediate quantity is a public attribute; solvers expose their full working for inspection.
- **GUI-independent solvers** — the analysis core runs headless; the UI is a thin consumer.
- **Deterministic and explainable** — no black-box methods. Failure assessment (future) uses measurable physical drivers and traceable indices, not machine learning or probabilistic life prediction.
- **Composition over inheritance** — the planetary train composes two pair-meshing objects rather than subclassing them; bearing geometry is a component of the family, not an identity.
- **Capacity belongs to the element, not to the solver** — `bearing.family.per_element_dynamic_capacity()` is the single source; a solver never carries its own copy of an ISO formula.
- **Fail fast on geometry** — validation happens at construction; invalid geometry is never silently accepted.
- **One file per concern** — each subtype, function group, solver and result shape lives in its own module; imports are explicit and traceable.
- **Flag, don't silently fix** — a suspected discrepancy against a standard is documented in place (see the `1.038` coefficient note in `core/README.md`), never quietly corrected.

---

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Shaft FEM · ISO/TS 16281 load distribution · gear force integration | **Done / maintained** |
| 2 | Bearing families (ball radial/thrust, roller radial/thrust, single + multi-row) · capability dispatch · L10r | **Active** — thrust roller load-distribution solver still missing |
| 2b | ISO 6336 gear strength — K_v Method B/C data layer in place, `LoadCapacity_solver/` empty | In progress |
| 3 | `fixtures/capabilities/` selector · fixture migration to `Bearing.assemble()` | Next |
| 4 | Fatigue analysis — Goodman / Morrow / Miner | Planned |
| 5 | Lubrication assessment — EHD film, grease (`solvers/lubrification/`) | Planned |
| 6 | Failure susceptibility scoring · SQLite data layer | Planned |
| 7 | PySide6 GUI hardening (`ui/` running, not frozen) | Ongoing |

---

## References

- ISO/TS 16281:2008 — *Rolling bearings: Methods for calculating the modified reference rating life for universally loaded bearings*
- ISO 281:2007 — *Rolling bearings: Dynamic load ratings and rating life*
- ISO 1281-1:2021 — *Rolling bearings: Explanatory notes on ISO 281* (thrust ball capacity, Sec 6.3/6.4)
- ISO 76:2006 — *Rolling bearings: Static load ratings*
- ISO 21771:2007 — *Gears: Cylindrical involute gears and gear pairs*
- ISO 53:2013 — *Cylindrical gears for general engineering: standard basic rack tooth profile*
- ISO 6336-1/-2/-3 — *Calculation of load capacity of spur and helical gears*
- ISO 6336-5 — *Strength and quality of materials*
- DIN 6885 — *Parallel keys and keyways*
- ISO 3912 — *Woodruff keys and keyways*
- Harris & Kotzalas, *Rolling Bearing Analysis*, 5th ed., Wiley
- Palmgren, *Grundlagen der Wälzlagertechnik*, 3rd ed., Franckh
- Shigley, *Mechanical Engineering Design*, 10th ed.
- Peterson, *Stress Concentration Factors*
- MAAG Gear Book, 2nd ed.
- Henriot, *Traité théorique et pratique des engrenages*
- Arnaudov & Karaivanov, *Planetary Gear Trains*