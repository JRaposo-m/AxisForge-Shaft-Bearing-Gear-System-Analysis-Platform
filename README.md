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
| Shaft torsion (torque, shear stress, twist angle) | Pure statics, run independently of the bending/axial solve | `solvers/.../shaft/fem_solvers/global_solver/torsion.py` |
| Submodel refinement with prescribed cut-node state | Lagrange multipliers | `solvers/.../shaft/fem_solvers/submodel_solver/` |
| Mesh convergence assessment | Richardson extrapolation + Grid Convergence Index | `solvers/mesh/convergence_solver.py` |
| Stress concentration at shoulders and keyways | Peterson / Shigley / Neuber | `solvers/.../shaft/static_solvers/postprocessing.py` |
| Internal rolling element load distribution, single- and multi-row | ISO/TS 16281 §4 (point contact), §5 (line contact) | `solvers/.../bearings/load_distribution/iso_16281/` |
| Per-element and per-lamina capacity, equivalent load, L10r | ISO 281, ISO 1281-1, ISO/TS 16281 | `core/.../bearings/families/`, the bearing postprocessing modules |
| Spur, helical and internal gear geometry and mesh forces | ISO 21771, ISO 53, KHK, MAAG | `core/.../gears/`, `solvers/.../gears/` |
| Planetary train kinematics and ideal torque split | Willis equation, Arnaudov & Karaivanov | `core/.../gears/parallel_axis/planetary_gear/` |
| Application and dynamic factor data layer | ISO 6336-1 Method B and Method C | `database/gears/` |
| Keyway geometry lookup | DIN 6885, ISO 3912 | `database/shaft/keyway/` |

Construction and resolution text reports, and the analysis-script fixtures that drive them, are no longer part of this repository — see the note under [Architecture](#architecture).

> **Removed, not yet replaced.** A matplotlib line-schematic module (`core/mechanical_system/parallel_axis/schematic.py`) previously appeared here as "Assembly visualisation". It no longer exists on disk — see [`core/README.md`](axisforge/core/README.md#schematic). If assembly visualisation is still wanted, it needs to be rebuilt; nothing in the current tree replaces it.

Gear load capacity, fatigue, lubrication and failure susceptibility are future phases — see [Roadmap](#roadmap).

---

## Status

| Package | Contents | Status |
|---|---|---|
| `config` | Tolerances, solver defaults, global constants | Implemented |
| `core/` | Shaft, bearings and families, gears, meshing, planetary trains, systems, loads, materials | Implemented — bearings restructured, see [`core/README.md`](axisforge/core/README.md#bearings) |
| `database/` | DIN 6885, ISO 3912, ISO 6336-1 K_A and K_v | Implemented — see the note under [Architecture](#architecture) |
| `mesh/` | 1D node generation, grading, Timoshenko **and Euler-Bernoulli** beam elements via a registered-formulation hierarchy, `BeamModelSettings` | Implemented — Euler-Bernoulli is **no longer reserved**; `element_type/` restructured into a `BeamFormulation` registry — see [`mesh/README.md`](axisforge/mesh/README.md#status) |
| `results/` | Result containers: shaft FEM (incl. axial displacement, bending rotation and twist angle), bearing load distribution (single- and multi-row, one shared shape) and reference life, per-bearing aggregate | Implemented — bearing containers unified, see [`results/README.md`](axisforge/results/README.md#status) |
| `solvers/.../shaft/` | Stiffness assembly, rigid-*support* FEM (`RigidSupportFEMSolver`), separate torsion solve, Lagrange-multiplier submodel solve, functional (not class-based) result assembly, stress concentration | Implemented; `static_failure` reserved, and `static_solvers/__init__.py` is currently broken — see the open items in [`solvers/README.md`](axisforge/solvers/README.md#status) |
| `solvers/.../bearings/` | ISO/TS 16281, single- and multi-row, one package (`iso_16281/`), dispatch, postprocessing | Implemented |
| `solvers/.../gears/` | Geometry and mesh forces | Implemented; ISO 6336 load capacity reserved |
| `solvers/mesh/` | Mesh convergence, Richardson GCI (`convergence_solver.py`) | Implemented |

`fixtures/` is **not part of this repository any more** — it moved out into a separate repository (`AxisForge-Design-Studies`, under `axisforge_bridge`), which consumes `axisforge` as an installed package rather than living inside it. It no longer has a row here for the same reason `axisforge` itself doesn't list its own downstream consumers. See the note under [Architecture](#architecture).

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

Four layers within this repository, plus one external consumer, with a strictly one-way dependency:

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
              ▬ ▬ ▬ ▬ ▬ ▬ ▬│▬ ▬ ▬ ▬ ▬ ▬ ▬ ▬    package boundary — axisforge ends here
                          ▼
     axisforge_bridge / fixtures            (separate repository: AxisForge-Design-Studies)
                                             studies, libraries, reports — installs and imports
                                             axisforge, never the reverse
```

Rules, in force:

- `core/` depends on `config`, `database/`, NumPy and the standard library. It never imports a solver.
- `results/` imports NumPy and the standard library only at runtime — nothing from `axisforge` at all (`TYPE_CHECKING`-only imports from `solvers/`, erased at runtime, are the sole exception, used for annotations).
- `solvers/` depends on `core/`, `mesh/`, `config` and `results/`. **No solver imports another solver's internals** — only the shared result containers.
- **`fixtures/` is not part of `axisforge` any more.** It moved out into its own repository, `AxisForge-Design-Studies` (package `axisforge_bridge`), which installs `axisforge` (this repository) as a normal dependency and imports it — the dependency is one-way, outward, and this repository has no knowledge of that one. Any reference to `fixtures/` elsewhere in this document that still shows it as a subpackage of `axisforge/` is describing the repository's state **before** that split, and should be read as historical unless it says otherwise.

`results/` is the narrow waist within this repository: a solver depends on the container it writes, a consumer on the container it reads, and neither depends on the other.

> **Open item.** `database/gears/.../Kv_methodB` and `Kv_methodC` import from `core/`, inverting the `database → core` direction stated above, and they do so through a package path (`core.mechanical_system.Parallel_Axis_systems`) that no longer exists. Both modules need their imports repaired before the gear load-capacity solver can consume them. (Not re-verified against the current `core/` tree in this pass — the path they should point to instead is `core.mechanical_system.parallel_axis.spur_helical`.)

---

## Repository layout

```
axisforge/
├── __init__.py                                 Top-level rollup — eager cascade config → core → mesh →
│                                               results → solvers, with a name-collision check across
│                                               subpackages and a best-effort __version__
├── config.py                                   Tolerances, solver defaults, global constants
│
├── core/                                       Domain model — geometry and data, no solving
│   ├── loads.py                                Load, RadialLoad, AxialLoad, TorqueLoad,
│   │                                           ExternalMoment, DistributedRadialLoad, LoadingProfile
│   ├── materials.py                            Material, GearMaterial
│   ├── machine_elements/
│   │   ├── shaft/shaft.py                      Shaft, ShaftSection, Shoulder, Keyway, KeywayType
│   │   ├── bearings/                           base.py (BearingType, BearingCatalog, BearingFamily)
│   │   │                                       bearing.py (Bearing)
│   │   │   └── families/                       family.py (every concrete family, @register_family)
│   │   │                                       capacity.py · iso16281_contact.py (shared math)
│   │   └── gears/parallel_axis/                gear_properties/ · gear_meshing/ · planetary_gear/
│   └── mechanical_system/parallel_axis/
│       └── spur_helical/                       ShaftSystem, GearElement,
│                                               SpurHelicalMeshLink, SpurHelicalGearSystem
│                                               (schematic.py removed — see the note under "What it does")
│
├── database/                                   Standard tabular data — no solving
│   ├── shaft/keyway/                           DIN 6885 (Parallel/) · ISO 3912 (Woodruff_key/)
│   └── gears/SpurHelicalGears/LoadCapacity_data/   csv_reader · Kv_methodB · Kv_methodC
│
├── mesh/shaft/                                 1D mesh generation and beam elements
│   ├── beam_model_settings.py                  BeamModelSettings — theory/shear/integration, once per shaft
│   ├── mesh_generation/                        mesh_1D.py (Mesh1D) · mesh_grade.py (Grader)
│   └── element_type/                           elem.py — BeamFormulation, ShearDeformableBeamFormulation,
│                                               EulerBernoulliBeam, TimoshenkoBeam, FrameElement,
│                                               ElemBase, Elem, QuadraticTimoshenkoElem (placeholder,
│                                               raises NotImplementedError), _FORMULATION_REGISTRY
│                                               shear_factor.py — ShearFactor (Cowper / Hutchinson)
│
├── results/                                    Result containers — shapes only, no computation
│   ├── _base.py                                Shared conventions (frozen/kw_only dataclass helpers)
│   ├── fem_results/                            shaft_results.py (ShaftResults, BearingNodeData)
│   │                                           submodel_results.py (SubmodelResult)
│   ├── convergence_results/                    convergence_results.py (ConvergenceRecord, MeshRefinementResult)
│   └── bearings/
│       ├── load_distribution/                  load_distribution_results.py — LoadDistributionResult /
│       │                                       BallLoadDistributionResult / LineContactLoadDistributionResult /
│       │                                       RollerLoadDistributionResult, BearingResult /
│       │                                       BallBearingResult / RollerBearingResult (n_rows >= 1, no
│       │                                       separate single-/multi-row class)
│       ├── life/                               basic_life_results.py (BasicReferenceRatingLifeResult)
│       └── bearing_analysis_result.py          BearingAnalysisResult — the per-bearing aggregate
│
├── solvers/
│   ├── machine_elements/
│   │   ├── shaft/
│   │   │   ├── fem_solvers/
│   │   │   │   ├── global_solver/              rigid_support.py (RigidSupportFEMSolver)
│   │   │   │   │                               torsion.py (TorsionSolver)
│   │   │   │   │                               global_postprocessing.py (build_shaft_result — replaces
│   │   │   │   │                               the previous ShaftResultsReader class)
│   │   │   │   ├── submodel_solver/            lagrange_multipliers.py (SubmodelSolver, SubmodelSolution)
│   │   │   │   │                               submodel_postprocessing.py (build_submodel_result)
│   │   │   │   ├── assembly/                   build_stiffness_matrix.py (StiffnessMatrixBuilder)
│   │   │   │   │                               load_assembly/ · numerics/
│   │   │   │   ├── constraints/                boundary_conditions.py · submodel_extraction.py
│   │   │   │   └── element_theories/           element_postprocessing.py (ElementTheoryPostProcessor,
│   │   │   │                                   Euler/TimoshenkoPostProcessing — one module, not a
│   │   │   │                                   timoshenko/euler_bernoulli subfolder split)
│   │   │   ├── static_solvers/                 postprocessing.py (ShaftPostProcessor)
│   │   │   │                                   static_failure.py (reserved)
│   │   │   │                                   __init__.py is currently broken — see solvers/README.md
│   │   │   └── utils.py                        Marin factors, Peterson Kt, Neuber q, Kf
│   │   ├── bearings/load_distribution/iso_16281/     dispatch.py · numerics.py · validation.py
│   │   │                                       contact_solver.py (single- and multi-row, ball and roller)
│   │   │                                       contact_postprocessing.py (stiffness, basic reference life)
│   │   └── gears/                              geometry.py (GearSolver) · utils.py
│   │       └── SpurHelicalGears/LoadCapacity_solver/   load_capacity.py (reserved)
│   └── mesh/convergence_solver.py              MeshConvergenceStudy, RichardsonGCI
│
└── (nothing below solvers/ — this repository ends here; see the note under Architecture)
```

`fixtures/` — construction fixtures, studies, result libraries, report writers — used to be the sixth top-level subpackage here. It has moved to a separate repository (`AxisForge-Design-Studies`, package `axisforge_bridge`), which now consumes `axisforge` as an installed dependency. Its own tree, and whatever README documents it, live there — not in this repository, and this document was not able to review it in this pass.

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
from axisforge.solvers.machine_elements.shaft.fem_solvers.global_solver.rigid_support import (
    RigidSupportFEMSolver,
)
from axisforge.solvers.machine_elements.shaft.fem_solvers.global_solver.global_postprocessing import (
    build_shaft_result,
)
from axisforge.solvers.mesh                             import MeshConvergenceStudy

# Every name above is also reachable straight off the top-level package,
# e.g. `axisforge.Bearing`, `axisforge.Mesh1D`, `axisforge.RigidSupportFEMSolver` —
# the rollups exist precisely so a caller does not have to know the full path.
```

Three properties of this surface are worth knowing — **one of them corrected from a previous pass of this document**:

- **Imports are eager, not lazy.** Every package's `__init__.py` (`core`, `mesh`, `results`, `solvers`, and the top-level `axisforge` package itself) imports its full public surface immediately, by explicit name, deliberately without lazy loading or `__getattr__` tricks — this is a consistent, stated design choice across the whole codebase now, the opposite of what a previous pass of this document claimed. The one exception is `solvers/machine_elements/shaft/static_solvers/__init__.py`, which still uses a lazy `__getattr__` pointing at a module that no longer exists — that is a known bug, not the intended pattern; see the open item in [`solvers/README.md`](axisforge/solvers/README.md#status).
- **Roll-ups cascade.** A parent package forwards to its child package, never to a leaf module, so a module rename never propagates past one file. Names are written out explicitly at each level (not `import *` re-exported blindly) so the public surface stays grep-able.
- **The surface is introspectable.** `dir(package)` and `package.__all__` list what a package offers without importing it. This is what lets `axisforge_bridge` (`AxisForge-Design-Studies`, the separate repository that consumes this one — see [Architecture](#architecture)) resolve an import set from a declaration, the way `fixtures/`'s capability selector used to when it still lived inside this repository.

One surface that **used to be** deliberately narrow no longer is:

- Concrete bearing families **are** re-exported eagerly now, from `axisforge.core.machine_elements.bearings.families` up through every rollup level to `axisforge` itself, generated automatically from the `@register_family` registry rather than written by hand. A previous pass of this document said the opposite ("families are not re-exported on purpose") — that was true of the old, one-file-per-subtype structure and no longer describes the current one. See [`core/README.md`](axisforge/core/README.md#import-surface).
- Ball and roller solver packages are still imported explicitly and never flattened into one namespace: point and line contact produce different result types, and a shared namespace would hide which contact model a name belongs to.

---

## Analysis pipeline

> **`fixtures/` has moved.** This pipeline used to run through `fixtures/` when that package lived inside this repository — it now lives in the separate `AxisForge-Design-Studies` repository (`axisforge_bridge`), which this review did not have access to. The example below is kept as a record of the pipeline's *shape* (construction → shaft solve → bearing load distribution → capacity/life → stress concentration → reports), not as a verified, current call sequence: several names in it are already known to be stale against the `axisforge` API documented above — in particular, `ShaftResultsReader` no longer exists as a class (see [`solvers/README.md`](axisforge/solvers/README.md#shaft-results-and-post-processing)). The authoritative version of this pipeline now belongs in `axisforge_bridge`'s own documentation, not here.

The canonical solve sequence, as it looked while `fixtures/` was still part of this repository:

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

- **One solver instance per shaft.** `RigidSupportFEMSolver` publishes its results as public attributes; a second `solve()` overwrites them. `solve_system()` creates a fresh solver per shaft for exactly this reason.
- **Capacity belongs to the bearing, not to the solver.** Every family owns its own ISO formulas and is called through the bearing: `bearing.family.per_element_dynamic_capacity(bearing, Cr=...)`, and for roller families `bearing.family.per_lamina_dynamic_capacity(bearing, Q_ci, Q_ce)`.
- **The library is the hand-off.** Downstream solvers read `RigidSupportFEMResultsLibrary`; they never reach back into the FEM solver's attributes.

Torsion (`T`, `tau`, `phi`) is solved separately from step 2 above, via `TorsionSolver` — the function that assembles a full `ShaftResults` (`build_shaft_result()`, `solvers/.../fem_solvers/global_solver/global_postprocessing.py` — see [`solvers/README.md`](axisforge/solvers/README.md#shaft-results-and-post-processing)) calls it internally, so `result` already carries the torsion fields; a caller only invokes `TorsionSolver` directly when it wants torsion without a full bending/axial solve.

A shaft may carry a mixed bearing set — different contact types and different row counts — in the same call. The bearing solver resolves each bearing to a solver from the capabilities its family declares, so adding a family requires no edit to any dispatch table. `solve()` returns one `BearingAnalysisResult` per bearing label (see [`results/README.md`](axisforge/results/README.md#bearingsbearing_analysis_resultpy)), not the bare load-distribution result this example's step 4 implies — whatever wrapper `axisforge_bridge` uses today (`BearingResultBundle` above, or its replacement) needs checking against that current shape there, not here.

---

## Documentation

Each top-level package has its own README with the module-by-module reference. This document is the entry point; the detail lives next to the code it describes.

| Package | Covers |
|---|---|
| [`axisforge/core/README.md`](axisforge/core/README.md) | Shaft · Bearings and families · Gears · Planetary trains · Systems · Loads · Materials · Database |
| [`axisforge/mesh/README.md`](axisforge/mesh/README.md) | Node generation, grading, beam elements |
| [`axisforge/results/README.md`](axisforge/results/README.md) | Result containers · units and invariants · how to read a result |
| [`axisforge/solvers/README.md`](axisforge/solvers/README.md) | Shaft FEM · Post-processing · Bearings (ISO/TS 16281) · Gears · Mesh convergence |
| `axisforge/fixtures/README.md` | **No longer applicable.** `fixtures/` is not part of this repository — capability declaration, construction fixtures, studies and report writers now live in the separate `AxisForge-Design-Studies` repository (package `axisforge_bridge`), which has its own documentation, not reachable from here. |

---

## Design principles

- **Low coupling, high cohesion.** Solvers depend only on explicit result containers, never on each other's internals.
- **Declared surface, eager import.** A package states what it offers and imports it immediately, by explicit name — see the correction under [Importing](#importing).
- **No hidden state.** Every intermediate quantity is a public attribute; a solver exposes its full working for inspection.
- **Headless solvers.** The analysis stack runs without any interface layer; presentation is always a consumer, never a dependency.
- **Deterministic and explainable.** No black-box methods, no probabilistic life prediction, no machine learning.
- **Composition over inheritance.** A planetary train composes two pair-meshing objects; a bearing composes a family rather than subclassing one.
- **Capacity belongs to the element.** A solver never carries its own copy of a standard's formula.
- **Fail fast on geometry.** Validation happens at construction; invalid geometry is never silently accepted.
- **Declared, self-registering capability over hand-written dispatch.** A bearing family registers itself (`@register_family`); a beam formulation registers itself (`@register_formulation`) — the same shape used twice, and the public export list is generated from the registry rather than kept in sync by hand.
- **One file per concern, but shared structure through a common base where the shape genuinely is shared.** Each subtype, function group, solver and result shape lives in its own module or class; where two variants share real structure (a ball and a roller row, say), that structure lives once on a shared abstract base rather than being duplicated.
- **Flag, do not silently fix.** A suspected discrepancy against a standard, or a structural inconsistency found during a review pass, is documented in place, never quietly corrected.
- **`validate()` returns, `validate_or_raise()` raises.** Every domain object follows this pair.

---

## Roadmap

| Phase | Focus | Status |
|---|---|---|
| 1 | Shaft FEM (Timoshenko **and now Euler-Bernoulli**, via `BeamModelSettings`) · torsion split into its own solver · mesh convergence · gear force integration | Complete for Timoshenko; Euler-Bernoulli implemented. A 3-node quadratic Timoshenko element (`QuadraticTimoshenkoElem`) is reserved in the class hierarchy but deliberately not functional — see [`mesh/README.md`](axisforge/mesh/README.md#elements) |
| 2 | Bearing families, capability dispatch, ISO/TS 16281 load distribution | Complete for radial ball, radial roller and multi-row thrust; `ThrustNeedleRollerFamily` (previously listed here) is currently absent from `core/`; no multi-row roller family in `core/` to exercise the roller multi-row solver against |
| 3 | Result containers extracted into `results/` | Complete for bearings — single- and multi-row now share one container shape (`BearingResult`, `n_rows >= 1`), not separate single-row/multi-row containers |
| 4 | Capability declaration and fixture library — Construction and Studies stages | **Moved out of this repository.** This phase now lives entirely in the separate `AxisForge-Design-Studies` repository (`axisforge_bridge`), which consumes `axisforge` as a dependency — see the notes under [Status](#status) and [Analysis pipeline](#analysis-pipeline). Its progress is tracked there, not here. |
| 5 | ISO 6336 gear load capacity — data layer in place, solver module reserved | In progress |
| 6 | ISO 281 rating life as a standalone solver | Planned — `results/bearings/life/basic_life_results.py` currently implements only the ISO/TS 16281 *reference* life; a `L10` catalogue-method sibling is reserved but not started |
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