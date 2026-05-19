# AxisForge — Shaft–Bearing–Gear System Analysis Platform

AxisForge is an open-source, modular Python platform for the static, structural, and tribological analysis of shaft–bearing–gear mechanical systems. It prioritises transparency of calculation, modular extensibility, and traceable decision support over comprehensive feature breadth.

AxisForge integrates the **GEARpie** gear calculation subsystem as its native gear analysis engine.

---

## Axis Convention

```
x   = axial axis; datum = left end of shaft; increases rightward
XZ  = horizontal plane  (Wt gear force, lateral loads)
XY  = vertical plane    (Wr gear force, gravity direction)

Positive radial load: downward (XY), forward (XZ)
Reactions:  sign determined by equilibrium equations (not forced positive)
Torsion T:  accumulates left-to-right; positive = CCW when viewed from +x
```

All internal arrays use this convention. It is documented in `models/statics_result.py` and must not be changed without updating all solvers and tests.

---

## System Architecture

### Directory Structure

```
axisforge/
│
├── core/                          # Domain model — system entities and data structures
│   ├── __init__.py
│   ├── shaft.py                   # Shoulder, ShaftSection, Shaft
│   ├── components.py              # Bearing, BearingType, GearElement
│   ├── loads.py                   # RadialLoad, AxialLoad, TorqueLoad, ExternalMoment, LoadPlane
│   ├── system.py                  # MechanicalSystem — top-level assembly container
│   ├── materials.py               # Material, embedded library (S355, 42CrMo4, AISI 1045, AISI 4340)
│   └── tolerances.py              # Fit and tolerance data structures (ISO 286) — Phase 3
│
├── solvers/                       # Analytical engines — stateless, independently testable
│   ├── statics.py                 # StaticsSolver — reactions, V(x), M(x), T(x)
│   ├── stress.py                  # StressSolver — combined stress, Goodman, ASME-Elliptic
│   ├── bearing_life.py            # BearingLifeSolver — L10, C/P, static safety (ISO 281)
│   └── critical_speed.py          # Phase 5
│
├── models/                        # Result dataclasses — output of solvers
│   ├── statics_result.py          # StaticsResult
│   ├── stress_result.py           # StressResult, CriticalSection
│   ├── bearing_result.py          # BearingLifeResult
│   └── system_report.py           # Aggregator of all results
│
├── integrations/
│   ├── gearpie_adapter.py         # GEARpie → AxisForge load and geometry interface
│   └── skf_connector.py           # SKF catalogue query — Phase 3
│
├── database/                      # Phase 3
│   ├── skf_bearings.db
│   └── schema/bearings_schema.sql
│
├── ui/                            # PySide6 graphical interface — Phase 2
│
├── selectors/                     # Decision-support logic — Phase 3–4
│   ├── bearing_selector.py
│   ├── arrangement_advisor.py
│   ├── lubrication_advisor.py
│   └── failure_assessor.py
│
├── reports/                       # Phase 3
│
├── tests/
│   ├── conftest.py                # Global fixtures
│   ├── test_core/
│   ├── test_solvers/
│   ├── test_integrations/
│   └── fixtures/
│       ├── sample_systems.py
│       └── expected_results.py    # Textbook reference values with source citations
│
├── axisforge_cli.py               # CLI entry point (Phase 1 deliverable)
├── config.py                      # Global constants
├── requirements.txt
└── pyproject.toml
```

### Import Dependency Rule

```
ui/ → solvers/ → core/
      models/  ← solvers/
      integrations/ → core/
```

`core/` imports numpy and stdlib only. No PySide6, matplotlib, or database calls inside `core/` or `solvers/`.

---

## Implementation Status

| Module | Status | Tests |
|---|---|---|
| `core/shaft.py` | ✅ Complete | 209/209, 98% coverage |
| `core/components.py` | ✅ Complete | included above |
| `core/loads.py` | ✅ Complete | included above |
| `core/materials.py` | ✅ Complete | included above |
| `core/system.py` | ✅ Complete | included above |
| `models/statics_result.py` | ✅ Complete | — |
| `solvers/statics.py` | ✅ Complete | 296/296, ≥90% coverage |
| `solvers/stress.py` | 🟡 In progress | auxiliary functions done; `solve()` next |
| `models/stress_result.py` | 🟡 In progress | to create this session |
| `solvers/bearing_life.py` | 🔲 Planned (Week 8) | — |
| `integrations/gearpie_adapter.py` | 🔲 Planned (Week 9) | — |
| `ui/` | 🔲 Phase 2 | — |
| `database/` | 🔲 Phase 3 | — |
| `selectors/` | 🔲 Phase 3+ | — |

---

## Module and Class Definitions

### `core/shaft.py`

**`Shoulder`** — geometric transition between two adjacent shaft sections.

| Attribute | Type | Description |
|---|---|---|
| `fillet_radius` | float | r [mm] |
| `diameter_large` | float | D [mm] |
| `diameter_small` | float | d [mm] |

Derived: `r_over_d`, `D_over_d` — inputs to Peterson interpolation.

**`ShaftSection`** — single uniform cylindrical segment.

| Attribute | Type | Description |
|---|---|---|
| `length` | float | [mm] |
| `diameter` | float | Outer diameter [mm] |
| `inner_diameter` | float | 0.0 for solid [mm] |
| `material_id` | str | Key into `core/materials.py` embedded library |
| `surface_finish_ra` | float | Ra [μm]; used for Marin factor kₐ |
| `shoulder_left` | `Shoulder \| None` | Left-side shoulder transition |
| `shoulder_right` | `Shoulder \| None` | Right-side shoulder transition |
| `label` | str | Optional identifier |

Derived: `radius`, `area`, `second_moment_of_area`, `polar_moment`, `section_modulus`, `polar_section_modulus`, `is_hollow`.

**`Shaft`** — ordered collection of `ShaftSection` objects.

| Method / Property | Returns | Description |
|---|---|---|
| `add_section(section)` | None | Appends section to right end |
| `total_length` | float | Sum of section lengths [mm] |
| `axial_start(index)` | float | Absolute start position of section[index] [mm] |
| `axial_end(index)` | float | Absolute end position of section[index] [mm] |
| `section_at(x)` | `tuple[ShaftSection, int]` | Section containing x |
| `diameter_at(x)` | float | Outer diameter at x [mm] |
| `I_at(x)` | float | Second moment of area at x [mm⁴] |
| `J_at(x)` | float | Polar moment at x [mm⁴] |
| `W_at(x)` | float | Bending section modulus at x [mm³] |
| `Wt_at(x)` | float | Polar section modulus at x [mm³] |
| `shoulders()` | `list[tuple[float, Shoulder]]` | All shoulders with axial positions |
| `validate()` | `list[str]` | Geometry errors; empty if valid |
| `validate_or_raise()` | None | Raises `ValueError` if invalid |

---

### `core/components.py`

**`Bearing`**

| Attribute | Type | Description |
|---|---|---|
| `position` | float | Axial position from datum [mm] |
| `designation` | str | e.g. `"6210"` |
| `bearing_type` | `BearingType` | `DEEP_GROOVE_BALL`, `ANGULAR_CONTACT_BALL`, `CYLINDRICAL_ROLLER`, `TAPER_ROLLER`, `SPHERICAL_ROLLER` |
| `C` | float | Basic dynamic load rating [N] |
| `C0` | float | Basic static load rating [N] |
| `arrangement` | str | `"fixed"` or `"floating"` |
| `X` | float | Radial load factor (Phase 1: 1.0) |
| `Y` | float | Axial load factor (Phase 1: 0.0) |
| `label` | str | Optional identifier |

Derived: `life_exponent` (3.0 ball, 10/3 roller).

**`GearElement`**

| Attribute | Type | Description |
|---|---|---|
| `position` | float | Axial position [mm] |
| `tangential_force` | float | Wt [N] → XZ plane |
| `radial_force` | float | Wr [N] → XY plane |
| `axial_force` | float | Wa [N]; 0 for spur |
| `pitch_diameter` | float | d [mm] |
| `torque` | float | T [N·mm]; auto-computed from Wt×d/2 if zero |
| `label` | str | Optional identifier |

---

### `core/materials.py`

**`Material`** — frozen dataclass.

| Attribute | Type | Description |
|---|---|---|
| `material_id` | str | Unique key |
| `Sut` | float | Ultimate tensile strength [MPa] |
| `Sy` | float | Yield strength [MPa] |
| `E` | float | Young's modulus [GPa] |
| `Se_base` | float \| None | Specimen endurance limit; if None → min(0.5×Sut, 700 MPa) |

Derived: `endurance_limit` (Shigley §6-2), `shear_yield_strength` (0.577×Sy).

Embedded library: `S355`, `CrMo42`, `AISI_1045`, `AISI_4340`.

---

### `core/system.py`

**`MechanicalSystem`**

| Method / Property | Returns | Description |
|---|---|---|
| `add_bearing(bearing)` | None | Registers; enforces position within shaft |
| `add_gear(gear)` | None | Registers gear |
| `add_load(load)` | None | Registers any load or moment |
| `bearings` | `list[Bearing]` | Sorted by position |
| `gears` | `list[GearElement]` | Sorted by position |
| `radial_loads_xz` | `list[RadialLoad]` | Filtered by plane |
| `radial_loads_xy` | `list[RadialLoad]` | Filtered by plane |
| `axial_loads` | `list[AxialLoad]` | — |
| `torque_loads` | `list[TorqueLoad]` | — |
| `external_moments` | `list[ExternalMoment]` | — |
| `validate_or_raise()` | None | Raises `ValueError` if invalid |

---

### `solvers/statics.py`

**`StaticsSolver`** — two-support shaft, statically determinate. Planes solved independently.

| Method | Returns | Description |
|---|---|---|
| `solve(system)` | `StaticsResult` | Reactions + discretised diagrams V(x), M(x), T(x) |

`StaticsResult` fields: `x`, `V_xz`, `V_xy`, `M_xz`, `M_xy`, `M_res`, `T`, `axial_force`, `reactions`.
Reactions dict keys: `"A_xz"`, `"A_xy"`, `"B_xz"`, `"B_xy"`, `"axial"`.

---

### `solvers/stress.py`

**`StressSolver`** — critical section identification, Kf/Kfs, Marin Se', fatigue criteria.

| Method | Returns | Description |
|---|---|---|
| `solve(system, statics_result, material)` | `StressResult` | Full stress analysis at all shoulder sections |

`StressResult` (in `models/stress_result.py`): list of `CriticalSection` objects sorted by `nf_goodman` ascending. Each `CriticalSection` exposes: `x`, `Kf`, `Kfs`, `sigma_a`, `tau_m`, `Se_prime`, `nf_goodman`, `nf_asme`, `von_mises_eq`.

Phase 1 assumptions: `kc = 1.0` (combined bending+torsion, per Shigley §6-14); `D/d = 1.5` fixed for Peterson; no keyway or press-fit stress raisers.

---

### `solvers/bearing_life.py`

**`BearingLifeSolver`** — ISO 281 L10 life and static safety.

| Method | Returns | Description |
|---|---|---|
| `solve_bearing(bearing, Fr, Fa, speed_rpm, design_life_hours)` | `BearingLifeResult` | L10h, S₀, C/P |
| `extract_bearing_forces(statics_result, system)` | `dict` | Fr and Fa per bearing |

Phase 1: X=1, Y=0. Documented limitation.

---

### `integrations/gearpie_adapter.py`

**`GEARpieAdapter`** — sole module that reads GEARpie output format.

| Method | Returns | Description |
|---|---|---|
| `load(path, axial_position, label)` | `GEARpieImportResult` | JSON or CSV; validates physical consistency |

---

## Incremental Development Principles

1. Solvers are GUI-independent. Fully testable before any Phase 2 work.
2. `core/` is the integration boundary. Stable since Week 2.
3. Component positions are floats (mm from datum), not section indices.
4. `GEARpieAdapter` is the only module that knows GEARpie's output format.
5. Database (SQLite) not used in Phase 1–2. C and C0 entered manually.
6. Development order per solver: data structure → test with reference value (TDD) → implement → pass.

---

## config.py Constants

| Constant | Value | Purpose |
|---|---|---|
| `SOLVER_RESOLUTION` | 1000 | Array length for discretised diagrams |
| `BOUNDARY_MOMENT_TOLERANCE` | 500.0 N·mm | Max |M| at supports (validation check) |
| `SOLVER_TOLERANCE` | 1e-6 | General numerical guard |
| `DEFAULT_DESIGN_LIFE_HOURS` | 20 000 h | Bearing life default |
| `SUT_ENDURANCE_CAP_MPa` | 1400 MPa | Se = 0.5×Sut valid below this (Shigley §6-2) |
| `RELIABILITY_FACTOR_99` | 0.868 | ke, 99% (Shigley Tab. 6-6) |
| `MIN_FILLET_RADIUS_mm` | 0.01 mm | Guard for notch sensitivity calc |
| `FLA_THRESHOLD_LOW/HIGH` | 0.25 / 0.60 | Failure risk classification — Phase 3 |

---

## Quality Criteria

**Modularity:** Imports flow inward (`ui` → `solvers` → `core`). No circular dependencies. `core/` has zero external library dependencies beyond NumPy and stdlib.

**Testability:** Every solver is a pure function. No global state, no GUI calls inside solvers. Coverage target: ≥ 90% for `solvers/` and `core/`. At least one textbook-validated reference case per solver.

**Physical validity:** Governing equation or standard referenced in every solver docstring. Simplifying assumptions documented explicitly.

**Numerical robustness:** NumPy throughout. Degenerate geometry caught at `validate_or_raise()` before solvers run.

---

## Installation

```bash
git clone https://github.com/<user>/axisforge.git
cd axisforge
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python axisforge_cli.py          # Phase 1 CLI
```

**Runtime dependencies:**

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥ 1.26 | Numerical computation |
| `scipy` | ≥ 1.12 | Integration, interpolation |
| `PySide6` | ≥ 6.6 | GUI framework (Phase 2+) |
| `matplotlib` | ≥ 3.8 | Diagram rendering (Phase 2+) |
| `pandas` | ≥ 2.1 | Catalogue queries (Phase 3+) |
| `weasyprint` | ≥ 61 | PDF report output (Phase 3+) |

**Dev dependencies:** `pytest`, `pytest-cov`, `ruff`, `mypy`

---

## Testing

```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=core --cov=solvers --cov-report=term-missing

# Run a specific validation case
python -m pytest tests/test_solvers/test_statics.py::TestStaticsSolverValidationCases::test_shigley_ex3_6_reaction_A -v
```

Validation cases in `tests/fixtures/expected_results.py` include source, chapter, example number, and expected values with tolerances.

Current: 296 tests passing. Coverage: `core/` 98%, `solvers/statics.py` ≥ 90%.

---

## GEARpie Integration

AxisForge reads GEARpie output via `GEARpieAdapter` — it never calls GEARpie functions directly. If GEARpie's output schema changes, only `integrations/gearpie_adapter.py` requires modification.

Supported formats: JSON (preferred), CSV (legacy).

---

## MVP Scope

### In scope (Phase 1 + Phase 2)

- Single horizontal shaft, two bearing supports (statically determinate)
- Shaft defined by up to 10 cylindrical sections
- Shoulder geometry at section transitions (fillet radius)
- Bearings, gear elements, and point loads at specified axial positions
- GEARpie result import
- Static analysis: reactions, V(x), M(x), T(x)
- Stress analysis: combined bending + torsion at shoulder sections; Goodman safety factor
- Bearing life: ISO 281 L10h
- PySide6 GUI with shaft schematic, component tree, results with diagrams
- JSON project save/load

### Explicitly excluded from MVP

| Feature | Reason |
|---|---|
| Multi-bearing indeterminate systems | Requires deflection solver |
| SKF catalogue database | Phase 3 |
| Shaft deflection calculation | Phase 4/FEM |
| Gear geometry design | Handled by GEARpie |
| Failure Likelihood Assessment | Phase 3 |
| Lubrication advisor | Phase 4 |
| Critical speed analysis | Phase 5 |
| Report generation | Phase 3/4 |

---

## Licence

MIT Licence. GEARpie is integrated as a separately licensed subsystem.

---

## References

- ISO 281:2007 — *Rolling bearings — Dynamic load ratings and rating life*
- ISO 76:2006 — *Rolling bearings — Static load ratings*
- ISO 6336-1/2/3:2019 — *Calculation of load capacity of spur and helical gears*
- ISO 286-1:2010 — *Geometrical product specifications — Limits and fits*
- Shigley, J.E., Budynas, R.G., Nisbett, J.K. — *Mechanical Engineering Design*, 10th ed.
- Peterson, R.E. — *Stress Concentration Factors*, 3rd ed.
- SKF General Catalogue — *Rolling Bearings*, publication 10000 EN
- Harris, T.A., Kotzalas, M.N. — *Rolling Bearing Analysis*, 5th ed.
- Stachowiak, G.W., Batchelor, A.W. — *Engineering Tribology*, 4th ed.
- Sadeghi, F. et al. — *A Review of Rolling Contact Fatigue*, ASME J. Tribology, 2009