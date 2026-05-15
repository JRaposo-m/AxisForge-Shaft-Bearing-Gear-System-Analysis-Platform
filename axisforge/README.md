# AxisForge — Shaft–Bearing–Gear System Analysis Platform

---

## Executive Summary

**AxisForge** is an open-source, modular desktop application for the static, structural, and tribological analysis of shaft–bearing–gear mechanical systems. Developed in Python with a PySide6 interface, it targets mechanical engineers and researchers who require an integrated, traceable, and extensible analysis environment for rotating machinery design.

The platform is not a replacement for commercial tools such as KISSsoft or Romax. It is a technically rigorous, academically grounded alternative that prioritises transparency of calculation, modular extensibility, and systematic decision support over comprehensive feature breadth.

AxisForge integrates the **GEARpie** gear calculation subsystem as its native gear analysis engine and is architected from the outset to support incremental capability expansion without structural refactoring.

---

## Problem Statement

The design of shaft–bearing–gear systems involves tightly coupled mechanical phenomena: shaft deflection influences bearing load distribution; gear-transmitted loads create complex bending and torsion combinations; bearing selection depends on dynamic load ratings that are themselves a function of the full system geometry. In practice, these interactions are often analysed in isolation using separate tools, spreadsheets, or hand calculations, leading to:

- Loss of traceability between geometric assumptions and analytical results
- Manual and error-prone iteration between gear loads, shaft deflection, and bearing selection
- Absence of a unified model representing the system as a whole
- No structured path from calculation to design recommendation

AxisForge addresses this by maintaining a single parametric model of the mechanical system from which all analyses are derived consistently and simultaneously.

---

## Technical Value Proposition

| Capability | Description |
|---|---|
| **Unified system model** | Shaft geometry, component positioning, and loads are defined once and shared across all solvers |
| **Integrated gear–shaft coupling** | GEARpie-computed gear loads are applied directly to the shaft model at the correct axial positions |
| **Bearing analysis with catalogue data** | L10 life and static safety factor computed against a structured SKF catalogue database |
| **Traceable calculation chain** | Every result links back to its governing equation, assumption, and input parameter |
| **Modular solver architecture** | Solvers are independent, testable units; replacing or extending a solver does not affect the rest of the system |
| **Failure Likelihood Assessment (roadmap)** | Deterministic multi-criteria scoring engine that converts solver outputs into relative risk classifications (LOW / MEDIUM / HIGH) per failure mode |
| **Design intelligence layer (roadmap)** | Rule-based recommendation engine covering bearing arrangement, fits, and lubrication selection |

---

## System Architecture

### Directory Structure

```
axisforge/
│
├── core/                          # Domain model — system entities and data structures
│   ├── __init__.py                # Package marker (__version__ = "0.1.0")
│   ├── shaft.py                   # Shoulder, ShaftSection, Shaft
│   ├── components.py              # Bearing, BearingType, GearElement
│   ├── loads.py                   # RadialLoad, AxialLoad, TorqueLoad, ExternalMoment, LoadPlane
│   ├── system.py                  # MechanicalSystem — top-level assembly container
│   ├── materials.py               # Material, embedded library (S355, 42CrMo4, AISI 1045, AISI 4340)
│   └── tolerances.py              # Fit and tolerance data structures (ISO 286) — Phase 3
│
├── solvers/                       # Analytical engines — stateless, independently testable
│   ├── __init__.py
│   ├── statics.py                 # StaticsSolver — reactions, V(x), M(x), T(x) — Phase 1
│   ├── stress.py                  # StressSolver — combined stress, Goodman, ASME-Elliptic — Phase 1
│   ├── bearing_life.py            # BearingLifeSolver — L10, C/P, static safety (ISO 281) — Phase 1
│   └── critical_speed.py          # (Phase 5) Lateral critical speed estimation
│
├── models/                        # Result dataclasses — output of solvers
│   ├── __init__.py
│   ├── statics_result.py          # StaticsResult — Phase 1
│   ├── stress_result.py           # StressResult, CriticalSection — Phase 1
│   ├── bearing_result.py          # BearingLifeResult — Phase 1
│   └── system_report.py           # Aggregator of all results — Phase 1
│
├── integrations/                  # External module adapters
│   ├── __init__.py
│   ├── gearpie_adapter.py         # GEARpie → AxisForge load and geometry interface
│   └── skf_connector.py           # SKF catalogue query and bearing data retrieval — Phase 3
│
├── database/                      # Structured data — catalogues and material libraries
│   ├── skf_bearings.db            # SQLite: SKF rolling bearing catalogue — Phase 3
│   ├── fits_iso286.csv            # ISO 286 tolerance tables — Phase 3
│   └── schema/
│       └── bearings_schema.sql
│
├── ui/                            # PySide6 graphical interface — Phase 2
│   ├── __init__.py
│   ├── main_window.py
│   ├── shaft_canvas.py
│   ├── component_panel.py
│   ├── properties_panel.py
│   ├── results_panel.py
│   ├── dialogs/
│   │   ├── bearing_selector.py
│   │   ├── gear_import.py
│   │   └── section_editor.py
│   └── widgets/
│       ├── diagram_widget.py
│       └── bmd_plot.py
│
├── selectors/                     # Decision-support logic — Phase 3–4
│   ├── __init__.py
│   ├── bearing_selector.py
│   ├── arrangement_advisor.py     # Phase 4
│   ├── lubrication_advisor.py     # Phase 4
│   └── failure_assessor.py        # Phase 3+
│
├── reports/                       # Output generation — Phase 3
│   ├── __init__.py
│   ├── report_generator.py
│   └── templates/
│       └── standard_report.html
│
├── tests/                         # Unit and integration tests (pytest)
│   ├── conftest.py                # Global fixtures
│   ├── test_core/
│   │   ├── test_shaft.py
│   │   ├── test_components.py
│   │   ├── test_loads.py
│   │   ├── test_materials.py
│   │   └── test_system.py
│   ├── test_solvers/
│   │   ├── test_statics.py        # Phase 1 — in progress
│   │   ├── test_stress.py         # Phase 1 — planned
│   │   └── test_bearing_life.py   # Phase 1 — planned
│   ├── test_integrations/
│   │   └── test_gearpie_adapter.py
│   └── fixtures/
│       ├── sample_systems.py
│       └── expected_results.py
│
├── docs/                          # Technical documentation
│   ├── theory/
│   │   ├── shaft_analysis.md
│   │   ├── bearing_life.md
│   │   ├── stress_criteria.md
│   │   └── failure_assessment.md
│   └── api/
│
├── axisforge_cli.py               # CLI entry point (Phase 1 deliverable)
├── config.py                      # Global constants: SOLVER_RESOLUTION, TOL_GEOMETRY_mm, etc.
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Module and Class Definitions

### `core/shaft.py`

**`Shoulder`**
Geometric transition between two adjacent shaft sections (diameter step with fillet).

| Attribute | Type | Description |
|---|---|---|
| `fillet_radius` | float | Fillet radius r [mm] |
| `diameter_large` | float | Larger diameter D [mm] |
| `diameter_small` | float | Smaller diameter d [mm] |

Derived properties: `r_over_d`, `D_over_d` — used for Peterson interpolation.

**`ShaftSection`**
Represents a single uniform cylindrical segment of a shaft.

| Attribute | Type | Description |
|---|---|---|
| `length` | float | Axial length [mm] |
| `diameter` | float | Outer diameter [mm] |
| `inner_diameter` | float | Inner diameter (hollow); 0.0 for solid [mm] |
| `material_id` | str | Reference to `core/materials.py` embedded library |
| `surface_finish_ra` | float | Ra [μm]; used for Marin factor kₐ |
| `shoulder_left` | `Shoulder \| None` | Left-side shoulder transition |
| `shoulder_right` | `Shoulder \| None` | Right-side shoulder transition |
| `label` | str | Optional human-readable identifier |

Derived properties: `radius`, `area`, `second_moment_of_area`, `polar_moment`, `section_modulus`, `polar_section_modulus`, `is_hollow`.

**`Shaft`**
Ordered collection of `ShaftSection` objects representing the complete shaft.

| Method / Property | Returns | Description |
|---|---|---|
| `add_section(section)` | None | Appends a section to the right end |
| `total_length` | float | Sum of all section lengths [mm] |
| `n_sections` | int | Number of sections |
| `axial_start(index)` | float | Absolute start position of section[index] [mm] |
| `axial_end(index)` | float | Absolute end position of section[index] [mm] |
| `section_at(x)` | `tuple[ShaftSection, int]` | Section and index containing coordinate x (right-hand rule at boundaries) |
| `diameter_at(x)` | float | Outer diameter at x [mm] |
| `I_at(x)` | float | Second moment of area at x [mm⁴] |
| `J_at(x)` | float | Polar moment at x [mm⁴] |
| `W_at(x)` | float | Bending section modulus at x [mm³] |
| `Wt_at(x)` | float | Polar section modulus at x [mm³] |
| `shoulders()` | `list[tuple[float, Shoulder]]` | All shoulders with their axial positions |
| `validate()` | `list[str]` | Geometry errors; empty list if valid |
| `validate_or_raise()` | None | Raises `ValueError` if invalid |

---

### `core/components.py`

**`Bearing`**
Rolling bearing placed on the shaft at a specified axial position.

| Attribute | Type | Description |
|---|---|---|
| `position` | float | Axial position from shaft datum [mm] |
| `designation` | str | SKF designation (e.g., `"6210"`); optional in Phase 1 |
| `bearing_type` | `BearingType` | Enum: `DEEP_GROOVE_BALL`, `ANGULAR_CONTACT_BALL`, `CYLINDRICAL_ROLLER`, `TAPER_ROLLER`, `SPHERICAL_ROLLER` |
| `C` | float | Basic dynamic load rating [N] |
| `C0` | float | Basic static load rating [N] |
| `arrangement` | str | `"fixed"` or `"floating"` |
| `contact_angle` | float | α [°]; 0 for radial DGBB |
| `X` | float | Radial load factor (Phase 1: 1.0) |
| `Y` | float | Axial load factor (Phase 1: 0.0) |
| `label` | str | Optional identifier |

Derived properties: `has_catalogue_data`, `life_exponent` (3.0 for ball, 10/3 for roller).

**`GearElement`**
Gear transmitting forces and torque into the shaft.

| Attribute | Type | Description |
|---|---|---|
| `position` | float | Axial position [mm] |
| `tangential_force` | float | Wt [N] |
| `radial_force` | float | Wr [N] |
| `axial_force` | float | Wa [N]; 0 for spur gears |
| `pitch_diameter` | float | d [mm]; used to compute torque from Wt |
| `torque` | float | T [N·mm]; auto-computed from Wt×d/2 if zero and d > 0 |
| `pressure_angle` | float | αₙ [°] |
| `helix_angle` | float | β [°]; 0 for spur gears |
| `label` | str | Optional identifier |

Derived properties: `is_helical`, `resultant_transverse_force`.

**`TorqueLoad` / `RadialLoad` / `AxialLoad` / `ExternalMoment`**
Point loads applied at a specified axial position. All defined in `core/loads.py`.

---

### `core/materials.py`

**`Material`** — frozen dataclass.

| Attribute | Type | Description |
|---|---|---|
| `material_id` | str | Unique identifier |
| `Sut` | float | Ultimate tensile strength [MPa] |
| `Sy` | float | Yield strength [MPa] |
| `E` | float | Young's modulus [GPa] |
| `density` | float | Density [kg/m³] |
| `Se_base` | float \| None | Specimen endurance limit [MPa]; if None, computed as min(0.5×Sut, 700) |

Derived properties: `endurance_limit` (applies Shigley §6-2 rule), `shear_yield_strength` (0.577×Sy).

**Embedded library:** `S355`, `CrMo42` (42CrMo4), `AISI_1045`, `AISI_4340`.

**Helper functions:** `get_material(material_id)`, `available_materials()`.

---

### `core/system.py`

**`MechanicalSystem`**
Top-level container representing one shaft with all its components and loads.

| Method / Property | Returns | Description |
|---|---|---|
| `add_bearing(bearing)` | None | Registers a bearing; enforces position within shaft |
| `add_gear(gear)` | None | Registers a gear element |
| `add_load(load)` | None | Registers any point load or moment |
| `bearings` | `list[Bearing]` | Sorted by axial position |
| `gears` | `list[GearElement]` | Sorted by axial position |
| `support_positions` | `list[float]` | Axial positions of all bearings |
| `radial_loads_xz` | `list[RadialLoad]` | Filtered by plane |
| `radial_loads_yz` | `list[RadialLoad]` | Filtered by plane |
| `axial_loads` | `list[AxialLoad]` | — |
| `torque_loads` | `list[TorqueLoad]` | — |
| `external_moments` | `list[ExternalMoment]` | — |
| `validate()` | `list[str]` | System-level consistency checks |
| `validate_or_raise()` | None | Raises `ValueError` if invalid |

---

### `solvers/statics.py` *(Phase 1 — in progress)*

**`StaticsSolver`**
Computes bearing reactions and internal force/moment distributions. Two-support shaft handled analytically in two independent planes (XZ and YZ).

| Method | Returns | Description |
|---|---|---|
| `solve(system)` | `StaticsResult` | Full analysis: reactions + discretised diagrams |

`StaticsResult` (in `models/statics_result.py`) contains: `x`, `V_xz`, `V_yz`, `M_xz`, `M_yz`, `M_res`, `T`, `axial_force`, `reactions`.

---

### `solvers/stress.py` *(Phase 1 — planned)*

**`StressSolver`**
Computes nominal and corrected stresses at critical cross-sections; evaluates fatigue safety factors.

| Method | Returns | Description |
|---|---|---|
| `solve(system, statics_result, material)` | `StressResult` | Full stress analysis at all critical sections |

Stress concentration factors (Kt, Kf) computed from Peterson curves interpolated against r/d and D/d.

---

### `solvers/bearing_life.py` *(Phase 1 — planned)*

**`BearingLifeSolver`**
Implements ISO 281 L10 life calculation and static safety verification.

| Method | Returns | Description |
|---|---|---|
| `solve_bearing(bearing, Fr, Fa, speed_rpm, design_life_hours)` | `BearingLifeResult` | L10h, S₀, C/P for a single bearing |
| `extract_bearing_forces(statics_result, system)` | `dict` | Fr and Fa per bearing from StaticsResult |

Phase 1 uses X=1, Y=0 (radial-dominant simplification). Documented as known limitation.

---

### `integrations/gearpie_adapter.py` *(Phase 1 — planned)*

**`GEARpieAdapter`**
Translates GEARpie output files into `GearElement` instances.

| Method | Returns | Description |
|---|---|---|
| `load(path, axial_position, label)` | `GEARpieImportResult` | Parses JSON or CSV; validates fields and physical consistency |

Supported formats: JSON (preferred), CSV (legacy). Validates Wt/Wr/Wa consistency against gear geometry.

---

### `selectors/bearing_selector.py` *(Phase 3)*

**`BearingSelector`**
Queries the SKF catalogue database and returns feasible bearing candidates ranked by L10 life.

---

### `reports/report_generator.py` *(Phase 3)*

**`ReportGenerator`**
Assembles a structured calculation report from solver results. Output: HTML (primary), PDF via `weasyprint`.

---

### `selectors/failure_assessor.py` *(Phase 3+)*

**`FailureLikelihoodAssessor`**
Deterministic multi-criteria scoring engine. Each failure mode is evaluated as:

```
score(Fᵢ) = Σ [ wⱼ × fⱼ(xⱼ) ]   for j = 1..N_drivers
```

Risk classification:

| Score | Risk class |
|---|---|
| < 0.25 | LOW |
| 0.25 – 0.60 | MEDIUM |
| > 0.60 | HIGH |

Supported failure modes and availability by phase — see `docs/theory/failure_assessment.md`.

---

## Phase Roadmap

### Phase 1 — Core Solver Foundation *(months 1–3)*

**Goal:** Validated solver pipeline, no GUI dependency.

- [x] `Shaft`, `ShaftSection`, `Shoulder` — geometry model
- [x] `Bearing`, `BearingType`, `GearElement` — components
- [x] `RadialLoad`, `AxialLoad`, `TorqueLoad`, `ExternalMoment` — loads
- [x] `MechanicalSystem` — top-level container with validation
- [x] `Material`, embedded library (S355, 42CrMo4, AISI 1045, AISI 4340)
- [x] Full test suite for `core/` — 209 tests, 98% coverage
- [ ] `StaticsSolver` — reactions, V(x), M(x), T(x)
- [ ] `StressSolver` — von Mises, Goodman, ASME-Elliptic, Kf at shoulders
- [ ] `BearingLifeSolver` — ISO 281 L10h, static safety factor S₀
- [ ] `GEARpieAdapter` — JSON/CSV import, force extraction, consistency validation
- [ ] `axisforge_cli.py` — CLI demonstrating full pipeline

**Deliverable:** CLI script printing full analysis of a defined shaft system.

---

### Phase 2 — Base GUI *(months 3–5)*

**Goal:** Functional PySide6 interface for interactive system definition and result visualisation.

- [ ] Main window layout: canvas, component panel, properties panel, results tabs
- [ ] Shaft schematic canvas
- [ ] Real-time solver update on model change
- [ ] GEARpie import wizard

**Deliverable:** Desktop application visualising all Phase 1 analysis results.

---

### Phase 3 — SKF Catalogue + Failure Assessment Foundation *(months 5–7)*

**Goal:** Database-driven bearing selection; first failure mode evaluations.

- [ ] SQLite schema and population for SKF rolling bearing data
- [ ] `BearingSelector.query()` — filter by bore, load, speed
- [ ] `FailureLikelihoodAssessor` framework — F01, F02, F06, F07, F08
- [ ] Failure Assessment tab in results panel

---

### Phase 4 — Recommendation Engine + Full FLA *(months 7–10)*

**Goal:** Complete rule-based engineering guidance layer.

- [ ] Bearing arrangement advisor (fixed/floating, O/X)
- [ ] Fit advisor (ISO 286)
- [ ] `LubricationAdvisor` — EHL film parameter Λ, grease/oil selection
- [ ] Remaining failure modes: F03, F04, F05, F09, F10, F11
- [ ] HTML/PDF report generation

---

### Phase 5 — Expansions *(beyond month 10)*

Architecturally anticipated but out of initial scope:

- Multi-shaft gearbox assembly
- 1D FEM deflection solver
- Thermal model
- Fatigue life under variable loading (Palmgren–Miner)
- Parametric study runner

---

## Minimum Viable Product (MVP)

The MVP is the combined output of Phase 1 and Phase 2.

### In scope

- Single horizontal shaft, two bearing supports (statically determinate)
- Shaft defined by up to 10 cylindrical sections with constant diameter per section
- Shoulder geometry at section transitions (fillet radius input)
- Placement of bearings, gear elements, and point loads/moments at specified axial positions
- GEARpie result import and force extraction
- Static analysis: bearing reactions, V(x), M(x), T(x)
- Stress analysis: combined bending + torsion at shoulder cross-sections; Goodman safety factor
- Bearing life: ISO 281 L10h for manually entered C and n
- PySide6 GUI: shaft schematic, component tree, properties panel, results with diagrams
- JSON-based project save/load

### Explicitly excluded from MVP

| Feature | Reason |
|---|---|
| Multi-bearing indeterminate systems (>2 supports) | Requires deflection solver |
| SKF catalogue database | Phase 3 |
| Shaft deflection calculation | Phase 4 / FEM |
| Gear geometry design | Handled by GEARpie |
| Failure Likelihood Assessment | Phase 3 |
| Lubrication advisor | Phase 4 |
| Critical speed analysis | Phase 5 |
| Report generation | Phase 3/4 |
| Thermal analysis | Phase 5 |

---

## Incremental Development Strategy

**Principle 1 — Solvers are independent of the GUI.**
Solvers accept data model objects and return result dataclasses. Fully testable via pytest before any GUI work.

**Principle 2 — Data model is the integration boundary.**
`core/` defines the domain entities all other modules depend on. Stable since end of Phase 1 week 2.

**Principle 3 — Each component position is a single float.**
All axial positioning by coordinate (mm from datum), not section index.

**Principle 4 — GEARpie integration is encapsulated.**
`GEARpieAdapter` is the only module that knows GEARpie's output format.

**Principle 5 — Database is optional until Phase 3.**
Phase 1–2: C and C0 entered manually. Database is a query optimisation, not a structural dependency.

**Development order per session:**
1. Write the data structure
2. Write the test with a hand-calculated reference (TDD)
3. Write the solver logic until tests pass
4. Only then connect to GUI

---

## Quality Criteria

**Modularity:** Imports flow inward (`ui` → `solvers` → `core`). No circular dependencies. `core/` has zero external library dependencies beyond NumPy and stdlib.

**Testability:** Every solver is a pure function. No global state, no GUI calls inside solvers. Coverage target: ≥ 90% for `solvers/` and `core/`. At least one textbook-validated reference case per solver.

**Physical validity:** All calculations reference the governing standard or equation in docstrings. Simplifying assumptions documented explicitly.

**Scalability:** Adding a solver requires one new file in `solvers/`, one result dataclass, one new tab in the results panel. No changes to existing modules.

**Numerical robustness:** NumPy throughout. Shaft discretisation resolution configurable in `config.py`. Degenerate geometry caught at `validate()` time before solvers are called.

---

## Installation

```bash
git clone https://github.com/<user>/axisforge.git
cd axisforge
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python axisforge_cli.py          # Phase 1 CLI
# python main.py                 # Phase 2+ GUI
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

**Development dependencies:**

```
pytest
pytest-cov
ruff
mypy
```

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=core --cov=solvers --cov-report=term-missing
```

Current status: 209 tests, 98% coverage (`core/` complete).

Validation cases are documented in `tests/fixtures/expected_results.py` with source, chapter, example number, and expected values.

---

## GEARpie Integration

GEARpie operates as an independent module. AxisForge reads GEARpie output files via `GEARpieAdapter` — it never calls GEARpie functions directly. If GEARpie's output schema changes, only `integrations/gearpie_adapter.py` requires modification.

Supported format: JSON (preferred), CSV (legacy). Interface specification: `docs/api/gearpie_interface.md`.

---

## Project Status

| Phase | Status |
|---|---|
| Phase 1 — Core Solver | 🟡 In progress — `core/` complete, solvers next |
| Phase 2 — Base GUI | 🔲 Planned |
| Phase 3 — SKF Catalogue + FLA Foundation | 🔲 Planned |
| Phase 4 — Recommendation Engine + Full FLA | 🔲 Planned |
| Phase 5 — Expansions | 🔲 Architectural placeholder |

---

## Licence

MIT Licence. See `LICENSE` for terms.

GEARpie is integrated as a separately licensed subsystem. Refer to its own licence for applicable terms.

---

## References

- ISO 281:2007 — *Rolling bearings — Dynamic load ratings and rating life*
- ISO 76:2006 — *Rolling bearings — Static load ratings*
- ISO 6336-1/2/3:2019 — *Calculation of load capacity of spur and helical gears*
- ISO 286-1:2010 — *Geometrical product specifications — Limits and fits*
- AGMA 9005 — *Industrial Gear Lubrication*
- AGMA 1010-F14 — *Appearance of Gear Teeth — Terminology of Wear and Failure*
- Shigley, J.E., Budynas, R.G., Nisbett, J.K. — *Mechanical Engineering Design*, 10th ed.
- SKF General Catalogue — *Rolling Bearings*, publication 10000 EN
- SKF — *Bearing Failures and Their Causes*, publication PUB PT 01
- Peterson, R.E. — *Stress Concentration Factors*, 3rd ed.
- Harris, T.A., Kotzalas, M.N. — *Rolling Bearing Analysis*, 5th ed.
- Stachowiak, G.W., Batchelor, A.W. — *Engineering Tribology*, 4th ed.
- Sadeghi, F. et al. — *A Review of Rolling Contact Fatigue*, ASME J. Tribology, 2009