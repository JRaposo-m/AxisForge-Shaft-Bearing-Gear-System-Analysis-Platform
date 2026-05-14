# AxisForge — Shaft–Bearing–Gear System Analysis Platform

> **Alternative names considered:**
> - **AxisForge** — *forging mechanical systems through rigorous analysis* ✓ **(selected)**
> - **ShaftLab** — direct, descriptive, laboratory-grade connotation
> - **RotorCore** — emphasis on rotating machinery and structural core
> - **MechAxis** — mechanical axis analysis, clean and professional
> - **DrivelineAE** — analysis & engineering of drivetrain systems

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
│   ├── __init__.py
│   ├── shaft.py                   # Shaft, ShaftSection
│   ├── components.py              # Bearing, GearElement, Pulley, Coupling
│   ├── loads.py                   # RadialLoad, AxialLoad, ExternalMoment, TorqueLoad
│   ├── system.py                  # MechanicalSystem — top-level assembly container
│   └── tolerances.py              # Fit and tolerance data structures (ISO 286)
│
├── solvers/                       # Analytical engines — stateless, independently testable
│   ├── __init__.py
│   ├── statics.py                 # Reaction forces, shear and bending moment diagrams
│   ├── torsion.py                 # Torsional load distribution along shaft
│   ├── stress.py                  # Combined stress state, fatigue criteria (Goodman, ASME-Elliptic)
│   ├── deflection.py              # Shaft deflection and slope (Euler–Bernoulli beam, FEM-1D)
│   ├── bearing_life.py            # L10 life, C/P ratio, static safety (ISO 281)
│   └── critical_speed.py          # (Phase 4) Lateral critical speed estimation
│
├── integrations/                  # External module adapters
│   ├── __init__.py
│   ├── gearpie_adapter.py         # GEARpie → AxisForge load and geometry interface
│   └── skf_connector.py          # SKF catalogue query and bearing data retrieval
│
├── database/                      # Structured data — catalogues and material libraries
│   ├── skf_bearings.db            # SQLite: SKF rolling bearing catalogue (C, C0, geometry)
│   ├── materials.db               # SQLite: steel grades (E, σ_y, σ_u, S-N data)
│   ├── fits_iso286.csv            # ISO 286 tolerance tables
│   └── schema/
│       ├── bearings_schema.sql
│       └── materials_schema.sql
│
├── ui/                            # PySide6 graphical interface
│   ├── __init__.py
│   ├── main_window.py             # Application shell, menu bar, layout manager
│   ├── shaft_canvas.py            # Interactive schematic rendering (matplotlib/pyqtgraph)
│   ├── component_panel.py         # Hierarchical component tree (QTreeWidget)
│   ├── properties_panel.py        # Context-sensitive property editor (QFormLayout)
│   ├── results_panel.py           # Tabbed results: diagrams, tables, bearing summary
│   ├── dialogs/
│   │   ├── bearing_selector.py    # Guided bearing selection dialog
│   │   ├── gear_import.py         # GEARpie result import wizard
│   │   └── section_editor.py     # Shaft section geometry editor
│   └── widgets/
│       ├── diagram_widget.py      # Shear/bending/torsion diagram renderer
│       └── bmd_plot.py            # Bending moment diagram with matplotlib
│
├── selectors/                     # Decision-support logic (rule-based, Phase 3–4)
│   ├── __init__.py
│   ├── bearing_selector.py        # Filter catalogue by load, speed, space constraints
│   ├── arrangement_advisor.py     # (Phase 4) Fixed/floating, O/X arrangement logic
│   ├── lubrication_advisor.py     # (Phase 4) Grease/oil selection, relubrication interval
│   └── failure_assessor.py        # (Phase 3+) Failure Likelihood Assessment — deterministic scoring
│
├── reports/                       # Output generation
│   ├── __init__.py
│   ├── report_generator.py        # Structured PDF/HTML report builder
│   └── templates/
│       └── standard_report.html
│
├── tests/                         # Unit and integration tests (pytest)
│   ├── test_statics.py
│   ├── test_bearing_life.py
│   ├── test_stress.py
│   ├── test_gearpie_adapter.py
│   ├── test_failure_assessor.py
│   └── fixtures/
│       └── sample_systems.py      # Validated reference cases
│
├── docs/                          # Technical documentation
│   ├── theory/
│   │   ├── shaft_analysis.md      # Governing equations, assumptions, references
│   │   ├── bearing_life.md        # ISO 281 implementation notes
│   │   ├── stress_criteria.md     # Fatigue criteria derivation
│   │   └── failure_assessment.md  # FLA scoring model, failure mode catalogue, driver weights
│   └── api/                       # Auto-generated (sphinx or mkdocs)
│
├── scripts/
│   └── populate_skf_db.py         # One-time database population from CSV source
│
├── main.py                        # Application entry point
├── config.py                      # Global paths, units, solver tolerances
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Module and Class Definitions

### `core/shaft.py`

**`ShaftSection`**
Represents a single cylindrical segment of a shaft.

| Attribute | Type | Description |
|---|---|---|
| `length` | float | Axial length of the section [mm] |
| `diameter` | float | Outer diameter [mm] |
| `inner_diameter` | float | Inner diameter (hollow); 0.0 for solid [mm] |
| `material_id` | str | Reference to `materials.db` |
| `shoulder_left` | `Shoulder \| None` | Geometry of left-side shoulder transition |
| `shoulder_right` | `Shoulder \| None` | Geometry of right-side shoulder transition |
| `surface_finish` | float | Ra value [μm]; used for fatigue correction factor Kₐ |

**`Shaft`**
Ordered collection of `ShaftSection` objects representing the complete shaft.

| Method | Returns | Description |
|---|---|---|
| `total_length()` | float | Sum of all section lengths |
| `axial_position(section_index)` | float | Absolute start position of a section from datum [mm] |
| `section_at(x)` | `ShaftSection` | Returns the section containing axial coordinate x |
| `second_moment_of_area(x)` | float | I(x) [mm⁴] at position x; accounts for hollow/solid geometry |
| `polar_moment(x)` | float | J(x) [mm⁴] for torsional calculations |
| `validate()` | `list[str]` | Returns list of geometry errors or empty list if valid |

---

### `core/components.py`

**`Bearing`**
Defines a rolling bearing placed on the shaft at a specified axial position.

| Attribute | Type | Description |
|---|---|---|
| `position` | float | Axial position from shaft datum [mm] |
| `bearing_id` | str | SKF designation (e.g., `"6210"`) or `None` if unselected |
| `bearing_type` | `BearingType` | Enum: DEEP_GROOVE_BALL, CYLINDRICAL_ROLLER, ANGULAR_CONTACT, TAPER_ROLLER, SPHERICAL_ROLLER |
| `arrangement` | `str` | `"fixed"` or `"floating"` |
| `preload` | float | Axial preload [N]; 0.0 for unpretensioned |
| `catalogue_data` | `BearingData \| None` | Populated after database query |

**`GearElement`**
Represents a gear (or sprocket/pulley) transmitting torque and radial/axial forces into the shaft.

| Attribute | Type | Description |
|---|---|---|
| `position` | float | Axial position [mm] |
| `gearpie_result` | `dict \| None` | Imported GEARpie result dictionary |
| `tangential_force` | float | Wt [N]; computed from GEARpie or user-specified |
| `radial_force` | float | Wr [N] |
| `axial_force` | float | Wa [N] |
| `pressure_angle` | float | Normal pressure angle [°] |
| `helix_angle` | float | [°]; 0 for spur gears |
| `pitch_diameter` | float | d [mm]; used to compute torque from tangential force |

**`RadialLoad` / `AxialLoad` / `ExternalMoment`**
Point loads applied at a specified axial position. Used for belt tensions, imbalance forces, and externally applied moments.

---

### `core/system.py`

**`MechanicalSystem`**
Top-level container representing one shaft with all its components and loads.

| Method | Returns | Description |
|---|---|---|
| `add_bearing(bearing)` | None | Registers a bearing; enforces minimum two-support constraint |
| `add_gear(gear)` | None | Registers a gear element |
| `add_load(load)` | None | Registers any point load or moment |
| `validate()` | `list[str]` | System-level consistency checks before solving |
| `get_support_positions()` | `list[float]` | Returns axial positions of all bearing supports |
| `get_load_vector()` | `dict` | Aggregated loads by position and direction |

---

### `solvers/statics.py`

**`StaticsSolver`**
Computes bearing reactions and internal force/moment distributions for a statically determinate or indeterminate shaft (two-support case handled analytically; multi-support via influence coefficients).

| Method | Returns | Description |
|---|---|---|
| `solve(system)` | `StaticsResult` | Computes reactions in XZ and YZ planes separately |
| `shear_diagram(result, plane)` | `np.ndarray` | Shear force V(x) sampled at solver resolution |
| `bending_moment_diagram(result, plane)` | `np.ndarray` | M(x) in specified plane |
| `resultant_moment(result)` | `np.ndarray` | M_res(x) = √(Mxz² + Myz²) |

`StaticsResult` dataclass contains reactions at each bearing, plus discretised V(x), M(x), T(x) arrays.

---

### `solvers/stress.py`

**`StressSolver`**
Computes nominal and corrected stresses at critical cross-sections; evaluates fatigue safety factors.

| Method | Returns | Description |
|---|---|---|
| `critical_sections(system, statics_result)` | `list[CrossSection]` | Identifies shoulders, keyways, press fits as stress raisers |
| `von_mises_stress(section)` | float | σ_eq at section centroid [MPa] |
| `safety_factor_goodman(section, material)` | float | Modified Goodman fatigue safety factor |
| `safety_factor_asme(section, material)` | float | ASME Elliptic criterion |

Stress concentration factors (Kt, Kf) are computed from Peterson curves implemented as interpolating functions against geometry ratios (r/d, D/d).

---

### `solvers/bearing_life.py`

**`BearingLifeSolver`**
Implements ISO 281 L10 life calculation and static safety verification.

| Method | Returns | Description |
|---|---|---|
| `equivalent_dynamic_load(bearing, reactions)` | float | P = X·Fr + Y·Fa [N] |
| `L10_life(bearing, P)` | float | Basic rating life [10⁶ revolutions] |
| `L10h_life(bearing, P, speed)` | float | Rating life in operating hours [h] |
| `static_safety_factor(bearing, reactions)` | float | S₀ = C₀ / P₀ |

---

### `integrations/gearpie_adapter.py`

**`GEARpieAdapter`**
Translates GEARpie output (gear geometry and load data) into `GearElement` instances compatible with AxisForge's solver pipeline.

| Method | Returns | Description |
|---|---|---|
| `load_from_file(path)` | `dict` | Parses GEARpie JSON/CSV result file |
| `to_gear_element(data, position)` | `GearElement` | Builds AxisForge `GearElement` from GEARpie payload |
| `extract_forces(data)` | `tuple[float, float, float]` | Returns (Wt, Wr, Wa) in Newtons |
| `validate_compatibility(data)` | `list[str]` | Checks for required fields and unit consistency |

---

### `selectors/bearing_selector.py`

**`BearingSelector`**
Queries the SKF catalogue database and returns feasible bearing candidates ranked by L10 life, given geometric and load constraints.

| Method | Returns | Description |
|---|---|---|
| `query(bore, load_P, speed, L10h_target)` | `list[BearingData]` | Returns bearings meeting constraints, sorted by margin |
| `filter_by_type(candidates, bearing_type)` | `list[BearingData]` | Narrows candidates by type |
| `recommend(candidates, system_context)` | `BearingData` | (Phase 4) Applies arrangement and space heuristics |

---

### `reports/report_generator.py`

**`ReportGenerator`**
Assembles a structured calculation report from solver results. Output formats: HTML (primary), PDF via `weasyprint`.

| Method | Returns | Description |
|---|---|---|
| `build(system, results)` | `ReportDocument` | Aggregates all results into a structured document object |
| `export_html(doc, path)` | None | Renders report to HTML file |
| `export_pdf(doc, path)` | None | Renders report to PDF via weasyprint |

The report includes: system geometry summary, reaction forces table, shear/bending/torsion diagrams, stress summary at critical sections, bearing L10 table, failure assessment summary, and a section for engineering observations.

---

### `selectors/failure_assessor.py`

**`FailureModeResult`**
Dataclass carrying the full output of a single failure mode evaluation.

| Field | Type | Description |
|---|---|---|
| `mode_id` | str | Failure mode identifier (e.g., `"F01"`) |
| `mode_name` | str | Human-readable name (e.g., `"Rolling Contact Fatigue"`) |
| `component` | str | Affected component and position (e.g., `"Bearing 6210 at x=150 mm"`) |
| `score` | float | Normalised risk score [0.0–1.0] |
| `risk_class` | str | `"LOW"` / `"MEDIUM"` / `"HIGH"` |
| `drivers` | `list[DriverContribution]` | Individual driver values and weighted contributions |
| `recommendation` | str | Plain-language corrective action |

**`DriverContribution`**
Represents the contribution of a single risk driver to a failure mode score.

| Field | Type | Description |
|---|---|---|
| `name` | str | Driver name (e.g., `"C/P ratio"`) |
| `raw_value` | float | Value as computed by the relevant solver |
| `normalized_value` | float | f(raw_value) ∈ [0.0, 1.0] via piecewise linear map |
| `weight` | float | Configured weight for this driver within the failure mode |
| `contribution` | float | `weight × normalized_value` |

**`FailureLikelihoodAssessor`**
Orchestrates evaluation of all applicable failure modes for a given system state.

| Method | Returns | Description |
|---|---|---|
| `assess(system, statics_result, stress_result, bearing_results, user_inputs)` | `list[FailureModeResult]` | Evaluates all registered failure modes; returns list sorted by descending score |
| `assess_bearing(bearing_result, user_inputs)` | `list[FailureModeResult]` | Evaluates only bearing-related modes for a single bearing position |
| `summary(results)` | `dict` | Returns highest risk class per component for dashboard display |

**Scoring model:**

Each failure mode Fᵢ is evaluated as a weighted sum of normalised driver scores:

```
score(Fᵢ) = Σ [ wⱼ × fⱼ(xⱼ) ]   for j = 1..N_drivers
```

Risk classification thresholds:

| Score range | Risk class |
|---|---|
| < 0.25 | LOW |
| 0.25 – 0.60 | MEDIUM |
| > 0.60 | HIGH |

**Supported failure modes:**

| ID | Failure Mode | Component | Available from Phase |
|---|---|---|---|
| F01 | Rolling Contact Fatigue (RCF) | Bearing | 3 |
| F02 | Static Overload | Bearing | 2 |
| F03 | Lubrication Starvation | Bearing | 4 |
| F04 | Contamination Fatigue | Bearing | 4 |
| F05 | False Brinelling | Bearing | 4 |
| F06 | Shaft Bending Fatigue | Shaft | 2 |
| F07 | Shoulder Stress Concentration | Shaft | 2 |
| F08 | Gear Tooth Pitting | Gear | 3 |
| F09 | Gear Micropitting | Gear | 4 |
| F10 | Gear Scuffing | Gear | 4 |
| F11 | Misalignment Sensitivity | System | 4 |
| F12 | Thermal Degradation | Bearing | 5 |

**Key driver definitions:**

*F01 — Rolling Contact Fatigue:*

| Driver | Source | Weight |
|---|---|---|
| C/P ratio | `BearingLifeSolver` | 0.45 |
| L10h margin vs. target | `BearingLifeSolver` | 0.35 |
| Contamination class η_c (user input) | Manual | 0.20 |

*F06 — Shaft Bending Fatigue:*

| Driver | Source | Weight |
|---|---|---|
| Goodman safety factor nf | `StressSolver` | 0.50 |
| Fatigue stress concentration Kf | `StressSolver` | 0.30 |
| Mean-to-alternating stress ratio σm/σa | `StressSolver` | 0.20 |

*F03 — Lubrication Starvation (Phase 4):*

| Driver | Source | Weight |
|---|---|---|
| EHL film parameter Λ = h_min / R_composite | `LubricationAdvisor` | 0.55 |
| Speed ratio n / n_lim | Catalogue data | 0.25 |
| Operating temperature margin vs. grease limit | User input | 0.20 |

The Λ parameter is computed using the Hamrock–Dowson minimum film thickness equation. Regime classification: Λ > 3.0 → full-film EHL; 1.5–3.0 → mixed; 1.0–1.5 → partial boundary; < 1.0 → boundary lubrication.

**GUI output (Results panel — "Failure Assessment" tab):**

```
FAILURE LIKELIHOOD ASSESSMENT
System: Shaft S1 — 1450 rpm, 5 000 N·m

Failure Mode                  Score   Risk     Priority
─────────────────────────────────────────────────────────
RCF — Bearing 6210 (pos. A)   0.71    HIGH        1
Lubrication Starvation (A)    0.58    MEDIUM      2
Shaft Bending Fatigue         0.31    MEDIUM      3
Gear Tooth Pitting (z1)       0.28    MEDIUM      4
Static Overload (pos. B)      0.12    LOW         5
Shoulder Fatigue (§2)         0.09    LOW         6

⚠ HIGH risk — Bearing pos. A: C/P = 1.8 (target > 3.0).
  Review bearing selection or reduce applied load.
```

Clicking any row expands a driver breakdown table showing individual raw values, normalised scores, weights, and contributions — making the scoring logic fully transparent.

---

### Phase 1 — Core Solver Foundation *(months 1–3)*

**Goal:** A working, validated solver pipeline with no GUI dependency.

- [ ] Implement `Shaft`, `ShaftSection`, `MechanicalSystem`, `Bearing`, `GearElement`, `Load` data structures
- [ ] Implement `StaticsSolver`: reactions and V(x), M(x), T(x) diagrams for two-bearing shaft
- [ ] Implement `StressSolver`: von Mises, Goodman, stress concentrations at shoulders and keyways
- [ ] Implement `BearingLifeSolver`: ISO 281 L10 and static safety factor
- [ ] Implement `GEARpieAdapter`: import and validate GEARpie results
- [ ] Unit tests for all solvers against hand-calculated reference cases
- [ ] Validate against at least two textbook examples (Shigley, or equivalent)

**Deliverable:** CLI script demonstrating full analysis of a defined shaft system, printing results to terminal.

---

### Phase 2 — Base GUI *(months 3–5)*

**Goal:** A functional PySide6 interface allowing interactive system definition and result visualisation.

- [ ] Main window layout: canvas (centre), component panel (left), properties panel (right), results (bottom/tab)
- [ ] Shaft schematic canvas: rendered from `MechanicalSystem` state; supports zoom, click-to-select
- [ ] Component panel: `QTreeWidget` listing shaft sections, bearings, gears, loads
- [ ] Properties panel: `QFormLayout` driven by selected component type; editable fields with validation
- [ ] Results panel: tabbed view for shear/bending diagrams, bearing L10 table, stress summary
- [ ] Real-time update: solver called on model change; results refresh automatically
- [ ] GEARpie import wizard: file picker, field mapping, preview before insertion

**Deliverable:** Desktop application capable of defining a shaft system and viewing all Phase 1 analysis results graphically.

---

### Phase 3 — SKF Catalogue Integration + Failure Assessment Foundation *(months 5–7)*

**Goal:** Bearing selection from a structured, queryable SKF catalogue database; first failure mode evaluations active.

- [ ] Build SQLite schema for rolling bearing data: `designation`, `bore`, `OD`, `width`, `C`, `C0`, `speed_limit_grease`, `bearing_type`
- [ ] Populate database from publicly available SKF catalogue data (manual curation or structured CSV import)
- [ ] Implement `BearingSelector.query()`: filter by bore diameter, load rating requirement, speed
- [ ] Bearing selection dialog in GUI: displays ranked candidates, allows manual override
- [ ] Store selected bearing designation on `Bearing` object; recalculate L10 immediately
- [ ] Visual indicator in component panel: green/amber/red bearing health status icons
- [ ] Implement `FailureLikelihoodAssessor` framework: scoring infrastructure, `FailureModeResult`, `DriverContribution` dataclasses
- [ ] Activate Phase-2-available failure modes: F01 (RCF), F02 (Static Overload), F06 (Shaft Bending Fatigue), F07 (Shoulder Fatigue), F08 (Gear Tooth Pitting)
- [ ] "Failure Assessment" tab in results panel: ranked table with risk classes and driver breakdown on click

**Deliverable:** End-to-end workflow from system definition → analysis → bearing selection from catalogue → updated L10 results → first failure risk summary.

---

### Phase 4 — Design Recommendation Engine + Full Failure Assessment *(months 7–10)*

**Goal:** Complete rule-based intelligence layer covering arrangement, fits, lubrication, and all supported failure modes.

- [ ] Bearing arrangement advisor: fixed/floating, O-arrangement, X-arrangement selection logic based on axial load pattern and thermal expansion expectations
- [ ] Fit advisor: shaft and housing tolerance recommendations per ISO 286, conditioned on rotation and load type
- [ ] `LubricationAdvisor`: grease vs. oil selection (AGMA 9005 viscosity tables), re-lubrication interval (SKF model), NLGI grade recommendation
- [ ] Implement EHL film parameter Λ (Hamrock–Dowson); classify lubrication regime per bearing
- [ ] Activate remaining failure modes: F03 (Lubrication Starvation), F04 (Contamination Fatigue), F05 (False Brinelling), F09 (Gear Micropitting), F10 (Gear Scuffing), F11 (Misalignment Sensitivity)
- [ ] Warning and recommendation system: design checker flags safety factors below threshold, suggests corrective actions with engineering rationale
- [ ] Structured output in results panel: "Recommendations" tab with rationale text alongside numerical results
- [ ] Report generation: structured HTML/PDF calculation report including FLA summary table

**Deliverable:** System that moves from pure calculation output to integrated engineering guidance — covering bearing selection, arrangement, lubrication, and quantified risk per failure mode.

---

### Phase 5 — Expansion *(beyond month 10)*

The following expansions are architecturally anticipated but explicitly out of scope for initial development:

- **Multi-shaft gearbox assembly**: multiple `MechanicalSystem` instances connected through gear pairs
- **1D FEM deflection solver**: replace analytical beam solver for complex, multi-section shafts
- **Thermal model**: steady-state bearing temperature estimation; influence on lubrication viscosity and life correction factor
- **Fatigue life under variable loading**: Palmgren–Miner cumulative damage; load spectrum input
- **Export to standard formats**: shaft geometry export for FEA pre-processing (neutral format)
- **Parametric study runner**: vary one parameter across a range; plot result sensitivities

---

## Minimum Viable Product (MVP)

The MVP is the output of Phase 1 and Phase 2 combined. It is defined by strict boundary conditions.

### In scope for MVP

- Single horizontal shaft, two bearing supports (statically determinate)
- Shaft defined by up to 10 cylindrical sections with constant diameter per section
- Shoulder geometry at section transitions (fillet radius input)
- Placement of bearings, gear elements, and point loads/moments at specified axial positions
- GEARpie result import and force extraction
- Static analysis: bearing reactions, V(x), M(x), T(x)
- Stress analysis: combined bending + torsion at shoulder cross-sections; Goodman safety factor
- Bearing life: ISO 281 L10h for user-specified C and n (manual catalogue entry, not yet database-driven)
- PySide6 GUI: shaft schematic, component tree, properties panel, results with diagrams
- JSON-based project save/load

### Explicitly excluded from MVP

| Feature | Reason |
|---|---|
| Multi-bearing indeterminate systems (>2 supports) | Requires deflection solver and compatibility equations |
| SKF catalogue database | Phase 3 scope |
| Shaft deflection calculation | Phase 4 / FEM scope |
| Gear geometry design | Handled by GEARpie separately |
| Failure Likelihood Assessment | Phase 3 scope (framework) |
| Lubrication advisor and Λ parameter | Phase 4 scope |
| Critical speed analysis | Phase 5 |
| Report generation | Phase 3/4 |
| Thermal analysis | Phase 5 |
| Tapered or splined shaft sections | Phase 3 |

---

## Incremental Development Strategy

The architecture is designed to support isolated, incremental development without cross-module blocking.

**Principle 1 — Solvers are independent of the GUI.**
All solvers accept data model objects and return result objects. They can be developed, tested, and validated entirely via `pytest` before any GUI work begins. The GUI is a consumer of solver outputs, not a participant in computation.

**Principle 2 — Data model is the integration boundary.**
`core/` defines the domain entities that all other modules depend on. Stabilising the data model early (Phase 1) prevents downstream refactoring. Changes to solvers, GUI, or database do not propagate back to `core/` unless a domain concept changes.

**Principle 3 — Each component position is a single float.**
All axial positioning is by coordinate, not by section index. This decouples component placement from shaft discretisation and makes interpolation, plotting, and solver sampling straightforward.

**Principle 4 — GEARpie integration is encapsulated.**
The `GEARpieAdapter` is the only module that knows GEARpie's output format. If GEARpie's output schema changes, only the adapter changes.

**Principle 5 — Database is optional until Phase 3.**
Bearings in Phase 1–2 can be defined with manually entered C and C0 values. The database is a query optimisation, not a structural dependency.

**Development order per session:**
1. Write the data structure
2. Write the solver logic
3. Write the unit test with a hand-calculated reference
4. Only then connect to GUI

---

## Quality Criteria

### Modularility
Each module has one clearly stated responsibility. Imports flow inward (`ui` → `core`, `solvers` → `core`). No circular dependencies. `core/` has zero external library dependencies beyond Python standard library and NumPy.

### Testability
Every solver function is a pure function: given the same input objects, it returns the same result. No global state, no GUI calls inside solvers. Test coverage target: ≥ 90% for `solvers/` and `core/`. Tests must include at least one textbook-validated reference case per solver.

### Physical Validity
All calculations include inline references to the governing standard or equation (ISO 281, ISO 6336, Shigley §n.m, etc.) in docstrings. Any simplifying assumption is documented explicitly. Results outside physically plausible ranges trigger warnings, not silent acceptance.

### Scalability
Adding a new solver (e.g., critical speed) requires: one new file in `solvers/`, one new result dataclass, one new tab in the results panel. No changes to existing solvers, data model, or tests.

### Numerical robustness
Solvers use NumPy throughout. Shaft discretisation uses a fixed resolution (default 1000 points/m, configurable in `config.py`). Division-by-zero and degenerate geometry cases (zero-length section, zero-diameter section) are caught at `validate()` time, before solvers are called.

---

## Installation

```bash
git clone https://github.com/<user>/axisforge.git
cd axisforge
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

**Runtime dependencies:**

| Package | Version | Purpose |
|---|---|---|
| `PySide6` | ≥ 6.6 | GUI framework |
| `numpy` | ≥ 1.26 | Numerical computation |
| `matplotlib` | ≥ 3.8 | Diagram rendering |
| `scipy` | ≥ 1.12 | Integration, interpolation |
| `pandas` | ≥ 2.1 | Data handling (catalogue queries) |
| `weasyprint` | ≥ 61 | PDF report output (Phase 3+) |

**Development dependencies** (in `requirements-dev.txt`):

```
pytest
pytest-cov
ruff
mypy
```

---

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=axisforge --cov-report=term-missing
```

Reference validation cases are defined in `tests/fixtures/sample_systems.py`. Each fixture documents its source (textbook, chapter, example number) and the expected result values against which the solver is compared.

---

## GEARpie Integration

GEARpie operates as an independent module. AxisForge does not call GEARpie functions directly; it reads GEARpie output files via `GEARpieAdapter`. This ensures that changes to GEARpie's internal implementation do not affect AxisForge stability.

The expected GEARpie export format (JSON) is documented in `docs/api/gearpie_interface.md`. If GEARpie's output format changes, only `integrations/gearpie_adapter.py` requires modification.

---

## Project Status

| Phase | Status |
|---|---|
| Phase 1 — Core Solver | 🔲 In development |
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
- SKF — *Bearing Maintenance Handbook*
- Peterson, R.E. — *Stress Concentration Factors*, 3rd ed.
- Harris, T.A., Kotzalas, M.N. — *Rolling Bearing Analysis*, 5th ed.
- Stachowiak, G.W., Batchelor, A.W. — *Engineering Tribology*, 4th ed.
- Sadeghi, F. et al. — *A Review of Rolling Contact Fatigue*, ASME J. Tribology, 2009