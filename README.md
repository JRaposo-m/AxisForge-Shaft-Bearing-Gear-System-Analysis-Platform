# AxisForge — Shaft–Bearing–Gear System Analysis Platform

AxisForge is an open-source, modular Python platform for the static, structural, and tribological analysis of shaft–bearing–gear mechanical systems. It prioritises transparency of calculation, modular extensibility, and traceable decision support over comprehensive feature breadth.

Gear geometry and force calculation is implemented natively according to ISO 21771. The GEARpie reference cases C14 and H501 are used exclusively for validation — AxisForge has no runtime dependency on GEARpie.

---

## Axis Convention

```
x   = axial axis; datum = left end of shaft; increases rightward
XZ  = horizontal plane  (Wt gear force, lateral loads)
XY  = vertical plane    (Wr gear force, opposed direction of gravity)

Positive radial load: Downard (XY), forward (XZ)
      - The loads are inputed as positive however in the statics.py they are inported as -l.magnitude and are treated as the opposite direction for the rest of the class
Reactions:  sign determined by equilibrium equations (not forced positive)
Torsion T:  accumulates left-to-right; positive = CCW when viewed from +x
```

All internal arrays use this convention. Documented in `models/statics_result.py`. Must not be changed without updating all solvers and tests.

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
│   └── tolerances.py              # ISO 286 fit and tolerance data structures — Phase 3
│
├── models/                        # Result dataclasses — output of solvers
│   ├── statics_result.py          # StaticsResult
│   ├── stress_result.py           # StressResult, CriticalSection
│   ├── bearing_result.py          # BearingLifeResult
│   ├── gear_result.py             # GearGeometryResult, GearForceResult, GearStrengthResult
│   └── system_report.py           # Aggregator of all results
│
├── solvers/                       # Analytical engines — organised by mechanical element
│   ├── __init__.py                # Public interface — all solvers importable from here
│   │
│   ├── shaft/                     # Shaft analysis solvers
│   │   ├── __init__.py
│   │   ├── utils.py               # ka, kb, kc, kd, ke, kt, neuber, kf helpers — stress helpers internos
│   │   ├── statics.py             # StaticsSolver — reactions, V(x), M(x), T(x)
│   │   ├── static_failure.py      # StaticFailureSolver — Von Mises (DE), Tresca (MSS)
│   │   ├── stress.py              # StressSolver — Kf/Kfs, Marin Se', Goodman, ASME-Elliptic
│   │   ├── deflection.py          # DeflectionSolver — FEM-1D — Phase 4
│   │   └── critical_speed.py      # CriticalSpeedSolver — Phase 5
│   │
│   ├── bearings/                  # Bearing analysis solvers
│   │   ├── __init__.py
│   │   ├── life.py                # BearingLifeSolver — L10h, C/P, S0 (ISO 281)
│   │   ├── friction.py            # FrictionSolver — M_rr, M_sl, M_seal, M_drag, P_loss, T_steady (SKF friction model)
│   │   ├── arrangement.py         # ArrangementSolver — fixed/floating, O/X logic — Phase 4
│   │   └── misalignment.py        # MisalignmentSolver — bearing angle from FEM deflection — Phase 4
│   │
│   ├── gears/                     # Gear analysis solvers
│   │   ├── __init__.py
│   │   ├── utils.py               # involute, solve_alpha_tw, contact_ratio_*, undercut_z_min, HAP, HFP
│   │   ├── geometry.py            # GearSolver — MAAG geometry, Ft/Fr/Fa (ISO 21771)
│   │   ├── strength.py            # GearStrengthSolver — σ_H, σ_F, KA/KV/KHβ/KHα (ISO 6336-2/3) — Phase 2
│   │   ├── profile_shift.py       # ProfileShiftSolver — optimal x1/x2 calculation — Phase 2
│   │   └── micropitting.py        # MicropittingSolver — Phase 4
│   │
│   └── lubrication/               # Lubrication and film solvers
│       ├── __init__.py
│       ├── film.py                # FilmSolver — Hamrock–Dowson h_min, Λ parameter — Phase 4
│       ├── regime.py              # RegimeSolver — EHL / mixed / boundary classification — Phase 4
│       ├── grease.py              # GreaseSolver — NLGI selection, relubrication interval — Phase 4
│       └── cfd_interface.py       # CFDInterface — OpenFOAM case generation and result import — Phase 5
│
├── integrations/
│   ├── skf_connector.py           # SKF catalogue query — Phase 3
│   └── openfoam/                  # OpenFOAM bridge — Phase 5
│       ├── case_generator.py      # AxisForge system → OpenFOAM case structure
│       ├── mesh_builder.py        # Hertz contact geometry → polyMesh
│       └── results_reader.py      # postProcessing/ → Λ_CFD, T_max, h_min
│
├── database/                      # Phase 3
│   ├── skf_bearings.db
│   ├── materials.db
│   └── schema/
│       ├── bearings_schema.sql
│       └── materials_schema.sql
│
├── selectors/                     # Decision-support logic — Phase 3–5
│   ├── bearing_selector.py        # Filter catalogue by load, speed, space
│   ├── arrangement_advisor.py     # Fixed/floating, O/X arrangement — Phase 4
│   ├── lubrication_advisor.py     # Grease/oil selection, viscosity grade — Phase 4
│   └── failure_assessor.py        # Failure Likelihood Assessment — Phase 3+
│
├── ui/                            # PySide6 graphical interface — Phase 2
│   ├── main_window.py
│   ├── shaft_canvas.py
│   ├── component_panel.py
│   ├── properties_panel.py
│   ├── results_panel.py
│   └── dialogs/
│       ├── bearing_selector.py
│       └── section_editor.py
│
├── reports/                       # Phase 3
│   ├── report_generator.py
│   └── templates/
│       └── standard_report.html
│
├── tests/
│   ├── conftest.py
│   ├── test_core/
│   ├── test_solvers/
│   │   ├── test_shaft/
│   │   │   ├── test_statics.py
│   │   │   ├── test_static_failure.py
│   │   │   └── test_stress.py
│   │   ├── test_bearings/
│   │   │   └── test_life.py
│   │   ├── test_gears/
│   │   │   ├── test_geometry.py       # C14 and H501 validation cases
│   │   │   └── test_strength.py
│   │   └── test_lubrication/
│   ├── test_integrations/
│   └── fixtures/
│       ├── sample_systems.py
│       └── expected_results.py        # Reference values with source citations
│
├── axisforge_cli.py               # CLI entry point (Phase 1 deliverable)
├── config.py
├── requirements.txt
└── pyproject.toml
```

### Import Dependency Rule

```
ui/ → solvers/ → core/
      models/  ← solvers/
```

`core/` imports NumPy and stdlib only. No PySide6, matplotlib, or database calls inside `core/` or `solvers/`. The `solvers/__init__.py` exposes the public interface — the rest of the application imports `from solvers import StaticsSolver`, not `from solvers.shaft.statics import StaticsSolver`. Internal reorganisation never propagates outward.

---

## Implementation Status

| Module | Status | Notes |
|---|---|---|
| `core/shaft.py` | ✅ Complete | 209/209 tests, 98% coverage |
| `core/components.py` | ✅ Complete | — |
| `core/loads.py` | ✅ Complete | — |
| `core/materials.py` | ✅ Complete | — |
| `core/system.py` | ✅ Complete | — |
| `models/statics_result.py` | ✅ Complete | — |
| `solvers/shaft/statics.py` | ✅ Complete | 296/296 tests, ≥90% coverage |
| `solvers/shaft/static_failure.py` | ✅ Complete | Von Mises (DE), Tresca (MSS) |
| `solvers/shaft/stress.py` | 🟡 In progress | Auxiliary functions done; `solve()` next |
| `solvers/gears/geometry.py` | ✅ Complete | ISO 21771; validated C14 <0.00%, H501 <0.5% |
| `solvers/bearings/life.py` | 🔲 Phase 1 Week 8 | — |
| `solvers/gears/strength.py` | 🔲 Phase 2 | ISO 6336-2/3 |
| `solvers/gears/profile_shift.py` | 🔲 Phase 2 | Optimal x1/x2 |
| `solvers/bearings/friction.py` | 🔲 Phase 3 | SKF friction model |
| `solvers/shaft/deflection.py` | 🔲 Phase 4 | FEM-1D |
| `solvers/bearings/misalignment.py` | 🔲 Phase 4 | Requires deflection |
| `solvers/lubrication/` | 🔲 Phase 4 | Full lubrication module |
| `integrations/openfoam/` | 🔲 Phase 5 | CFD bridge |
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
| `material_id` | str | Key into `core/materials.py` |
| `surface_finish_ra` | float | Ra [μm]; used for Marin factor kₐ |
| `shoulder_left` | `Shoulder \| None` | Left-side shoulder transition |
| `shoulder_right` | `Shoulder \| None` | Right-side shoulder transition |
| `label` | str | Optional identifier |

**`Shaft`** — ordered collection of `ShaftSection` objects.

| Method / Property | Returns | Description |
|---|---|---|
| `add_section(section)` | None | Appends section to right end |
| `total_length` | float | Sum of section lengths [mm] |
| `axial_start(index)` | float | Absolute start of section[index] [mm] |
| `axial_end(index)` | float | Absolute end of section[index] [mm] |
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

Embedded library: `S355`, `CrMo42`, `AISI_1045`, `AISI_4340`.

---

### `solvers/shaft/statics.py`

**`StaticsSolver`** — two-support shaft, statically determinate. Planes solved independently.

| Method | Returns | Description |
|---|---|---|
| `solve(system)` | `StaticsResult` | Reactions + discretised V(x), M(x), T(x) |

`StaticsResult` fields: `x`, `V_xz`, `V_xy`, `M_xz`, `M_xy`, `M_res`, `T`, `axial_force`, `reactions`.
Reaction keys: `"A_xz"`, `"A_xy"`, `"B_xz"`, `"B_xy"`, `"axial"`.

---

### `solvers/shaft/stress.py`

**`StressSolver`** — critical section identification, Kf/Kfs, Marin Se', fatigue criteria.

| Method | Returns | Description |
|---|---|---|
| `solve(system, statics_result, material)` | `StressResult` | Full fatigue analysis at all shoulder sections |

`StressResult`: list of `CriticalSection` sorted by `nf_goodman` ascending. Each exposes: `x`, `Kf`, `Kfs`, `sigma_a`, `tau_m`, `Se_prime`, `nf_goodman`, `nf_asme`, `von_mises_eq`.

Phase 1 assumptions: `kc = 1.0`; D/d = 1.5 fixed for Peterson. Phase 2: Peterson 2D interpolation via `scipy.interpolate.RegularGridInterpolator`.

---

### `solvers/gears/geometry.py`

**`GearSolver`** — MAAG gear geometry and force calculation (ISO 21771). Native implementation; no external dependency.

| Method | Returns | Description |
|---|---|---|
| `compute_geometry(mn, z1, z2, alpha_n_deg, beta_deg, al, x1, x2, b)` | `GearGeometryResult` | Full MAAG geometry including αtw, εα, εβ, all diameters |
| `compute_forces(T1_Nm, geometry)` | `GearForceResult` | Ft, Fr, Fa in Newtons; T1, T2 in N·mm |

Validated against GEARpie reference cases: C14 (spur, x≠0) desvio = 0.00%; H501 (helical, x≠0) desvio < 0.5%.

---

### `solvers/gears/strength.py` *(Phase 2)*

**`GearStrengthSolver`** — ISO 6336-2/3 contact and bending stress.

| Method | Returns | Description |
|---|---|---|
| `solve(geometry, forces, material_pinion, material_wheel, load_factors)` | `GearStrengthResult` | σ_H, σ_F, safety factors SH and SF |

Load factors: KA (application), KV (dynamic), KHβ (load distribution), KHα (transverse load distribution).

---

### `solvers/bearings/life.py`

**`BearingLifeSolver`** — ISO 281 L10 life and static safety.

| Method | Returns | Description |
|---|---|---|
| `solve_bearing(bearing, Fr, Fa, speed_rpm, design_life_hours)` | `BearingLifeResult` | L10h, S₀, C/P, pass/fail |
| `extract_bearing_forces(statics_result, system)` | `dict` | Fr and Fa per bearing label |

Phase 1: X=1, Y=0. Phase 2: full ISO 281 Table 1 X/Y factors; aISO life modification factor.

---

### `solvers/lubrication/film.py` *(Phase 4)*

**`FilmSolver`** — EHL minimum film thickness and lambda ratio.

| Method | Returns | Description |
|---|---|---|
| `solve(contact_geometry, lubricant, operating_conditions)` | `FilmResult` | h_min (Hamrock–Dowson), Λ = h_min / R_composite |
| `classify_regime(lambda_ratio)` | str | `"full_film"` / `"mixed"` / `"partial_boundary"` / `"boundary"` |

Regime thresholds: Λ > 3.0 → full-film EHL; 1.5–3.0 → mixed; 1.0–1.5 → partial boundary; < 1.0 → boundary lubrication.

Phase 5: `FilmSolver` replaced or supplemented by `cfd_interface.py` for CFD-computed h_min.

---

### `integrations/openfoam/` *(Phase 5)*

**`CaseGenerator`** — generates complete OpenFOAM case structure from AxisForge system state.

| Method | Returns | Description |
|---|---|---|
| `generate(contact_geometry, lubricant, conditions, output_path)` | `OpenFOAMCase` | Writes `0/`, `constant/`, `system/` directories |

The generated case includes: velocity boundary conditions from surface speeds, transport properties from lubricant selection, Hertz contact geometry as polyMesh. The engineer does not interact with OpenFOAM directly.

**`ResultsReader`** — imports OpenFOAM postProcessing output back into AxisForge.

| Method | Returns | Description |
|---|---|---|
| `read(case_path)` | `CFDFilmResult` | Extracts h_min, T_max, p_max from field data |

`CFDFilmResult` feeds directly into `FilmSolver` to replace the analytical Λ with the CFD-computed value, which then updates the Failure Likelihood Assessment.

---

### `selectors/failure_assessor.py`

**`FailureLikelihoodAssessor`** — deterministic multi-criteria scoring engine.

Scoring model per failure mode Fᵢ:

```
score(Fᵢ) = Σ [ wⱼ × fⱼ(xⱼ) ]   for j = 1..N_drivers
```

Risk classification:

| Score | Class |
|---|---|
| < 0.25 | LOW |
| 0.25 – 0.60 | MEDIUM |
| > 0.60 | HIGH |

Supported failure modes by phase:

| ID | Mode | Component | Phase |
|---|---|---|---|
| F01 | Rolling Contact Fatigue | Bearing | 3 |
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

---

## Incremental Development Principles

1. **Solvers are GUI-independent.** Fully testable before any Phase 2 work.
2. **`core/` is the integration boundary.** Stable since Week 2. Changes here propagate everywhere.
3. **Component positions are floats** (mm from datum), not section indices.
4. **`solvers/__init__.py` is the public interface.** Internal solver paths are implementation detail.
5. **Database not used in Phase 1–2.** C and C0 entered manually on `Bearing`.
6. **Development order per solver:** data structure → test with reference value (TDD) → implement → pass.
7. **Each solver sub-package is independently extensible.** Adding a new shaft solver requires one new file in `solvers/shaft/` — no changes to bearings, gears, or lubrication.

---

## config.py Constants

| Constant | Value | Purpose |
|---|---|---|
| `SOLVER_RESOLUTION` | 1000 | Array length for discretised diagrams |
| `BOUNDARY_MOMENT_TOLERANCE` | 500.0 N·mm | Max \|M\| at supports (validation check) |
| `SOLVER_TOLERANCE` | 1e-6 | General numerical guard |
| `DEFAULT_DESIGN_LIFE_HOURS` | 20 000 h | Bearing life default |
| `SUT_ENDURANCE_CAP_MPa` | 1400 MPa | Se = 0.5×Sut valid below this (Shigley §6-2) |
| `RELIABILITY_FACTOR_99` | 0.868 | ke at 99% reliability (Shigley Tab. 6-6) |
| `MIN_FILLET_RADIUS_mm` | 0.01 mm | Guard for notch sensitivity calculation |
| `FLA_THRESHOLD_LOW` | 0.25 | Risk classification boundary |
| `FLA_THRESHOLD_HIGH` | 0.60 | Risk classification boundary |
| `EHL_LAMBDA_FULL_FILM` | 3.0 | Λ above which full-film EHL is assumed |
| `EHL_LAMBDA_BOUNDARY` | 1.0 | Λ below which boundary lubrication is assumed |

---

## Quality Criteria

**Modularity:** Imports flow inward. No circular dependencies. `core/` has zero external library dependencies beyond NumPy and stdlib. Each solver sub-package has one clearly stated responsibility.

**Testability:** Every solver is a pure function. No global state, no GUI calls inside solvers. Coverage target ≥ 90% for `solvers/` and `core/`. At least one textbook-validated reference case per solver. Tests mirror the solver directory structure.

**Physical validity:** Governing equation or standard referenced in every solver docstring. Simplifying assumptions documented explicitly. Results outside physically plausible ranges raise warnings before solvers run.

**Extensibility:** Adding a new solver requires one new file in the appropriate sub-package. No changes to existing solvers, data model, or tests outside that sub-package.

**Numerical robustness:** NumPy throughout. Degenerate geometry caught at `validate_or_raise()` before any solver is called.

---

## Installation

```bash
git clone https://github.com/<user>/axisforge.git
cd axisforge
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python axisforge_cli.py
```

**Runtime dependencies:**

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥ 1.26 | Numerical computation |
| `scipy` | ≥ 1.12 | Integration, interpolation, Peterson 2D |
| `PySide6` | ≥ 6.6 | GUI framework — Phase 2+ |
| `matplotlib` | ≥ 3.8 | Diagram rendering — Phase 2+ |
| `pandas` | ≥ 2.1 | Catalogue queries — Phase 3+ |
| `weasyprint` | ≥ 61 | PDF report output — Phase 3+ |

**Dev dependencies:** `pytest`, `pytest-cov`, `ruff`, `mypy`

---

## Testing

```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=core --cov=solvers --cov-report=term-missing

# Run a specific validation case
python -m pytest tests/test_solvers/test_gears/test_geometry.py::TestGearSolverValidation::test_c14_forces -v
```

Validation cases in `tests/fixtures/expected_results.py` include source, chapter or standard reference, and expected values with tolerances. Current: 296 tests passing. Coverage: `core/` 98%, `solvers/shaft/statics.py` ≥ 90%.

---

## GEARpie Validation

The GEARpie reference cases C14 (spur, x≠0) and H501 (helical, x≠0) are used to validate `solvers/gears/geometry.py`. AxisForge does not call GEARpie at runtime — all gear geometry and force calculation is implemented natively according to ISO 21771. Validation results: Ft desvio = 0.00% (C14), < 0.5% (H501).

---

## MVP Scope (Phase 1 + Phase 2)

### In scope

- Single horizontal shaft, two bearing supports (statically determinate)
- Shaft defined by up to 10 cylindrical sections
- Shoulder geometry at section transitions
- Bearings, gear elements, and point loads at specified axial positions
- Gear geometry and forces from native `GearSolver` (ISO 21771)
- Static analysis: reactions, V(x), M(x), T(x)
- Static failure: Von Mises (DE), Tresca (MSS)
- Fatigue: combined bending + torsion at shoulder sections; Goodman, ASME-Elliptic
- Bearing life: ISO 281 L10h with full X/Y factors (Phase 2)
- PySide6 GUI with shaft schematic, component tree, results with diagrams
- JSON project save/load

### Explicitly excluded from MVP

| Feature | Reason |
|---|---|
| Multi-bearing indeterminate systems | Requires deflection solver — Phase 4 |
| SKF catalogue database | Phase 3 |
| Shaft deflection (FEM-1D) | Phase 4 |
| ISO 6336 gear strength | Phase 2 |
| Failure Likelihood Assessment | Phase 3 |
| Lubrication module | Phase 4 |
| CFD integration | Phase 5 |
| Critical speed | Phase 5 |
| Report generation | Phase 3 |

---

## Licence

MIT Licence.

---

## References

- ISO 21771:2007 — *Gears — Cylindrical involute gears and gear pairs — Concepts and geometry*
- ISO 281:2007 — *Rolling bearings — Dynamic load ratings and rating life*
- ISO 76:2006 — *Rolling bearings — Static load ratings*
- ISO 6336-1/2/3:2019 — *Calculation of load capacity of spur and helical gears*
- ISO 286-1:2010 — *Geometrical product specifications — Limits and fits*
- Shigley, J.E., Budynas, R.G., Nisbett, J.K. — *Mechanical Engineering Design*, 10th ed.
- Peterson, R.E. — *Stress Concentration Factors*, 3rd ed.
- SKF General Catalogue — *Rolling Bearings*, publication 10000 EN
- Harris, T.A., Kotzalas, M.N. — *Rolling Bearing Analysis*, 5th ed.
- Stachowiak, G.W., Batchelor, A.W. — *Engineering Tribology*, 4th ed.
- Hamrock, B.J., Dowson, D. — *Ball Bearing Lubrication*, Wiley, 1981
- Sadeghi, F. et al. — *A Review of Rolling Contact Fatigue*, ASME J. Tribology, 2009
- SKF — *The SKF Model for Calculating the Frictional Moment*, publication 2013