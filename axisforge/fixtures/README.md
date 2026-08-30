# AxisForge — Fixtures

Modular template library for rapid construction of analysis and validation scripts. These files are **not pytest unit tests**. They are reusable building blocks for assembling `design_xxx.py` scripts — functional analysis, pipeline verification, and textbook validation runs.

Each fixture layer depends only on the layers below it. The dependency chain runs from core classes upward through gears / shafts / bearings to systems, solvers, outputs, plots, and finally integration scripts.

← back to [project root](../../README.md)

---

## Table of Contents

- [Status](#status)
- [Directory structure](#directory-structure)
- [Fixture dependency chain](#fixture-dependency-chain)
- [`capabilities/` — the capability selector](#capabilities--the-capability-selector)
- [Implemented fixtures](#implemented-fixtures)
- [Key design contracts](#key-design-contracts)
- [Usage rules](#usage-rules)

---

## Status

| Layer | File | Status |
|---|---|---|
| `gears/` | `spur_helical.py` | Implemented and validated |
| `shafts/` | `shaft_fixture.py` | Implemented and validated |
| `bearings/` | `dgbb_generic.py` | Implemented — **stale**, still on the pre-`assemble()` bearing API (see below) |
| `systems/` | `linear_gear_chain.py` | Implemented and validated |
| `capabilities/` | `catalogue.py`, `selection.py`, `design_capabilities.py` | **Specified here, not yet implemented** |
| `solvers/` | `fem_simple.py`, `iso16281_coupled.py`, `static_analysis.py` | Planned |
| `outputs/` | all | Planned |
| `plots/` | all | Planned |
| `convergence/` | `mesh_gci_study.py` | Planned |
| `integration/` | all | Planned |

> **`bearings/dgbb_generic.py` is out of date.** `DGBBFixture.make_ready()` still constructs a `DeepGrooveBallBearing` and calls `setup_internal_geometry()` / `compute_hertz_point_contact()` — the retired concrete-subclass architecture. The current path is `Bearing.assemble(family=DeepGrooveBallFamily(), catalog=BearingCatalog(...), geometry=dict(...), analyses={"point_contact": True})`. Migrating this fixture is a prerequisite for `capabilities/`, because the selector resolves bearings through `families/`, not through the legacy subtypes. The note in `core/machine_elements/bearings/__init__.py` tracks the same migration.

---

## Directory structure

```
fixtures/
|
|-- capabilities/                   (specified, not implemented)
|   |-- __init__.py                 Capabilities, CapabilitySet, catalogue()
|   |-- catalogue.py                reads the package roll-ups -> the menu
|   |-- selection.py                the Capabilities dataclass + resolve()
|   +-- design_capabilities.py      the file the user edits and imports
|
|-- gears/
|   +-- spur_helical.py
|
|-- shafts/
|   +-- shaft_fixture.py
|
|-- bearings/
|   |-- dgbb_generic.py             (needs migration to Bearing.assemble)
|   |-- crb_generic.py              (reserved)
|   +-- angular_contact.py          (reserved)
|
|-- systems/
|   +-- linear_gear_chain.py
|
|-- solvers/
|   |-- fem_simple.py               (planned)
|   |-- iso16281_coupled.py         (planned)
|   +-- static_analysis.py          (planned)
|
|-- outputs/
|   |-- console_bearing.py          (planned)
|   |-- console_system_header.py    (planned)
|   +-- console_shaft_summary.py    (planned)
|
|-- plots/
|   |-- deflection_3panel.py        (planned)
|   |-- bearing_polar.py            (planned)
|   |-- convergence_gci.py          (planned)
|   +-- deflection_global.py        (planned)
|
|-- convergence/
|   +-- mesh_gci_study.py           (planned)
|
+-- integration/
    |-- load_distribution_full.py   (planned)
    +-- convergence_study_full.py   (planned)
```

---

## Fixture dependency chain

```
integration/                         <- full pipeline entry points
    ^
outputs/ + plots/                    <- console and matplotlib output
    ^
solvers/                             <- FEM, ISO 16281, static analysis
    ^
systems/linear_gear_chain.py         <- assembles ShaftSystems, resolves power flow
    ^              ^              ^
gears/         shafts/          bearings/
    ^
capabilities/                        <- decides WHICH of the above get imported
    ^
axisforge package __init__.py        <- the roll-up tables capabilities/ reads
    ^
axisforge core classes               <- never modified here
```

`capabilities/` sits **below** every other fixture layer and **above** the package roll-ups. It reads; it is never read from. No `__init__.py` anywhere in `axisforge/` imports anything from `fixtures/`.

---

## `capabilities/` — the capability selector

### Purpose

A `design_xxx.py` script currently opens with twenty to forty import lines, hand-written, each spelling out a full dotted path that the reorganisation just changed. The point of `capabilities/` is that the script declares **what it wants to do**, once, and gets back exactly the objects that job needs — nothing else imported, nothing else in the namespace.

This is only possible because of the `__init__.py` roll-up work: every package now publishes `__all__` (what it offers) and `_LAZY` (where each name lives) as plain data, and imports nothing until a name is touched. `capabilities/` reads those two tables. See the [root README](../../README.md#package-surface--the-__init__py-roll-up-system) for the mechanism.

### Direction of dependency — non-negotiable

```
design_xxx.py  →  fixtures/capabilities/  →  package __init__.py  →  leaf module
```

Never the reverse. `families/__init__.py` states it in its own docstring: *"É a partir daqui que `bearing.py` é capaz de listar/selecionar famílias, e mais tarde de onde um `capabilities.py` para bearings vai ler — nunca ao contrário."* If a roll-up ever imported from `capabilities/`, the lazy import graph would become circular and the whole scheme collapses.

### The domain map

`catalogue.py` owns one table: domain key → roll-up package. That table is the *only* place a dotted path is written down; everything else is derived.

| Domain key | Roll-up package | Offers |
|---|---|---|
| `shaft` | `axisforge.core.machine_elements.shaft` | `Shaft`, `ShaftSection`, `Shoulder`, `Keyway`, `KeywayType` |
| `bearings` | `axisforge.core.machine_elements.bearings` | `Bearing`, `BearingCatalog`, `BearingFamily`, `BearingType` |
| `bearing_families` | `...bearings.families` | the concrete `*Family` classes |
| `gears` | `axisforge.core.machine_elements.gears.parallel_axis` | `SpurHelicalGear`, `InternalGear`, `SpurHelicalGearMeshing`, `InternalGearMeshing`, `PlanetaryGearTrainMeshing` |
| `systems` | `axisforge.core.mechanical_system.parallel_axis.spur_helical` | `ShaftSystem`, `GearElement`, `SpurHelicalMeshLink`, `SpurHelicalGearSystem` |
| `loads` | `axisforge.core.loads` | `RadialLoad`, `AxialLoad`, `TorqueLoad`, `ExternalMoment`, `DistributedRadialLoad`, ... |
| `materials` | `axisforge.core.materials` | `get_material`, `get_gear_material`, ... |
| `mesh` | `axisforge.mesh.shaft.mesh_generation` | `Mesh1D`, `Grader` |
| `elements` | `axisforge.mesh.shaft.element_type` | `TimoshenkoBeam` *(and `Elem`, once rolled up)* |
| `fem` | `...shaft.oneD_analysis.FEM_solvers` | `SimpleFEMSolver`, `SubmodelSolver`, `SubmodelResult` |
| `static` | `...shaft.oneD_analysis.static` | `ShaftResults`, `ShaftResultsReader`, `SimpleFEMResultsLibrary`, `BearingNodeData` |
| `bearing_solvers` | `...bearings.ISO_16281` | `RollingBearingSolver`, `BearingResultsLibrary`, `resolve_solver_cls`, ... |
| `ball` | `...ISO_16281.Ball_Bearing` | `ISO16281BallSolver`, `contact_distribution`, `bearing_stiffness`, ... |
| `roller` | `...ISO_16281.Roller_Bearing` | `ISO16281RollerSolver`, `lamina_distribution`, `stress_riser_factor`, ... |
| `gear_solvers` | `axisforge.solvers.machine_elements.gears` | `GearSolver` |
| `convergence` | `axisforge.solvers.mesh` | `MeshConvergenceStudy`, `RichardsonGCI`, ... |
| `schematic` | `axisforge.core.mechanical_system.parallel_axis.schematic` | `draw_gear_system`, `draw_shaft_detail`, `recommended_figsize` |

Adding a domain = adding one row. Adding a class inside an existing domain = **nothing**, because the menu is `__all__`, read at call time.

### Proposed API

```python
# fixtures/capabilities/catalogue.py

DOMAINS: dict[str, str]                      # the table above

def catalogue() -> dict[str, list[str]]:
    """{domain: sorted(__all__)} — imports only the __init__.py of each
    roll-up, so nothing heavy is executed. Safe to call at import time."""

def analyses() -> dict[str, list[str]]:
    """{analysis_name: [family names declaring it]} — built from each
    BearingFamily's CAPABILITIES. Resolving a family class is unavoidable
    here (CAPABILITIES is a class attribute), so this touches the family
    modules and nothing below them."""

def where(name: str) -> tuple[str, str]:
    """(domain, dotted module path) for one exported name, read from the
    roll-up's _LAZY chain. Diagnostics only — never used to import."""

def print_menu() -> None:
    """ASCII table of everything selectable. Windows-cp1252 safe."""
```

```python
# fixtures/capabilities/selection.py

@dataclass(frozen=True)
class Capabilities:
    shaft:            tuple[str, ...] = ()
    bearings:         tuple[str, ...] = ()
    bearing_families: tuple[str, ...] = ()
    gears:            tuple[str, ...] = ()
    systems:          tuple[str, ...] = ()
    loads:            tuple[str, ...] = ()
    materials:        tuple[str, ...] = ()
    mesh:             tuple[str, ...] = ()
    elements:         tuple[str, ...] = ()
    fem:              tuple[str, ...] = ()
    static:           tuple[str, ...] = ()
    bearing_solvers:  tuple[str, ...] = ()
    ball:             tuple[str, ...] = ()
    roller:           tuple[str, ...] = ()
    gear_solvers:     tuple[str, ...] = ()
    convergence:      tuple[str, ...] = ()
    schematic:        tuple[str, ...] = ()
    analyses:         tuple[str, ...] = ()   # e.g. ("point_contact", "line_contact")

    def validate(self) -> list[str]:
        """Every requested name checked against catalogue(); every requested
        analysis against analyses(). Returns error strings, never raises —
        the AxisForge convention."""

    def validate_or_raise(self) -> None: ...

    def resolve(self) -> dict[str, object]:
        """Import exactly the selected names, via their roll-ups. Returns
        {name: object}. Raises on a name collision across domains rather
        than silently shadowing."""

    def summary(self) -> str:
        """What was selected, what it pulled in, per domain. ASCII only."""
```

### How a script uses it

```python
# design_two_stage_gearbox.py
from axisforge.fixtures.capabilities import Capabilities

CAPS = Capabilities(
    shaft            = ("Shaft", "ShaftSection", "Shoulder"),
    bearings         = ("Bearing", "BearingCatalog"),
    bearing_families = ("DeepGrooveBallFamily", "CylindricalRollerFamily"),
    gears            = ("SpurHelicalGear", "SpurHelicalGearMeshing"),
    systems          = ("ShaftSystem", "GearElement",
                        "SpurHelicalMeshLink", "SpurHelicalGearSystem"),
    loads            = ("TorqueLoad", "RadialLoad"),
    fem              = ("SimpleFEMSolver",),
    static           = ("ShaftResultsReader", "SimpleFEMResultsLibrary"),
    bearing_solvers  = ("RollingBearingSolver",),
    ball             = ("contact_distribution",),
    analyses         = ("point_contact", "line_contact"),
)

CAPS.validate_or_raise()
globals().update(CAPS.resolve())

# ---- PARAMETERS -------------------------------------------------------
...
```

Nothing on the roller postprocessing side, nothing planetary, nothing in `ui/` is imported. Change one line in the `Capabilities(...)` block and the import surface changes with it.

### Why `analyses` is part of the selection

`analyses` is not decoration. It is the same string set that

- `Bearing.assemble(..., analyses={"point_contact": True})` validates against `family.CAPABILITIES`, and
- `ISO_16281/dispatch.py` matches to pick a solver.

Declaring it in one place lets the fixture layer catch "you selected `CylindricalRollerFamily` but only enabled `point_contact`" **before** the script builds a single object — instead of a `NotImplementedError` from `assemble()` or a `SolverDispatchError` several hundred lines in.

### Prerequisites before implementing

1. Fix the roll-up defects listed in the [root README](../../README.md#known-inconsistencies-in-the-current-roll-ups) — `capabilities/` reads exactly those `__all__`/`_LAZY` tables, so a wrong lazy path becomes a wrong menu entry.
2. Migrate `bearings/dgbb_generic.py` to `Bearing.assemble()`.
3. Decide whether `Elem` joins `element_type/__init__.py`'s `__all__` (the fixture layer will want it for submodel work).
4. Convert `solvers/machine_elements/gears/__init__.py` to the lazy pattern (it is currently eager, on a stale path, and would break `catalogue()` on first read of the `gear_solvers` domain).

### Explicit non-goals

- **Not a plugin registry.** Nothing registers itself with `capabilities/`; it only reads what the packages already declare.
- **Not a DI container.** It returns classes and functions, not configured instances.
- **Not a stability guarantee.** A name that disappears from a roll-up's `__all__` must fail loudly in `validate()`, not be silently aliased.

---

## Implemented fixtures

### `gears/spur_helical.py`

Two independent classes at different levels of completeness: a single gear before pairing, and a meshed pair with full working geometry resolved.

**`GearFixture`** *(frozen dataclass)* — immutable wrapper for one `SpurHelicalGear`, storing the construction parameters (`z`, `mn`, `b`, `position`, `label`, `alpha_n_deg`, `beta_n_deg`, `x`, `material_id`, `Ra`, `haP`, `cP`, `rfP`) so derived copies can be produced without mutating the original.
- `with_x(x)`, `with_position(position)`, `validate()`, `validate_or_raise()`, `summary()`.

**`GearPairFixture`** — encapsulates the `SpurHelicalGearMeshing` instance that holds the real working geometry.
- `from_fixtures(driver, driven, label="", al=None, equalise_gs=False, addendum_reduction=False, gear1_is_driver=True)` *(classmethod)*.
- Properties: `al`, `u`, `epsilon_alpha`, `epsilon_beta`, `epsilon_gamma`, `x1`, `x2`.
- `validate()` / `validate_or_raise()` / `summary()`.
- `iso6336_contact_stress(...)`, `iso6336_bending_stress(...)`, `lubrication_assessment(...)` — declared, all `NotImplementedError`. They are the seams the ISO 6336 solver will fill; the data layer for `K_A` / `K_v` already exists in `axisforge/database/`.

**Factory:** `make_spur_helical(z, mn, b, position, label, alpha_n_deg=20.0, beta_n_deg=0.0, x=0.0, material_id="AISI_1045", Ra=0.8, haP=1.0, cP=0.25, rfP=0.38) -> GearFixture`.

`Ca`, `Cf`, `Rq`, `Rz` are not exposed here — callers needing non-standard values instantiate `SpurHelicalGear` directly.

### `shafts/shaft_fixture.py`

Generic N-section shaft fixture. Supports any stepped geometry via a list of `SectionSpec` descriptors.

**`SectionSpec`** *(frozen dataclass)* — `length`, `diameter`, `material_id="AISI_1045"`, `surface_ra=0.8`, `label=""`, `shoulder_left`, `shoulder_right`. Uses `Shoulder` from the core directly — no intermediate spec layer.

**`ShaftFixture`** *(frozen dataclass)* — `shaft`, `sections: tuple[SectionSpec, ...]`, `name`. Properties `total_length`, `n_sections`.
- `with_sections(sections)`, `with_name(name)`, `with_params(...)` *(3-section only; raises `ValueError` otherwise)*, `validate()`, `validate_or_raise()`, `summary()`.

**Factories:**
```python
make_shaft(sections: list[SectionSpec], name: str = "shaft") -> ShaftFixture
make_stepped_3section(total_length, d_seat, d_body, l_seat_a, l_seat_b, fillet_r,
                      material_id="AISI_1045", surface_ra=0.8, name="shaft") -> ShaftFixture
```
`l_body = total_length - l_seat_a - l_seat_b` is always derived; the same `Shoulder` object is applied to both faces of the body section.

Reference instance: `SHAFT_50_30_200` — d_body=50, d_seat=30, total_length=200, l_seat_a=l_seat_b=30, fillet_r=2.0, AISI_1045 (r/d = 0.067, D/d = 1.667).

> `surface_ra` applies uniformly across all sections. For section-specific treatment, build the `SectionSpec` list manually. `SectionSpec` does not yet carry keyways — a shaft needing a `Keyway` must have it attached to the `ShaftSection` after `make_shaft()`, or the spec extended.

### `bearings/dgbb_generic.py` *(stale — see [Status](#status))*

**`BearingFixture`** *(frozen dataclass, base)* — `bearing`, `d`, `D`, `b`, `C`, `C0`, `designation`, `position`, `arrangement`, `label`. `with_position()` / `with_arrangement()` raise `NotImplementedError` to force subclass implementation.

**`DGBBFixture`** — adds `ri`, `re`, `Dw`, `Dpw`, `Z`, `E`, `nu`, `s`, `alpha_0_deg`.
- `with_position()`, `with_arrangement()`, `with_label()`, `make_ready()`, `summary()`.

**Factory:** `make_dgbb(d, D, b, C, C0, designation, ri, re, Dw, Dpw, Z, E=206000.0, nu=0.3, s=None, alpha_0_deg=None, position=0.0, arrangement="locating", label="") -> DGBBFixture`.

**Migration target.** `make_ready()` should become a thin `Bearing.assemble()` call, and `ri`/`re` should stop being fixture inputs — `DeepGrooveBallFamily.reference_raceway_radii(Dw)` owns them (ISO 281 Table 1). `alpha_0_deg` should be dropped for DGBB entirely: that family is specified by clearance `s` alone.

### `systems/linear_gear_chain.py`

System fixture for a linear n-stage parallel-axis gear chain. **Responsibility: assembly only.** It receives already-built `ShaftFixture`, `GearPairFixture` and `Bearing` objects, creates the `ShaftSystem` containers, places gears and bearings, builds the links and resolves power flow.

**`StageSpec`** *(frozen dataclass)* — the minimum the system fixture needs that is not already in the `GearPairFixture`: `pos_driver`, `pos_driven`, `phi_deg`, `label=""`.

**`build_systems(...)`**
```python
build_systems(
    shaft_fixtures:      list[ShaftFixture],
    pairs:               list[GearPairFixture],
    bearings_per_shaft:  list[list[Bearing]],
    stage_specs:         list[StageSpec],
    P_W:                 float,
    rpm_in:              float,
    rotation_dir:        int = 1,
    label:               str = "",
    speed_rpm_per_shaft: list[float] | None = None,
) -> GearSystemResult
```

All shaft, gear and bearing objects must be fully constructed before calling. Bearings are passed **as already-assembled `Bearing` objects, one list per shaft** — the older `bearing_factory` callback is gone.

`shaft_origin_x` is computed automatically: `x[0] = 0.0`, `x[i+1] = x[i] + pos_driver[i] − pos_driven[i]`.

External loads (motor overhang, pulley tension, ...) are applied by the caller on `result.shaft_systems` after `build_systems()` returns. `resolve()` is idempotent — safe to re-call.

**`GearSystemResult`** — output dataclass:

| Attribute | Type | Description |
|---|---|---|
| `gearbox` | `SpurHelicalGearSystem` | Fully resolved gearbox |
| `shaft_systems` | `list[ShaftSystem]` | Ordered, source to last driven |
| `pairs` | `list[GearPairFixture]` | One per stage |
| `gear_fixtures` | `list[tuple[GearFixture, GearFixture]]` | (driver, driven) per stage with resolved x1, x2 |
| `gear_elements` | `list[tuple[GearElement, GearElement]]` | (ge_driver, ge_driven) per stage; the same objects used in the links |
| `links` | `list[SpurHelicalMeshLink]` | One per stage |
| `label` | `str` | Gearbox label |

Convenience: `summary()`, `print_gear_summary()`, `driver_fixture(i)`, `driven_fixture(i)`, `driver_element(i)`, `driven_element(i)`.

**Fan-out / power split** is out of scope here. It requires a dedicated pre-solver that computes `torque_split` before the links reach `SpurHelicalGearSystem`. `build_systems()` handles linear chains only.

---

## Key design contracts

### Immutability

All fixture objects (`GearFixture`, `GearPairFixture`, `ShaftFixture`, `BearingFixture`, `DGBBFixture`) are `frozen=True` dataclasses. Modification always produces a new object:

```python
gear2  = gear.with_x(0.15)
shaft2 = shaft.with_params(d_body=55.0)
brg2   = brg.with_position(35.0).with_arrangement("floating")
```

The core `Bearing` is immutable for a different and stronger reason: `Bearing.__setattr__` refuses every write once `assemble()` completes. A fixture cannot patch an assembled bearing — it must build a new one.

### Profile shift

`GearFixture.x` is the declared value. The working coefficients `x1`, `x2` are resolved by `SpurHelicalGearMeshing` inside `GearPairFixture.from_fixtures()`. Access via `pair.x1` / `pair.x2`, or `result.driver_fixture(i).x` after `build_systems()`.

### Bearing readiness

A bearing is ready when it has been assembled with the analyses the downstream solver needs:

```python
brg = Bearing.assemble(
    family=DeepGrooveBallFamily(),
    catalog=BearingCatalog(d=50, D=90, b=20, C=37_100, C0=23_200,
                           designation="6210", position=35.0, label="A"),
    geometry=dict(Dw=12.7, Dpw=70.0, Z=10, E=206000, s=0.012),
    analyses={"point_contact": True},
)
# brg.has_internal_geometry() is True; brg.cp is set
# brg.is_enabled("point_contact") is True -> dispatch.py will find a solver
```

`Bearing.assemble()` raises at construction if the requested analysis is unsupported by the family or if any `REQUIRED_FOR` field is missing — the failure is at the fixture, not several hundred lines into the solve.

### System topology constraints

- Each `SpurHelicalGearSystem` built by `linear_gear_chain.py` is a linear chain: shaft_1 → shaft_2 → ... → shaft_n.
- n stages produce n+1 shafts and n mesh links.
- Non-linear topologies require multiple independent `SpurHelicalGearSystem` instances.
- Planetary arrangements are out of scope for this fixture — use `PlanetaryGearTrainMeshing` from `core/machine_elements/gears/parallel_axis/planetary_gear/` directly.

### Fixture exposure principle

Fixtures organise and expose the program's capabilities; they do not hide them. Every intermediate object produced during assembly is accessible from `GearSystemResult` without reconstruction. Downstream solvers (ISO/TS 16281, ISO 6336, fatigue, convergence) consume objects directly from the result.

`capabilities/` is the same principle applied one level up: it makes the *set of available capabilities* itself inspectable, instead of something you learn by reading import lines.

---

## Typical analysis script structure

```python
# 1. Declare capabilities (planned — fixtures/capabilities/)
CAPS = Capabilities(...)
globals().update(CAPS.resolve())

# 2. Declare parameters
stage_specs   = [StageSpec(...), ...]
shaft_fixtures = [make_stepped_3section(...), ...]
pairs          = [GearPairFixture.from_fixtures(...), ...]
bearings       = [[Bearing.assemble(...), Bearing.assemble(...)], ...]

# 3. Build and resolve
result = build_systems(shaft_fixtures, pairs, bearings, stage_specs, P_W, rpm_in)

# 4. Add external loads
result.shaft_systems[0].add_load(TorqueLoad(...))    # motor coupling
result.shaft_systems[-1].add_load(RadialLoad(...))   # driven machine

# 5. Close torque equilibrium (if needed)
result.shaft_systems[0].add_load(TorqueLoad(0.0, -T_in, ...))

# 6. FEM             (planned: fixtures/solvers/fem_simple.py)
# 7. ISO/TS 16281    (planned: fixtures/solvers/iso16281_coupled.py)
# 8. Output + plots  (planned: fixtures/outputs/, fixtures/plots/)
```

---

## Usage rules

- Each fixture is self-contained and independently importable.
- The dependency chain runs bottom-up: capabilities → gears / shafts / bearings → systems → solvers → outputs / plots → integration.
- `integration/` files are the only entry points that combine the full pipeline. Copy one and adjust the `PARAMETERS` block to produce a new `design_xxx.py`.
- **Do not modify a fixture for a specific case — duplicate and rename.**
- Named reference instances must not be mutated. Use the appropriate `with_*()` method to produce modified copies.
- Console output in fixtures and scripts is **ASCII only** — Windows PowerShell (cp1252).
- `capabilities/` may read `__all__` and `_LAZY`; it must never write to a package or be imported by one.