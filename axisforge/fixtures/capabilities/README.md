# axisforge/fixtures — Analysis Script Building Blocks

Reusable building blocks for assembling `design_xxx.py` analysis scripts — functional runs, pipeline verification and textbook validation. These are **not** pytest tests; unit and validation tests live in `tests/`.

Each layer depends only on the layers below it, and none of them is imported by `axisforge` itself.

← back to [project root](../../README.md)

---

## Table of Contents

- [Status](#status)
- [Structure](#structure)
- [Capabilities](#capabilities)
- [Gear fixtures](#gear-fixtures)
- [Shaft fixtures](#shaft-fixtures)
- [Bearing fixtures](#bearing-fixtures)
- [System fixtures](#system-fixtures)
- [Design contracts](#design-contracts)
- [Script structure](#script-structure)
- [Usage rules](#usage-rules)

---

## Status

| Layer | Module | Status |
|---|---|---|
| `gears/` | `spur_helical.py` | Implemented |
| `shafts/` | `shaft_fixture.py` | Implemented |
| `bearings/` | `dgbb_generic.py` | Implemented, pending migration to the current bearing assembly API |
| `bearings/` | `crb_generic.py`, `angular_contact.py` | Reserved |
| `systems/` | `linear_gear_chain.py` | Implemented |
| `capabilities/` | `__init__.py` | Implemented |
| `solvers/` | `fem_simple.py`, `iso16281_coupled.py`, `static_analysis.py` | Planned |
| `outputs/` | `bearings.py`, `gears.py`, `shaft.py` | Implemented |
| `plots/` | deflection, polar, convergence | Planned |
| `convergence/` | `mesh_gci_study.py` | Planned |
| `integration/` | full pipelines | Planned |

---

## Structure

```
fixtures/
├── capabilities/       Declares what a script needs; resolves it to imports
├── gears/              GearFixture, GearPairFixture
├── shafts/             SectionSpec, ShaftFixture
├── bearings/           BearingFixture, DGBBFixture
├── systems/            StageSpec, GearSystemResult, build_systems
├── solvers/            FEM, ISO/TS 16281 and static-analysis wrappers
├── outputs/            Console reporters
├── plots/              matplotlib figures
├── convergence/        Mesh convergence runs
└── integration/        Full pipelines, ready to copy into a design script
```

The dependency chain runs bottom-up:

```
integration/
    ↑
outputs/  plots/
    ↑
solvers/
    ↑
systems/
    ↑
gears/  shafts/  bearings/
    ↑
capabilities/
    ↑
axisforge core, mesh and solver packages
```

---

## Capabilities

`capabilities/` lets a script declare **what it needs to do** and receive exactly the objects that job requires — instead of forty hand-written import lines whose dotted paths must be kept in step with the package tree.

It works by reading what each package already publishes in its `__init__.py`: the list of names it offers, and where each one lives. It reads only; nothing in `axisforge` imports from `fixtures`.

The whole declaration lives in one module, `capabilities/__init__.py`, because the thing that matters here is not a menu per domain but a single dependency chain that runs through all of them. Console reporters are the one part kept apart, in `outputs/` — see [below](#interface).

### The chain

A design script's needs form four stages, each meaningless without the one before it:

```
Construction  →  MeshLoads  →  Resolution  →  ElementAnalysis
```

| Stage | Class | Does | Needs |
|---|---|---|---|
| 1 | `ConstructionCapabilities` | Instantiates the objects — shaft, bearings, bearing families, gears, the shaft/gear system container. Nothing is solved. `use_fixtures` (default on) layers the `fixtures/{shafts,bearings,gears,systems}/` wrapper on top, per domain requested — see [Fixture wrappers](#fixture-wrappers). | Nothing |
| 2 | `MeshLoadsCapabilities` | `power_flow` resolves the gearbox's torque/speed (`SpurHelicalGearSystem.resolve()`); `gear_forces` computes the mesh force at a gear from that resolved torque (`GearSolver.compute_forces()`), producing the load the shaft FEM needs as input. | `construction.system`; `gear_forces` also needs `construction.gears` and `power_flow` |
| 3 | `ResolutionCapabilities` | `shaft_fem` solves the shaft (`SimpleFEMSolver`, read back through `ShaftResultsReader`/`SimpleFEMResultsLibrary`); `bearing_loads` solves the bearings' internal load distribution from those results (`RollingBearingSolver`). | `shaft_fem` needs `construction.shaft`; `bearing_loads` needs `shaft_fem` and `construction.bearings` |
| 4 | `ElementAnalysisCapabilities` | Per-element analyses read off a resolved system: `point_contact`, `line_contact`, `multirow_capacity` for bearings; `shaft_static_report` for the shaft; `gear_load_capacity` for gears (reserved). | `point_contact`/`line_contact` need `bearing_loads`; `multirow_capacity` needs one of them; `shaft_static_report` needs `shaft_fem` |

Each class's `validate_chain()` walks back through every earlier stage, so a script that asks for `bearing_loads` without `shaft_fem`, or `gear_forces` without `power_flow`, fails at `validate_or_raise()` — before a single name is imported, and with the specific missing prerequisite named in the error.

`gear_load_capacity` is registered for the ISO 6336 solver reserved in [`solvers/README.md`](../solvers/README.md#gear-solver): enabling it raises `not implemented yet`, rather than failing on an import that does not exist.

### Fixture wrappers

`ConstructionCapabilities.use_fixtures` (default `True`) decides, for the whole stage at once, whether resolving a domain also pulls in that domain's convenience wrapper. The table below is a schema for that wiring, not a finished one: the four wrapper modules it points at are themselves still due to be developed, one at a time, alongside each capability stage — the names will be kept in step with them as that happens, without the shape of `ConstructionCapabilities` needing to change.

| Domain | Core names (always) | Wrapper added when `use_fixtures=True` | From |
|---|---|---|---|
| `shaft` | `Shaft`, `ShaftSection`, `Shoulder` | `SectionSpec`, `ShaftFixture`, `make_shaft`, `make_stepped_3section` | `fixtures.shafts.shaft_fixture` |
| `bearings` | `Bearing`, `BearingCatalog` | `BearingFixture`, `DGBBFixture`, `make_dgbb` | `fixtures.bearings.dgbb_generic` |
| `gears` | `SpurHelicalGear`, `SpurHelicalGearMeshing` | `GearFixture`, `GearPairFixture`, `make_spur_helical` | `fixtures.gears.spur_helical` |
| `system` | `ShaftSystem`, `GearElement`, ... | `StageSpec`, `GearSystemResult`, `build_systems` | `fixtures.systems.linear_gear_chain` |

The wrapper is additive, never a replacement — the core names stay resolved either way, since a wrapper does not cover everything (`SectionSpec` still takes the core `Shoulder` directly, for instance). Setting `use_fixtures=False` gives the bare core classes only, for a script that builds every object by hand instead of through a fixture. It is only pulled in for a domain actually requested — an empty `gears` tuple means no `GearFixture` either, `use_fixtures` or not. `bearing_families` has no wrapper of its own and is unaffected by the flag.

This is the one place `capabilities/` reaches sideways into another fixtures package rather than down into `axisforge` — see the note on [Structure](#structure): `gears/`, `shafts/`, `bearings/` and `systems/` are drawn below `capabilities/` there because they are the simpler, more foundational layer, not because dependencies only run that way. `bearings` has no `use_fixtures=True` path through the current bearing assembly API yet — `fixtures/bearings/dgbb_generic.py` still predates `Bearing.assemble()` with a family, see [Bearing fixtures](#bearing-fixtures).

### Outputs

Two stages carry an `outputs` field: `MeshLoadsCapabilities` (for `gear_forces`) and `ElementAnalysisCapabilities` (for each of its four analyses). Setting `outputs["point_contact"] = True` pulls in that analysis' console reporter from `fixtures/outputs/`; leaving it `False` — or absent — runs the analysis without importing anything that prints. The dependency runs one way: an output cannot be requested for an analysis that is not itself enabled, but an analysis can be enabled with its output left off.

| Analysis / mesh load | Reporter | Module |
|---|---|---|
| `gear_forces` | `print_gear_forces_report` | `fixtures.outputs.gears` |
| `point_contact` | `print_point_contact_report` | `fixtures.outputs.bearings` |
| `line_contact` | `print_line_contact_report` | `fixtures.outputs.bearings` |
| `multirow_capacity` | `print_multirow_capacity_report` | `fixtures.outputs.bearings` |
| `shaft_static_report` | `print_static_report` | `fixtures.outputs.shaft` |

Every reporter is a pure consumer of already-computed results — it prints, it does not solve — and ASCII only, per the console-output convention.

### Interface

| Name | Purpose |
|---|---|
| `catalogue` | Everything selectable, per domain, without resolving anything. |
| `where` | The domain and module a given name comes from — for diagnostics. |
| `print_menu` | An ASCII table of the whole catalogue, plus the chained-analysis vocabulary. |
| `ConstructionCapabilities`, `MeshLoadsCapabilities`, `ResolutionCapabilities`, `ElementAnalysisCapabilities` | The four chained stages. Each takes the previous stage as a field, so a stage cannot be built without its prerequisite already declared. |
| `Capabilities` | The full declaration for a script: an `ElementAnalysisCapabilities` (which carries the whole chain beneath it), plus a flat name tuple per domain with no stage of its own — `loads`, `materials`, `mesh`, `elements`, `schematic`, `convergence`. |
| `Capabilities.validate` / `validate_or_raise` | Checks every requested name and the whole capability chain. |
| `Capabilities.resolve` | Imports exactly what was selected and returns it by name, raising on a collision across domains rather than shadowing. |
| `Capabilities.summary` | What was selected and what it pulled in, per stage. |

### Usage

```python
from axisforge.fixtures.capabilities import (
    Capabilities, ConstructionCapabilities, MeshLoadsCapabilities,
    ResolutionCapabilities, ElementAnalysisCapabilities,
)

construction = ConstructionCapabilities(
    shaft            = ("Shaft", "ShaftSection", "Shoulder"),
    bearings         = ("Bearing", "BearingCatalog"),
    bearing_families = ("DeepGrooveBallFamily",),
    gears            = ("SpurHelicalGear", "SpurHelicalGearMeshing"),
    system           = ("ShaftSystem", "GearElement",
                        "SpurHelicalMeshLink", "SpurHelicalGearSystem"),
)
mesh_loads = MeshLoadsCapabilities(
    construction=construction, power_flow=True, gear_forces=True,
    outputs={"gear_forces": True},
)
resolution = ResolutionCapabilities(mesh_loads=mesh_loads, shaft_fem=True, bearing_loads=True)
analysis   = ElementAnalysisCapabilities(
    resolution=resolution,
    point_contact=True,
    shaft_static_report=True,
    outputs={"point_contact": True, "shaft_static_report": False},   # solve the shaft report, do not print it
)

CAPS = Capabilities(element_analysis=analysis, loads=("TorqueLoad", "RadialLoad"))
CAPS.validate_or_raise()
globals().update(CAPS.resolve())
```

Nothing else is imported: no roller postprocessing, no planetary geometry, no GUI, and — because its output stayed off — no shaft report printer either, even though the shaft is fully solved.

**Not** a plugin registry, not a dependency-injection container, and not a stability layer: it returns classes and functions, and a name that disappears from a package's surface fails loudly.

---

## Gear fixtures

`gears/spur_helical.py`. Two classes at different stages of completeness: a single gear before pairing, and a meshed pair with the working geometry resolved.

### `GearFixture`

Immutable wrapper for one gear, storing the construction parameters so derived copies can be produced without mutating the original: teeth count, module, face width, position, label, pressure and helix angles, profile shift, material, roughness and the basic rack parameters.

| Member | Purpose |
|---|---|
| `with_x` | A copy at a different profile shift. |
| `with_position` | A copy at a different axial position. |
| `validate`, `validate_or_raise` | Delegates to the core gear. |
| `summary` | |

`make_spur_helical` is the factory; it builds the core gear and wraps it.

Tip and root relief and the finer roughness parameters are not exposed — a caller needing them instantiates the core gear directly.

### `GearPairFixture`

Wraps the meshing object that holds the real working geometry.

| Member | Purpose |
|---|---|
| `from_fixtures` | Builds a pair from two gear fixtures, optionally with an imposed centre distance, profile-shift equalisation or addendum reduction. |
| `al`, `u` | Working centre distance and ratio. |
| `epsilon_alpha`, `epsilon_beta`, `epsilon_gamma` | Contact ratios. |
| `x1`, `x2` | The working profile shift coefficients, after correction. |
| `validate`, `validate_or_raise`, `summary` | |
| `iso6336_contact_stress`, `iso6336_bending_stress`, `lubrication_assessment` | Declared seams for later phases; not implemented. |

---

## Shaft fixtures

`shafts/shaft_fixture.py`. Any stepped geometry, described as an ordered list of section specifications.

### `SectionSpec`

Declarative descriptor for one section: length, diameter, material, surface roughness, label, and optional shoulders. It uses the core `Shoulder` directly — there is no intermediate spec layer.

### `ShaftFixture`

Immutable wrapper for a shaft, keeping the section specs that produced it so modified copies can be rebuilt without parsing the core object.

| Member | Purpose |
|---|---|
| `total_length`, `n_sections` | Aggregate properties. |
| `with_sections` | A copy from a different section list, keeping the name. |
| `with_name` | A copy under a different name. |
| `with_params` | A copy with individual parameters overridden — only for fixtures produced by the three-section factory. |
| `validate`, `validate_or_raise`, `summary` | |

| Factory | Purpose |
|---|---|
| `make_shaft` | Generic, N sections, full control over each. |
| `make_stepped_3section` | The canonical seat–body–seat case from scalar parameters; the body length is always derived, and the same shoulder is applied to both body faces. |

A named reference instance is provided for the canonical stepped shaft.

Surface roughness applies uniformly across sections; for section-specific treatment, build the spec list by hand. Keyways are not part of the spec — attach them to the core section after building, or extend the spec.

---

## Bearing fixtures

`bearings/dgbb_generic.py`. A catalogue-level base class and a deep-groove ball subclass that adds internal geometry.

| Class | Purpose |
|---|---|
| `BearingFixture` | Immutable base: the bearing plus its catalogue and mounting parameters. The copy methods raise, to force subclass implementation. |
| `DGBBFixture` | Adds the internal geometry parameters and the copy methods, plus a method that returns a solver-ready bearing. |
| `make_dgbb` | The factory. |

This module predates the current bearing assembly API and does not go through `Bearing.assemble()` with a family. Build bearings directly, as shown under [Design contracts](#design-contracts), until it is aligned.

---

## System fixtures

`systems/linear_gear_chain.py`. Assembly only: it receives fully built shaft, gear-pair and bearing objects, creates the shaft containers, places gears and bearings, builds the mesh links and resolves the power flow.

### `StageSpec`

The minimum a stage needs beyond its gear pair: the driver and driven axial positions, the line-of-centres angle, and a label.

### `build_systems`

Takes the shaft fixtures, the gear pairs, the bearings grouped per shaft, the stage specifications, the input power and speed, and optionally the rotation direction, a label and per-shaft speeds. Returns a `GearSystemResult`.

Bearings are passed as already-assembled bearing objects, one list per shaft. Shaft axial origins are computed from the stage positions. External loads are applied by the caller on the returned shaft systems; resolving is idempotent, so it is safe to re-run after adding them.

### `GearSystemResult`

| Attribute | Contents |
|---|---|
| `gearbox` | The fully resolved gear system. |
| `shaft_systems` | The shaft containers, ordered from source to last driven. |
| `pairs` | One gear pair fixture per stage. |
| `gear_fixtures` | Driver and driven gear fixtures per stage, with the resolved profile shifts. |
| `gear_elements` | Driver and driven gear elements per stage — the same objects used in the links. |
| `links` | One mesh link per stage. |
| `label` | Gearbox label. |

| Member | Purpose |
|---|---|
| `summary` | Delegates to the resolved gearbox. |
| `print_gear_summary` | Per-stage gear table, ASCII. |
| `driver_fixture`, `driven_fixture` | The gear fixtures for a stage. |
| `driver_element`, `driven_element` | The gear elements for a stage. |

**Scope.** Linear chains only: n stages give n+1 shafts and n links. A fan-out needs a pre-solver to compute the torque split before the links are built, and is out of scope here. Non-linear topologies are modelled as several independent gear systems; planetary arrangements use the planetary meshing class directly.

---

## Design contracts

**Immutability.** Every fixture object is a frozen dataclass; modification produces a new object through a `with_*` method.

```python
gear2  = gear.with_x(0.15)
shaft2 = shaft.with_params(d_body=55.0)
brg2   = brg.with_position(35.0).with_arrangement("floating")
```

The core bearing is immutable for a stronger reason: it refuses every write once assembled. A fixture cannot patch an assembled bearing — it builds a new one.

**Profile shift.** The value declared on a gear fixture is provisional. The working coefficients are resolved by the meshing object when the pair is built, and read back from the pair or from the result.

**Bearing readiness.** A bearing is ready when it has been assembled with the analyses the downstream solver needs. Assembly raises immediately if the family does not support a requested analysis or if a required geometry field is missing, so the failure surfaces at the fixture rather than hundreds of lines into the solve.

```python
brg = Bearing.assemble(
    family=DeepGrooveBallFamily(),
    catalog=BearingCatalog(d=50, D=90, b=20, C=37_100, C0=23_200,
                           designation="6210", position=35.0, label="A"),
    geometry=dict(Dw=12.7, Dpw=70.0, Z=10, E=206_000, s=0.012),
    analyses={"point_contact": True},
)
```

**Exposure.** Fixtures organise and expose the program's capabilities; they do not hide them. Every intermediate object produced during assembly is reachable from the result without reconstruction, and downstream solvers consume those objects directly.

---

## Script structure

```python
# 1 — Declare capabilities
CAPS = Capabilities(...)
globals().update(CAPS.resolve())

# 2 — Parameters
stage_specs    = [StageSpec(...), ...]
shaft_fixtures = [make_stepped_3section(...), ...]
pairs          = [GearPairFixture.from_fixtures(...), ...]
bearings       = [[Bearing.assemble(...), Bearing.assemble(...)], ...]

# 3 — Build and resolve
result = build_systems(shaft_fixtures, pairs, bearings, stage_specs, P_W, rpm_in)

# 4 — External loads
result.shaft_systems[0].add_load(TorqueLoad(...))     # motor coupling
result.shaft_systems[-1].add_load(RadialLoad(...))    # driven machine

# 5 — FEM, bearings, output
```

---

## Usage rules

- Each fixture is self-contained and independently importable.
- `integration/` files are the only entry points that combine the full pipeline. Copy one and adjust its parameter block to produce a new design script.
- Do not modify a fixture for a specific case — duplicate and rename it.
- Named reference instances are never mutated; use a `with_*` method for a modified copy.
- Console output is ASCII only.
- `capabilities/` reads what packages declare and is never imported by one.