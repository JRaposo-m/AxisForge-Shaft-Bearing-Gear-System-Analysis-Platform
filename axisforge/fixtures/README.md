# AxisForge — Fixtures

Modular template library for rapid construction of analysis and validation
scripts. These files are **not pytest unit tests**. They are reusable building
blocks for assembling `design_xxx.py` scripts — functional analysis, pipeline
verification, and textbook validation runs.

Each fixture layer depends only on the layers below it. The dependency chain
runs from core classes upward through gears / shafts / bearings to systems,
solvers, outputs, plots, and finally integration scripts.

---

## Status

| Layer | File | Status |
|---|---|---|
| `gears/` | `spur_helical.py` | Implemented and validated |
| `shafts/` | `shaft_fixture.py` | Implemented and validated |
| `bearings/` | `dgbb_generic.py` | Implemented and validated |
| `systems/` | `linear_gear_chain.py` | Implemented and validated |
| `solvers/` | `fem_simple.py` | Planned |
| `solvers/` | `iso16281_coupled.py` | Planned |
| `outputs/` | all | Planned |
| `plots/` | all | Planned |
| `convergence/` | `mesh_gci_study.py` | Planned |
| `integration/` | all | Planned |

---

## Directory Structure

```
fixtures/
|
|-- gears/
|   +-- spur_helical.py
|
|-- shafts/
|   +-- shaft_fixture.py
|
|-- bearings/
|   |-- dgbb_generic.py
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

## Fixture Dependency Chain

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
axisforge core classes               <- never modified here
```

---

## Implemented Fixtures

---

### `gears/spur_helical.py`

Gear fixtures for spur and helical gears. Two independent classes at
different levels of completeness: a single gear before pairing, and a
meshed pair with full working geometry resolved.

#### Classes

**`GearFixture`** `(frozen dataclass)`

Immutable wrapper for a single `SpurHelicalGear`. Stores all construction
parameters so that modified copies can be produced without parsing the
core object. The `x` field holds the profile shift as declared; the
working value after meshing is resolved by `GearPairFixture`.

| Attribute | Type | Description |
|---|---|---|
| `gear` | `SpurHelicalGear` | Core gear object |
| `z` | `int` | Number of teeth |
| `mn` | `float` | Normal module [mm] |
| `b` | `float` | Face width [mm] |
| `position` | `float` | Axial position on shaft [mm] |
| `label` | `str` | Identifier |
| `alpha_n_deg` | `float` | Normal pressure angle [deg] |
| `beta_n_deg` | `float` | Helix angle [deg] |
| `x` | `float` | Profile shift coefficient |
| `material_id` | `str` | Material identifier |
| `Ra` | `float` | Arithmetic mean roughness [um] |
| `haP`, `cP`, `rfP` | `float` | ISO 53 rack coefficients |

Methods: `with_x(x)`, `with_position(pos)`, `validate()`, `validate_or_raise()`, `summary()`

**`GearPairFixture`** `(dataclass)`

Meshed pair with full working geometry. Constructed via `from_fixtures()`.
Holds the `SpurHelicalGearMeshing` instance that carries the resolved
`x1`, `x2`, `al`, `alphatw`, and contact ratio values.
Extension point for ISO 6336 and lubrication analysis (stubs declared).

| Attribute | Type | Description |
|---|---|---|
| `driver` | `GearFixture` | Driver gear fixture |
| `driven` | `GearFixture` | Driven gear fixture |
| `meshing` | `SpurHelicalGearMeshing` | Working geometry |
| `label` | `str` | Stage label |

Properties: `al`, `u`, `x1`, `x2`, `epsilon_alpha`, `epsilon_beta`, `epsilon_gamma`

Methods: `from_fixtures(driver, driven, label, al, equalise_gs, addendum_reduction)`,
`validate()`, `validate_or_raise()`, `summary()`

Stubs (Phase 2/3): `iso6336_contact_stress()`, `iso6336_bending_stress()`, `lubrication_assessment()`

#### Factory

```python
make_spur_helical(
    z, mn, b, position, label,
    alpha_n_deg=20.0, beta_n_deg=0.0, x=0.0,
    material_id="AISI_1045", Ra=0.8,
    haP=1.0, cP=0.25, rfP=0.38,
) -> GearFixture
```

#### Reference instances

| Name | z | mn | b [mm] | Notes |
|---|---|---|---|---|
| `SPUR_20T_MN2` | 20 | 2 | 20 | Typical pinion, first stage |
| `SPUR_40T_MN2` | 40 | 2 | 20 | Intermediate gear |
| `SPUR_60T_MN2` | 60 | 2 | 20 | Typical wheel, second stage |

All: `alpha_n_deg=20`, `beta_n_deg=0`, `x=0`, `material_id="AISI_1045"`, `position=0.0`.
Do not mutate. Use `with_x()` or `with_position()` for assembly-ready copies.

---

### `shafts/shaft_fixture.py`

Generic N-section shaft fixture. Supports any stepped geometry via a list
of `SectionSpec` descriptors. A convenience factory covers the canonical
seat | body | seat case.

#### Classes

**`SectionSpec`** `(frozen dataclass)`

Declarative descriptor for one shaft section. Uses `Shoulder` from the
core directly — no intermediate spec layer.

| Attribute | Type | Description |
|---|---|---|
| `length` | `float` | Axial length [mm] |
| `diameter` | `float` | Outer diameter [mm] |
| `material_id` | `str` | Material identifier |
| `surface_ra` | `float` | Arithmetic mean roughness Ra [um] |
| `label` | `str` | Section label |
| `shoulder_left` | `Shoulder or None` | Left face transition |
| `shoulder_right` | `Shoulder or None` | Right face transition |

**`ShaftFixture`** `(frozen dataclass)`

Immutable wrapper for a `Shaft` built from an ordered list of `SectionSpec`
objects. Stores the section list as a tuple so that `with_sections()` can
reconstruct modified copies without parsing the core `Shaft`.

| Attribute | Type | Description |
|---|---|---|
| `shaft` | `Shaft` | Core shaft object |
| `sections` | `tuple[SectionSpec, ...]` | Ordered section descriptors |
| `name` | `str` | Shaft name |

Properties: `total_length`, `n_sections`

Methods: `with_sections(sections)`, `with_name(name)`, `with_params(...)` (3-section only),
`validate()`, `validate_or_raise()`, `summary()`

#### Factories

```python
make_shaft(
    sections: list[SectionSpec],
    name: str = "shaft",
) -> ShaftFixture
```
Generic factory for N sections. Full control over each section.

```python
make_stepped_3section(
    total_length, d_seat, d_body,
    l_seat_a, l_seat_b, fillet_r,
    material_id="AISI_1045", surface_ra=0.8, name="shaft",
) -> ShaftFixture
```
Convenience factory for seat_A | body | seat_B geometry.
`l_body = total_length - l_seat_a - l_seat_b` is always derived.
The same `Shoulder` object is applied to both faces of the body section.

#### Reference instance

`SHAFT_50_30_200` — d_body=50 mm, d_seat=30 mm, total_length=200 mm,
l_seat_a=l_seat_b=30 mm, fillet_r=2.0 mm, AISI_1045.
r/d = 0.067, D/d = 1.667.

---

### `bearings/dgbb_generic.py`

Bearing fixtures for Deep Groove Ball Bearings. Two-level hierarchy:
a catalog-level base class and a DGBB-specific subclass that adds
internal geometry and solver readiness.

#### Classes

**`BearingFixture`** `(frozen dataclass)` — base class

Stores catalog and mounting parameters. Not instantiated directly;
`with_position()` and `with_arrangement()` raise `NotImplementedError`
to enforce subclass implementation.

| Attribute | Type | Description |
|---|---|---|
| `bearing` | `Bearing` | Core bearing object |
| `d`, `D`, `b` | `float` | Bore, outer diameter, width [mm] |
| `C`, `C0` | `float` | Dynamic / static load rating [N] |
| `designation` | `str` | Catalog designation |
| `position` | `float` | Axial position on shaft [mm] |
| `arrangement` | `str` | `'locating'`, `'floating'`, `'non-locating'` |
| `label` | `str` | Identifier |

**`DGBBFixture`** `(frozen dataclass)` — extends `BearingFixture`

Adds internal geometry parameters required by `IterativeBearingFEMSolver`
(ISO/TS 16281). Internal geometry is stored but not applied until
`make_ready()` is called, keeping the fixture immutable.

Additional attributes:

| Attribute | Type | Description |
|---|---|---|
| `ri`, `re` | `float` | Inner / outer groove radius [mm] |
| `Dw` | `float` | Ball diameter [mm] |
| `Dpw` | `float` | Pitch circle diameter [mm] |
| `Z` | `int` | Number of balls |
| `E`, `nu` | `float` | Young's modulus [MPa], Poisson's ratio |
| `s` | `float or None` | Diametral operating clearance [mm] |
| `alpha_0_deg` | `float or None` | Free contact angle [deg] |

Exactly one of `s` / `alpha_0_deg` must be provided.

Methods: `with_position(pos)`, `with_arrangement(arr)`, `with_label(label)`,
`validate()`, `validate_or_raise()`, `summary()`

**`make_ready() -> DeepGrooveBallBearing`**

Builds a fresh `DeepGrooveBallBearing`, calls `setup_internal_geometry()`
and `compute_hertz_point_contact()`. Returns a solver-ready object with
`has_internal_geometry() == True` and `cp` set. The fixture itself is
not modified. Each call returns a new independent instance.

#### Factory

```python
make_dgbb(
    d, D, b, C, C0, designation,
    ri, re, Dw, Dpw, Z,
    E=206_000.0, nu=0.3,
    s=None, alpha_0_deg=None,
    position=0.0, arrangement="locating", label="",
) -> DGBBFixture
```

#### Reference instances

| Name | Designation | d | D | b | C [N] | C0 [N] |
|---|---|---|---|---|---|---|
| `DGBB_6208` | SKF 6208 | 40 | 80 | 18 | 29 600 | 17 800 |
| `DGBB_6210` | SKF 6210 | 50 | 90 | 20 | 35 100 | 23 200 |
| `DGBB_6304` | SKF 6304 | 20 | 52 | 15 | 15 900 | 7 800 |

All: `position=0.0`, `arrangement='locating'`. Internal geometry stored but
not applied. Call `make_ready()` on a positioned copy for solver use:

```python
bearing = DGBB_6208.with_position(35.0).with_label("A").make_ready()
```

---

### `systems/linear_gear_chain.py`

Assembles and resolves a linear n-stage parallel-axis gear chain.
Produces a `GearSystemResult` that exposes every intermediate object
for downstream analysis without reconstruction.

#### Input dataclasses

**`BearingSpec`**

| Field | Type | Description |
|---|---|---|
| `position` | `float` | Axial position on shaft [mm] |
| `label` | `str` | Bearing label |
| `arrangement` | `str` | `'locating'`, `'floating'`, `'non-locating'` |

**`StageSpec`**

| Field | Type | Default | Description |
|---|---|---|---|
| `z_driver` | `int` | — | Driver tooth count |
| `z_driven` | `int` | — | Driven tooth count |
| `pos_driver` | `float` | — | Driver gear axial position [mm] |
| `pos_driven` | `float` | — | Driven gear axial position [mm] |
| `phi_deg` | `float` | — | Line-of-centres angle [deg] |
| `mn` | `float` | — | Normal module [mm] |
| `alpha_n_deg` | `float` | — | Normal pressure angle [deg] |
| `b` | `float` | — | Face width [mm] |
| `beta_n_deg` | `float` | `0.0` | Helix angle [deg] |
| `x_driver` | `float` | `0.0` | Profile shift, driver |
| `x_driven` | `float` | `0.0` | Profile shift, driven |
| `al` | `float or None` | `None` | Imposed working centre distance [mm] |
| `equalise_gs` | `bool` | `False` | Henriot equalisation of specific sliding |
| `addendum_reduction` | `bool` | `False` | MAAG addendum reduction coefficient k |
| `material_id` | `str` | `"AISI_1045"` | Gear material ID |
| `label` | `str` | `""` | Stage label |

Profile shift resolution priority: `equalise_gs=True` overrides `x_driver`/`x_driven`;
`al` imposed overrides both; otherwise `x_driver`/`x_driven` are used as declared.

**`ShaftSpec`**

| Field | Type | Default | Description |
|---|---|---|---|
| `total_length` | `float` | — | Total shaft length [mm] |
| `d_seat` | `float` | — | Bearing seat diameter [mm] |
| `d_body` | `float` | — | Body diameter under gear [mm] |
| `l_seat_a` | `float` | — | Left seat length [mm] |
| `l_seat_b` | `float` | — | Right seat length [mm] |
| `fillet_r` | `float` | — | Shoulder fillet radius [mm] |
| `bearings` | `list[BearingSpec]` | — | Ordered bearing descriptors |
| `speed_rpm` | `float` | — | Nominal shaft speed [rpm] |
| `material_id` | `str` | `"AISI_1045"` | Shaft material ID |
| `name` | `str` | `""` | Shaft name |

#### Main factory

```python
build_systems(
    stages:          list[StageSpec],
    shaft_specs:     list[ShaftSpec],   # len == len(stages) + 1
    bearing_factory: callable,          # (BearingSpec, ShaftSpec) -> Bearing
    P_W:             float,             # input power [W]
    rpm_in:          float,             # input shaft speed [rpm]
    rotation_dir:    int = 1,
    label:           str = "",
) -> GearSystemResult
```

`bearing_factory` is injected by the caller. It receives a `BearingSpec`
and the corresponding `ShaftSpec` (for context such as `d_seat` or
`speed_rpm`) and returns a concrete `Bearing` object. This keeps the
system fixture independent of any specific bearing type.

`shaft_origin_x` is computed automatically:
`x[0] = 0.0`, `x[i+1] = x[i] + pos_driver[i] - pos_driven[i]`.

External loads (motor overhang, pulley tension, etc.) are applied by
the caller on `result.shaft_systems` after `build_systems()` returns.
`resolve()` is idempotent — safe to re-call.

#### `GearSystemResult` — output dataclass

| Attribute | Type | Description |
|---|---|---|
| `gearbox` | `SpurHelicalGearSystem` | Fully resolved gearbox |
| `shaft_systems` | `list[ShaftSystem]` | Ordered source to last driven |
| `pairs` | `list[GearPairFixture]` | One per stage |
| `gear_fixtures` | `list[tuple[GearFixture, GearFixture]]` | (driver, driven) per stage with resolved x1, x2 |
| `gear_elements` | `list[tuple[GearElement, GearElement]]` | (ge_driver, ge_driven) per stage; same objects used in links |
| `links` | `list[SpurHelicalMeshLink]` | One per stage |
| `label` | `str` | Gearbox label |

Convenience methods: `summary()`, `print_gear_summary()`,
`driver_fixture(i)`, `driven_fixture(i)`,
`driver_element(i)`, `driven_element(i)`

---

## Key Design Contracts

### Immutability

All fixture objects (`GearFixture`, `ShaftFixture`, `BearingFixture`,
`DGBBFixture`) are `frozen=True` dataclasses. Modification always
produces a new object:

```python
gear2  = gear.with_x(0.15)
shaft2 = shaft.with_params(d_body=55.0)
brg2   = brg.with_position(35.0).with_arrangement("floating")
```

### Profile shift

`GearFixture.x` is the declared value. The working profile shift
coefficients `x1`, `x2` are resolved by `SpurHelicalGearMeshing` inside
`GearPairFixture.from_fixtures()`. Access via `pair.x1`, `pair.x2` or
via `result.driver_fixture(i).x` after `build_systems()`.

### Bearing readiness

`DGBBFixture.bearing` does not have internal geometry set. Always call
`make_ready()` on a positioned copy before passing to any solver:

```python
brg = DGBB_6210.with_position(35.0).with_label("A").make_ready()
# brg.has_internal_geometry() is True
# brg.cp is set — ready for IterativeBearingFEMSolver
```

### N-section shafts

`make_shaft(sections, name)` accepts any `list[SectionSpec]`.
`make_stepped_3section(...)` is a convenience wrapper for the canonical
seat | body | seat case. For non-standard geometry:

```python
sections = [
    SectionSpec(length=30, diameter=30, label="seat_A"),
    SectionSpec(length=20, diameter=35, label="step"),
    SectionSpec(length=80, diameter=50,
                shoulder_left=Shoulder(...), label="body"),
    SectionSpec(length=20, diameter=35, label="step"),
    SectionSpec(length=30, diameter=30, label="seat_B"),
]
shaft = make_shaft(sections, name="shaft_1")
```

### Fixture exposure principle

Fixtures organise and expose the program's capabilities; they do not
hide them. Every intermediate object produced during assembly is
accessible from `GearSystemResult` without reconstruction. Downstream
solvers (ISO 6336, ISO/TS 16281, fatigue, convergence) consume objects
directly from the result.

### Fan-out / power split

Out of scope for `linear_gear_chain.py`. Fan-out requires a dedicated
pre-solver that computes `torque_split` values before passing links to
`SpurHelicalGearSystem`. `build_systems()` handles linear chains only.

---

## System Topology Constraints

- Each `SpurHelicalGearSystem` is always a linear chain: shaft_1 -> shaft_2 -> ... -> shaft_n
- n stages produce n+1 shafts and n mesh links
- Non-linear topologies require multiple independent `SpurHelicalGearSystem` instances
- Planetary and differential arrangements are out of scope

---

## Typical Analysis Script Structure

```python
# 1. Declare parameters
stages      = [StageSpec(...), ...]
shaft_specs = [ShaftSpec(...), ...]

def bearing_factory(b_spec, sh_spec):
    return DGBB_6208.with_position(b_spec.position)...make_ready()

# 2. Build and resolve
result = build_systems(stages, shaft_specs, bearing_factory, P_W, rpm_in)

# 3. Add external loads
result.shaft_systems[0].add_load(TorqueLoad(...))   # motor coupling
result.shaft_systems[-1].add_load(RadialLoad(...))  # driven machine

# 4. Close torque equilibrium (if needed)
result.shaft_systems[0].add_load(TorqueLoad(0.0, -T_in, ...))

# 5. Solve FEM (planned: fixtures/solvers/fem_simple.py)
# 6. Solve ISO 16281 (planned: fixtures/solvers/iso16281_coupled.py)
# 7. Output and plots (planned: fixtures/outputs/, fixtures/plots/)
```

---

## Usage Rules

- Each fixture is self-contained and independently importable.
- The dependency chain runs bottom-up: gears / shafts / bearings -> systems
  -> solvers -> outputs / plots -> integration.
- `integration/` files are the only entry points that combine the full
  pipeline. Copy and adjust the `PARAMETERS` block to produce a new
  `design_xxx.py`.
- Do not modify a fixture for a specific case — duplicate and rename.
- Named reference instances must not be mutated. Use the appropriate
  `with_*()` method to produce modified copies.
- `Ca`, `Cf`, `Rq`, `Rz` on gears are not exposed in `make_spur_helical`.
  Callers requiring non-standard values instantiate `SpurHelicalGear`
  directly.
- `surface_ra` on shafts applies uniformly across all sections. For
  section-specific surface treatments, build the `SectionSpec` list
  manually and call `make_shaft()` directly.