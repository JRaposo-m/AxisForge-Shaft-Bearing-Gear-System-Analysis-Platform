# axisforge/mesh — Module Reference

1D node generation, grading and beam elements for the shaft solvers in [`solvers/README.md`](../solvers/README.md#shaft-fem). This package produces the discretisation; it never solves on it.

← back to [project root](../../README.md)

---

## Table of Contents

- [Scope](#scope)
- [Status](#status)
- [Position in the architecture](#position-in-the-architecture)
- [Structure](#structure)
- [Import surface](#import-surface)
- [Conventions](#conventions)
- [Node generation](#node-generation)
- [Elements](#elements)
- [Design contracts](#design-contracts)
- [Extending this package](#extending-this-package)

---

## Scope

| In scope | Out of scope |
|---|---|
| Where the nodes are | What is solved on them |
| Element geometry and material properties | Assembly of the global stiffness matrix |
| The element stiffness matrix and its quadrature | Boundary conditions, load vectors, the solve |
| Standardised refinement levels for a subdomain | Deciding whether a mesh has converged |
| The analysis-level beam model choice (`BeamModelSettings`) | Which theory a *given* solve should use — that is the caller's decision, made once and passed in |

The mesh convergence study lives with the solvers, in `solvers/mesh/`, because it drives repeated solves. This package holds only the geometry-side primitives it refines.

---

## Status

| Group | Module | Contents | Status |
|---|---|---|---|
| `mesh_generation/` | `mesh_1D.py` | `Mesh1D` | Implemented |
| `mesh_generation/` | `mesh_grade.py` | `Grader` | Implemented |
| `element_type/` | `elem.py` | `BeamFormulation` (contract), `ShearDeformableBeamFormulation`, `EulerBernoulliBeam`, `TimoshenkoBeam`, `FrameElement`, `ElemBase`, `Elem`, `QuadraticTimoshenkoElem`, `_FORMULATION_REGISTRY` | Implemented — **substantially restructured**, see [Elements](#elements). `Elem` (2-node) is the only element actually usable today; `QuadraticTimoshenkoElem` is a documented placeholder that raises `NotImplementedError` on construction. |
| `element_type/` | `shear_factor.py` | `ShearFactor` (`cowper_factor`, `hutchinson_factor`, `shear_correction_factor`) | Implemented — `hutchinson_factor` raises `NotImplementedError` for a hollow section (unverified against its source); `cowper_factor` supports both solid and hollow. |
| `shaft/` | `beam_model_settings.py` | `BeamModelSettings`, `VALID_BEAM_THEORIES`, `VALID_SHEAR_THEORIES`, `VALID_INTEGRATION_METHODS` | Implemented |

> **Superseded structure.** An earlier pass of this document described `element_type/timoshenko.py` (`TimoshenkoBeam`) and a reserved, empty `element_type/euler_bernoulli.py`, with `elem.py` defining a single flat `Elem` class. That is gone: `timoshenko.py`/`euler_bernoulli.py` no longer exist as separate files (only their stale `.pyc`s remain), and `elem.py` now holds a full beam-formulation hierarchy plus a registry, described in [Elements](#elements) below. Euler-Bernoulli is **implemented**, not reserved.

---

## Position in the architecture

```
   config    core/                        geometry, materials, loads
      └────────┼───────┐
               ▼       │
            mesh/  ◀───┘                  this package
               │
               ▼
           solvers/
```

`mesh/` reads `core/` (via `axisforge.core.materials.get_material`, and — inside `Mesh1D` — the shaft system's sections, bearings, gears and loads) and `config` for tolerances (`MESH_MIN_NODE_DIST_MM`). Nothing in `core/` reads `mesh/`, and `mesh/` reads no solver.

Within the package the dependency also runs one way: **`Elem`/`ElemBase` read a `Mesh1D` (via `from_mesh()`), and `Mesh1D` knows nothing about elements.** A mesh is a list of positions; giving it knowledge of what will be built on it would make it impossible to refine one without rebuilding the other. `elem.py` also depends downward on its sibling `shear_factor.py` (imported by module, never the reverse) — the same pattern `core/machine_elements/bearings/families/family.py` uses for its own siblings `iso16281_contact.py`/`capacity.py`.

---

## Structure

```
mesh/shaft/
├── beam_model_settings.py      BeamModelSettings — the analysis-level theory choice
├── mesh_generation/
│   ├── mesh_1D.py               Mesh1D — the node grid
│   └── mesh_grade.py            Grader — standardised refinement levels
└── element_type/
    ├── elem.py                  BeamFormulation hierarchy, ElemBase, Elem, QuadraticTimoshenkoElem
    └── shear_factor.py          ShearFactor — Cowper / Hutchinson shear correction
```

Element *geometry* (`Elem`/`ElemBase`) and element *theory* (`BeamFormulation` and its concrete subclasses) are separate on purpose. An element knows its length, section properties and node indices regardless of which beam theory integrates it; the theory is resolved by `(beam_theory, element_order)` against `_FORMULATION_REGISTRY`, the same self-registering pattern used by `core`'s bearing families (`@register_family`) and `solvers`' contact-solver dispatch.

---

## Import surface

| Import from | Names |
|---|---|
| `axisforge.mesh` (rollup) | Everything below, re-exported eagerly, written out explicitly (not `import *`) for grep-ability — see the package's own `__init__.py` |
| `axisforge.mesh.shaft.mesh_generation` | `Mesh1D`, `Grader` |
| `axisforge.mesh.shaft.element_type` | `BeamFormulation`, `ShearDeformableBeamFormulation`, `EulerBernoulliBeam`, `TimoshenkoBeam`, `FrameElement`, `ElemBase`, `Elem`, `QuadraticTimoshenkoElem`, `ShearFactor`, `available_formulations` |
| `axisforge.mesh.shaft.beam_model_settings` | `BeamModelSettings`, `VALID_BEAM_THEORIES`, `VALID_SHEAR_THEORIES`, `VALID_INTEGRATION_METHODS` |

`available_formulations()` returns `{(beam_theory, element_order): concrete_class}` from `_FORMULATION_REGISTRY`, for anything that wants to list or validate what's registered (a settings UI, a test asserting every valid theory has a formulation) without importing the private registry name directly.

---

## Conventions

| Symbol | Meaning |
|---|---|
| `x` | Axial coordinate, increasing left → right, in **mm** |
| Natural coordinate | `zeta` on `[-1, +1]` within an element |
| DOF per node | Three: axial displacement, transverse displacement, rotation |

Node positions are absolute millimetres along the shaft, with `x = 0` at the left face of the first section. Young's modulus is in **MPa**, areas in **mm²**, second moments of area in **mm⁴**.

Each of the two bending planes is solved independently against the same element stiffness matrix, so the element carries no plane of its own.

---

## Node generation

### `Mesh1D`

Generates the 1D node grid for one shaft system. The grid is built from **mandatory positions only** — there is no adaptive refinement here. Every section boundary, bearing extent, gear extent and load position lands on a node; positions closer together than `MESH_MIN_NODE_DIST_MM` are merged.

| Member | Purpose |
|---|---|
| `build` | Computes, or returns the cached, node positions. |
| `x_nodes`, `n_nodes` | The node grid and its size. |
| `add_grader` | Injects a grader's refinement level as additional mandatory nodes. Call it **after** the base mesh exists, since a grader is constructed against the current node list. Invalidates the cache. |
| `clear_graders` | Removes every injected grader. |
| `show_nodes` | Numbered node table for inspection. |

Extra mandatory nodes can also be passed at construction (`extra_mandatory`). This is how a convergence study locks a converged node set into a production run, and how gear-face refinement is applied.

The distinction matters: refinement in AxisForge is always **explicit**. A node exists because something physical is there, or because a grader or a caller asked for it — never because a solver decided mid-run that it wanted one.

### `Grader`

Produces standardised refinement levels for a subdomain by successive elementwise bisection of the base mesh.

| Member | Purpose |
|---|---|
| `get_grade` | Sorted node positions at a requested level. Level zero is the set of base nodes already inside the subdomain; each further level bisects every interval once, so level *N* has 2^*N* intervals per base interval. |

Grades are named `grade_0`, `grade_1`, … and anything not matching that pattern is rejected. The naming is not cosmetic: the Richardson extrapolation in the convergence study depends on a known, constant refinement ratio between consecutive levels, and bisection is what guarantees it.

---

## Elements

### `BeamModelSettings`

`shaft/beam_model_settings.py`. The analysis-level beam model choice — decided once per shaft, not per element. No defaults on purpose: `beam_theory`, `shear_theory` and `integration_method` must all be passed explicitly.

| Field | Constraint |
|---|---|
| `beam_theory` | One of `VALID_BEAM_THEORIES` = `("euler_bernoulli", "timoshenko")` |
| `shear_theory` | Required (one of `VALID_SHEAR_THEORIES` = `("cowper", "hutchinson")`) when `beam_theory="timoshenko"`; must be `None` for `"euler_bernoulli"` |
| `integration_method` | Required (one of `VALID_INTEGRATION_METHODS` = `("single_point", "exact")`) when `beam_theory="timoshenko"`; must be `None` for `"euler_bernoulli"` |

`element_order` (`"linear"` / `"quadratic"`) is **not yet a field on `BeamModelSettings`** — `Elem` defaults it to `"linear"` via `getattr(settings, "element_order", "linear")`, which is what keeps every existing caller working unchanged while the 3-node element remains unimplemented (see `QuadraticTimoshenkoElem` below).

### The `BeamFormulation` hierarchy

`element_type/elem.py`. This is the part of the package that changed most since the previous pass of this document: what used to be a flat `Elem` (2-node, hard-coded to Euler-Bernoulli-or-Timoshenko) plus a separate `TimoshenkoBeam` module is now a small formulation contract with a registry, and a node-topology-agnostic element base.

| Class | Role |
|---|---|
| `BeamFormulation` (ABC) | Contract every bending theory implements: `shape_functions(zeta, elem)`, `bending_strain_matrix(zeta, elem)`, `stiffness_element(elem, **kwargs)`, plus a concrete, shared `natural_coordenates(x, elem)` (global → natural coordinate, identical for every formulation spanning `[x_a, x_b]`). |
| `ShearDeformableBeamFormulation` (ABC, subclass of `BeamFormulation`) | Adds `shear_rigidity(elem, *, kGA_override=None)` and `shear_strain_matrix(zeta, elem, integration)`. A theory is Timoshenko *because* it has an independent shear field, not because of an optional mixin — `ElemBase` dispatches shear-specific calls with `isinstance(formulation, ShearDeformableBeamFormulation)`, never a string comparison on `beam_theory`. |
| `EulerBernoulliBeam` | Concrete, registered as `("euler_bernoulli", "linear")`. Cubic Hermite shape functions, closed-form 4×4 bending stiffness. |
| `TimoshenkoBeam` | Concrete, registered as `("timoshenko", "linear")`. Linear shape functions for `v`; shear strain matrix branches on `integration` (`"single_point"` vs `"exact"`, matching `BeamModelSettings.integration_method`); `shear_rigidity()` calls `ShearFactor.shear_correction_factor()`. |
| `FrameElement` | Axial DOF behaviour (`u_a`, `u_b`) — **not** part of the `BeamFormulation` hierarchy and **not** in the registry: every element has axial behaviour unconditionally, regardless of `beam_theory`, so there is no strategy to select between. A stateless singleton shared by every `ElemBase` subclass. |
| `ElemBase` (ABC) | Node-topology-agnostic contract. Formulation dispatch (`shape_functions`, `bending_strain_matrix`, `stiffness_element`, `shear_rigidity`, `shear_strain_matrix`, `natural_coordenates`) and axial dispatch (`axial_shape_functions`, `axial_strain_matrix`, `axial_stiffness_element`) are concrete and shared here, because none of it reads node indices directly. Only `node_indices` and `validate()` are left abstract — the one thing that genuinely differs between a 2-node and a future 3-node element. |
| `Elem` | The concrete 2-node element used throughout AxisForge today. Behaviourally unchanged from before this split — the formulation dispatch it used to implement directly now lives in `ElemBase`. |
| `QuadraticTimoshenkoElem` | **Design placeholder, not functional.** Subclasses `ElemBase`; `__init__` raises `NotImplementedError` unconditionally. Reserves the slot in the hierarchy for a future 3-node quadratic Timoshenko element without pretending the design is finished — see the four open questions in its own docstring (axial DOF handling at the mid-node, 3-index `node_indices`/`validate()`, `Mesh1D` not yet emitting a mid-node position, and the not-yet-registered `QuadraticTimoshenkoBeam` formulation, which still mutates shared singleton state and needs an `integration="two_point"` case `ElemBase` doesn't know about). This is a documented reservation, not a contradiction with `beam_model_settings.py`'s own note that the 3-noded element "will not be implemented yet" — the two are consistent: the slot exists, the implementation deliberately does not. |

`_FORMULATION_REGISTRY: dict[tuple[beam_theory, element_order], BeamFormulation]` holds one stateless singleton instance per registered `(beam_theory, element_order)` pair, populated by the `@register_formulation(...)` class decorator at import time — `EulerBernoulliBeam` and `TimoshenkoBeam` are the only two entries today, both `element_order="linear"`.

### `Elem`

One 2-node beam element between two mesh nodes, carrying its length, Young's modulus, second moment of area, cross-sectional area, Poisson ratio, `radius_ratio` (inner/outer radius, `0.0` for solid — read by `ShearFactor` for a hollow-section shear correction) and the two node indices.

| Member | Purpose |
|---|---|
| `from_mesh` | Builds the full element list from a `Mesh1D` and a `BeamModelSettings`, resolving section properties and materials. |
| `from_x_nodes` | Builds elements from an explicit node list, for the submodel solver. |
| `find_node_index` | Locates a node within tolerance; raises if none matches. |
| `validate`, `validate_or_raise` | Element sanity — positive length, plausible modulus units, node index/ordering checks, and (for a shear-deformable theory) `shear_theory`/`integration_method` validity, including a check that flags `shear_theory="hutchinson"` combined with a hollow section as not yet implemented. |

`find_node_index` raising rather than returning a sentinel is deliberate. A boundary condition or a bearing constraint applied at a position that is not a node is a modelling error, and it must not be silently applied to the nearest node instead.

### `ShearFactor`

`element_type/shear_factor.py`. Pure math, no knowledge of `Elem`/`BeamFormulation` — imported by module (`import ... as sf`) from `elem.py`, never the reverse.

| Member | Purpose |
|---|---|
| `cowper_factor(v, ratio)` | Cowper (1966) shear correction factor — solid and hollow circular sections both implemented. |
| `hutchinson_factor(v, ratio)` | Hutchinson (2001) shear correction factor — solid section only; raises `NotImplementedError` for a hollow section, flagged in the source as unverified against the original reference. |
| `shear_correction_factor(v, ratio, E, A, theory, *, kGA_override=None)` | Dispatches to one of the two above by `theory`, or — if `kGA_override` is given — back-computes the *effective* shear factor implied by that target `K·G·A`, ignoring `theory` entirely. |

**Adaptive quadrature order**, referenced by the shaft solver's assembly step (`QuadratureOrderEstimator`, in `solvers/`), is what lets the shaft solver integrate an arbitrary distributed load — uniform or callable intensity, constant or callable direction — without the caller choosing a rule. That estimator lives in `solvers/`, not here; this package only supplies the element geometry and stiffness it integrates.

---

## Design contracts

- **The mesh does not know about elements.** `Elem`/`ElemBase` read `Mesh1D`; never the reverse.
- **Refinement is explicit.** Nodes appear because geometry, a grader or a caller put them there. No solver adds a node.
- **Grades are geometric, with a constant ratio.** Bisection only. The convergence study's extrapolation depends on it.
- **A missing node is an error, not a rounding problem.** `find_node_index` raises rather than snapping to the nearest node.
- **Beam theory is a registered formulation, resolved by a declared contract, not a branch.** `ElemBase` never asks "is this Timoshenko?" by comparing strings outside of dispatch; it resolves `(beam_theory, element_order)` against `_FORMULATION_REGISTRY` and, for shear-specific behaviour, checks `isinstance(formulation, ShearDeformableBeamFormulation)`. Adding a formulation is registering a new class, not adding a branch to `Elem`.
- **A design placeholder says so and refuses to run, rather than half-working.** `QuadraticTimoshenkoElem.__init__` raises `NotImplementedError` unconditionally — the slot exists in the hierarchy, but nothing pretends the 3-node element is usable.
- **No solving here.** No boundary conditions, no load vector, no global assembly.

---

## Extending this package

**Adding a beam theory.** Add the class to `element_type/elem.py` (or a sibling module it imports, mirroring how `shear_factor.py` sits beside it), subclass `BeamFormulation` (or `ShearDeformableBeamFormulation` if it has an independent shear field), and decorate it `@register_formulation("your_theory")` — or `@register_formulation("your_theory", "quadratic")` for a non-linear element order. Then register it in `solvers/.../shaft/fem_solvers/element_theories/element_postprocessing.py`'s own dispatch for internal-force recovery (see [`solvers/README.md`](../solvers/README.md#shaft-results-and-post-processing)) — the two registries are separate and both need the new theory. `VALID_BEAM_THEORIES` in `beam_model_settings.py` also needs the new name added, since `BeamModelSettings.__post_init__` validates against it.

**Finishing `QuadraticTimoshenkoElem`.** Read its class docstring first — the four open questions (axial DOF at the mid-node, 3-index node validation, `Mesh1D` mid-node placement, and the unregistered/unfixed `QuadraticTimoshenkoBeam` formulation) are the actual blockers, not just "register it".

**Adding a refinement strategy.** A non-bisecting grader is possible, but the convergence study assumes a constant refinement ratio between consecutive levels. A new strategy must either preserve that or state its own ratio for the extrapolation to use.

**Adding a mandatory-position source.** Extend `_mandatory_positions` in `Mesh1D`. Anything with a physical axial extent belongs there; anything that is a numerical preference belongs in a grader or in the caller's extra mandatory nodes.
