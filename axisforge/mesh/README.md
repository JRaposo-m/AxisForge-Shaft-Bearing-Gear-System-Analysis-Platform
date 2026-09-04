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

The mesh convergence study lives with the solvers, in `solvers/mesh/`, because it drives repeated solves. This package holds only the geometry-side primitives it refines.

---

## Status

| Group | Module | Contents | Status |
|---|---|---|---|
| `mesh_generation/` | `mesh_1D.py` | `Mesh1D` | Implemented |
| `mesh_generation/` | `mesh_grade.py` | `Grader` | Implemented |
| `element_type/` | `elem.py` | `Elem` | Implemented |
| `element_type/` | `timoshenko.py` | `TimoshenkoBeam` | Implemented |
| `element_type/` | `euler_bernoulli.py` | Euler–Bernoulli beam element | Reserved — module exists, defines nothing |

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

`mesh/` reads `core/` — a shaft's sections, a system's bearings, gears and loads — and `config` for tolerances. Nothing in `core/` reads `mesh/`, and `mesh/` reads no solver.

Within the package the dependency also runs one way: **`Elem` reads a `Mesh1D`, and `Mesh1D` knows nothing about elements.** A mesh is a list of positions; giving it knowledge of what will be built on it would make it impossible to refine one without rebuilding the other.

---

## Structure

```
mesh/shaft/
├── mesh_generation/
│   ├── mesh_1D.py           Mesh1D — the node grid
│   └── mesh_grade.py        Grader — standardised refinement levels
└── element_type/
    ├── elem.py              Elem — one element's geometry and material
    ├── timoshenko.py        TimoshenkoBeam — selective integration
    └── euler_bernoulli.py   Reserved
```

Element *geometry* (`Elem`) and element *theory* (`TimoshenkoBeam`) are separate on purpose. An element knows its length, section properties and node indices regardless of which beam theory will integrate it; the theory is selected by name in `StiffnessMatrixBuilder`, and `euler_bernoulli` is the reserved second entry in that registry.

---

## Import surface

| Import from | Names |
|---|---|
| `axisforge.mesh.shaft.mesh_generation` | `Mesh1D`, `Grader` |
| `axisforge.mesh.shaft.element_type` | `TimoshenkoBeam` |
| `axisforge.mesh.shaft.element_type.elem` | `Elem` |

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

Generates the 1D node grid for one shaft system. The grid is built from **mandatory positions only** — there is no adaptive refinement here. Every section boundary, bearing extent, gear extent and load position lands on a node; positions closer together than the minimum node distance are merged.

| Member | Purpose |
|---|---|
| `build` | Computes, or returns the cached, node positions. |
| `x_nodes`, `n_nodes` | The node grid and its size. |
| `add_grader` | Injects a grader's refinement level as additional mandatory nodes. Call it **after** the base mesh exists, since a grader is constructed against the current node list. Invalidates the cache. |
| `clear_graders` | Removes every injected grader. |
| `show_nodes` | Numbered node table for inspection. |

Extra mandatory nodes can also be passed at construction. This is how a convergence study locks a converged node set into a production run, and how gear-face refinement is applied.

The distinction matters: refinement in AxisForge is always **explicit**. A node exists because something physical is there, or because a grader or a caller asked for it — never because a solver decided mid-run that it wanted one.

### `Grader`

Produces standardised refinement levels for a subdomain by successive elementwise bisection of the base mesh.

| Member | Purpose |
|---|---|
| `get_grade` | Sorted node positions at a requested level. Level zero is the set of base nodes already inside the subdomain; each further level bisects every interval once, so level *N* has 2^*N* intervals per base interval. |

Grades are named `grade_0`, `grade_1`, … and anything not matching that pattern is rejected. The naming is not cosmetic: the Richardson extrapolation in the convergence study depends on a known, constant refinement ratio between consecutive levels, and bisection is what guarantees it.

---

## Elements

### `Elem`

One 1D beam element between two mesh nodes, carrying its length, Young's modulus, second moment of area, cross-sectional area, Poisson ratio and the two node indices.

| Member | Purpose |
|---|---|
| `from_mesh` | Builds the full element list from a `Mesh1D`, resolving section properties and materials. |
| `from_x_nodes` | Builds elements from an explicit node list, for the submodel solver. |
| `find_node_index` | Locates a node within tolerance; raises if none matches. |
| `validate`, `validate_or_raise` | Element sanity — positive length, plausible modulus units. |

`find_node_index` raising rather than returning a sentinel is deliberate. A boundary condition or a bearing constraint applied at a position that is not a node is a modelling error, and it must not be silently applied to the nearest node instead.

### `TimoshenkoBeam`

Timoshenko beam element with selective integration — the shear term is under-integrated relative to the bending term, which is what avoids shear locking on slender elements.

| Member | Purpose |
|---|---|
| `stiffness_element` | The element stiffness matrix — axial, shear and bending. |
| `shape_functions` | Element interpolation functions in natural coordinates. |
| `deformation_matrix`, `elasticity_matrix` | Strain–displacement and constitutive matrices. |
| `global_to_natural_radial` | The mapping from natural to global axial coordinate over an element. |
| `vetor_global_to_natural` | Rewrites any function of the global coordinate in natural coordinates. |
| `jacobian` | Element Jacobian. |
| `gauss_quadrature` | Gauss–Legendre points and weights for an arbitrary order. |
| `shape_function_degree` | Estimated polynomial degree of a shape function. |
| `gauss_order` | The quadrature order that integrates the product of shape function, load intensity and direction exactly, estimated by sampling and polynomial fitting. |

**Adaptive quadrature order** is what lets the shaft solver integrate an arbitrary distributed load — uniform or callable intensity, constant or callable direction — without the caller choosing a rule. The order is derived from the actual functions in hand rather than assumed, so a load that is polynomial is integrated exactly and one that is not is integrated at an order chosen from its own sampled behaviour.

---

## Design contracts

- **The mesh does not know about elements.** `Elem` reads `Mesh1D`; never the reverse.
- **Refinement is explicit.** Nodes appear because geometry, a grader or a caller put them there. No solver adds a node.
- **Grades are geometric, with a constant ratio.** Bisection only. The convergence study's extrapolation depends on it.
- **A missing node is an error, not a rounding problem.** `find_node_index` raises rather than snapping to the nearest node.
- **Element geometry and beam theory are separate concerns.** An `Elem` is theory-agnostic; the theory is selected in the solver's registry.
- **No solving here.** No boundary conditions, no load vector, no global assembly.

---

## Extending this package

**Adding a beam theory.** Add the module under `element_type/`, implementing at minimum `stiffness_element(elem)`, and register it in `StiffnessMatrixBuilder`'s theory registry in `solvers/.../shaft/fem_solvers/build_stiffness_matrix.py`. Take an `Elem` and return a matrix; do not read a `Mesh1D` directly. `euler_bernoulli.py` is the reserved slot for the second entry.

**Adding a refinement strategy.** A non-bisecting grader is possible, but the convergence study assumes a constant refinement ratio between consecutive levels. A new strategy must either preserve that or state its own ratio for the extrapolation to use.

**Adding a mandatory-position source.** Extend `_mandatory_positions` in `Mesh1D`. Anything with a physical axial extent belongs there; anything that is a numerical preference belongs in a grader or in the caller's extra mandatory nodes.