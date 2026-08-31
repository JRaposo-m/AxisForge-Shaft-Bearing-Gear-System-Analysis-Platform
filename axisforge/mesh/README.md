# axisforge/mesh — Module Reference

1D node generation, grading and beam elements for the shaft solver in [`solvers/README.md`](../solvers/README.md#shaft-fem).

← back to [project root](../../README.md)

---

## Import surface

| Import from | Names |
|---|---|
| `axisforge.mesh.shaft.mesh_generation` | `Mesh1D`, `Grader` |
| `axisforge.mesh.shaft.element_type` | `TimoshenkoBeam` |
| `axisforge.mesh.shaft.element_type.elem` | `Elem` |

```
mesh/shaft/
├── mesh_generation/
│   ├── mesh_1D.py       Mesh1D
│   └── mesh_grade.py    Grader
└── element_type/
    ├── elem.py          Elem
    └── timoshenko_selective_integration/
        └── timoshenko.py    TimoshenkoBeam
```

The mesh convergence study lives with the solvers, in `solvers/mesh/`, because it drives repeated solves. This package holds only the geometry-side primitives it refines.

---

## Node generation

### `Mesh1D`

Generates the 1D node grid for one shaft system. The grid is built from mandatory positions only — there is no adaptive refinement here. Every section boundary, bearing extent, gear extent and load position lands on a node; positions closer together than the minimum node distance are merged.

| Member | Purpose |
|---|---|
| `build` | Computes, or returns the cached, node positions. |
| `x_nodes`, `n_nodes` | The node grid and its size. |
| `add_grader` | Injects a grader's refinement level as additional mandatory nodes. Call it after the base mesh exists, since a grader is constructed against the current node list. Invalidates the cache. |
| `clear_graders` | Removes every injected grader. |
| `show_nodes` | Numbered node table for inspection. |

Extra mandatory nodes can also be passed at construction — this is how the convergence study locks a converged node set into a production run, and how gear-face refinement is applied.

### `Grader`

Produces standardised refinement levels for a subdomain by successive elementwise bisection of the base mesh.

| Member | Purpose |
|---|---|
| `get_grade` | Sorted node positions at a requested level. Level zero is the set of base nodes already inside the subdomain; each further level bisects every interval once, so level N has 2^N intervals per base interval. |

Grades are named `grade_0`, `grade_1`, … and anything not matching that pattern is rejected.

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

The dependency runs one way: `Elem` reads a `Mesh1D`, and `Mesh1D` knows nothing about elements.

### `TimoshenkoBeam`

Timoshenko beam element with selective integration.

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

Adaptive quadrature order is what lets the FEM solver integrate an arbitrary distributed load — uniform or callable intensity, constant or callable direction — without the caller choosing a rule.