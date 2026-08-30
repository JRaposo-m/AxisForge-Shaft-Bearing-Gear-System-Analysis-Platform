# axisforge/mesh — Module Reference

1D FEM node generation, grading, and beam elements for the shaft solver in [`solvers/README.md`](../solvers/README.md#solvers--shaft-fem).

← back to [project root](../../README.md)

---

## Package layout

The `oneD/` level is gone — the package is `mesh/shaft/`, split into node generation and element formulation. The **mesh convergence study has moved out** to [`solvers/mesh/mesh_convergence_study.py`](../solvers/README.md#solvers--mesh-convergence): it drives solves, so it belongs with the solvers; `mesh/` now contains only the geometry-side primitives it refines.

```
mesh/
└── shaft/
    ├── mesh_generation/
    │   ├── __init__.py         ← Mesh1D, Grader
    │   ├── mesh_1D.py
    │   └── mesh_grade.py
    └── element_type/
        ├── __init__.py         ← TimoshenkoBeam
        ├── elem.py             ← Elem
        └── timoshenko_selective_integration/
            ├── __init__.py     ← TimoshenkoBeam
            └── timoshenko.py
```

| Was | Is now |
|---|---|
| `mesh/oneD/shaft/mesh_generation/` | `mesh/shaft/mesh_generation/` |
| `mesh/oneD/shaft/Elements/elem.py` | `mesh/shaft/element_type/elem.py` |
| `mesh/oneD/shaft/Elements/Timoshenko_Selective_Integration/` | `mesh/shaft/element_type/timoshenko_selective_integration/` |
| `mesh/oneD/shaft/mesh_generation/mesh_convergence_study.py` | `solvers/mesh/mesh_convergence_study.py` |

> `element_type/__init__.py` exports only `TimoshenkoBeam`. `Elem` is **not** rolled up there — it is imported directly from `.elem` by `SimpleFEMSolver`, `SubmodelSolver` and the convergence study. Add it to the roll-up if you want `element_type` to be the single import point for both.

---

## mesh/shaft/mesh_generation

**`mesh_1D.py` — `Mesh1D`**

Generates the 1D FEM node grid for one `ShaftSystem`. The mesh is built from **mandatory positions only** (no adaptive refinement): every section boundary, bearing extent, gear extent and load position must land on a node.

- `Mesh1D(shaft_system, extra_mandatory=None)` — `extra_mandatory` is the hook the convergence study and gear-face refinement use to lock extra nodes in.
- `build()` / `x_nodes` *(property)* / `n_nodes` — sorted, deduped node grid; positions closer than `MESH_MIN_NODE_DIST_MM` are merged.
- `add_grader(grader, grade)` — inject a `Grader` + grade level as mandatory nodes. Call **after** the base mesh exists, since the `Grader` must have been constructed against the current `x_nodes`. Invalidates the cache.
- `clear_graders()` — remove all injected graders.
- `show_nodes(print_output=True)` — numbered node table for inspection.

**`mesh_grade.py` — `Grader`**

Produces standardised mesh grades for a subdomain `[x_lo, x_hi]` by successive elementwise bisection of the base mesh.

- `Grader(x_lo, x_hi, x_nodes)`.
- `get_grade("grade_N")` — sorted node positions at refinement level N. `grade_0` is the set of base nodes already inside the subdomain; each further grade bisects every interval once, so `grade_N` has `2^N` intervals per base interval.
- Internals: `_parse_grade` (rejects anything not matching `grade_<int>`), `_base_nodes`, `_bisect_once`.

Consumed by the convergence study and by gear-face refinement (`SimpleFEMSolver(extra_mandatory=Grader(lo, hi, base_nodes).get_grade("grade_3"))`).

---

## mesh/shaft/element_type

**`elem.py` — `Elem`**

Single 1D beam element between two mesh nodes: `length`, `E`, `I`, `A`, `v` (Poisson), `idx_node_1`, `idx_node_2`.

- `Elem.from_mesh(mesh, node_tol=MESH_MIN_NODE_DIST_MM)` *(classmethod)* — builds the full element list from a `Mesh1D`, reading `mesh.shaft_system` and `mesh.x_nodes` directly and resolving section properties and materials (`get_material`). `Mesh1D` itself carries no knowledge of `Elem`; the dependency runs one way only.
- `Elem.from_x_nodes(x_nodes, shaft_system)` *(classmethod)* — builds elements from an explicit node list (used by the submodel solver, where the node set is not a `Mesh1D`).
- `Elem.find_node_index(x_nodes, x, tol=MESH_MIN_NODE_DIST_MM)` *(staticmethod)* — locate a node within tolerance; raises `ValueError` if none matches.
- `validate()` / `validate_or_raise()` — positive length, plausible modulus units.

**`timoshenko_selective_integration/timoshenko.py` — `TimoshenkoBeam`**

Timoshenko beam element with selective integration (shear factor 5/6).

- `stiffness_element(elem)` — 6×6 element stiffness (axial + shear + bending).
- `shape_functions(zeta, elem)`, `deformation_matrix(zeta, elem)`, `elasticity_matrix(elem)` — interpolation and constitutive matrices in natural coordinates.
- `global_to_natural_radial(x1, x2, elem)` → the mapping `ζ ↦ x(ζ)` over `ζ ∈ [−1, 1]`; `vetor_global_to_natural(f, x_map)` composes any `f(x)` with it; `jacobian(elem)`.
- `gauss_quadrature(n)` — Gauss–Legendre points and weights (`numpy.polynomial.legendre` for n > 3).
- `gauss_order(q, x_lo, x_hi, elem, theta_fn=None)` — **adaptive** quadrature order: estimates the polynomial degree of the shape functions (`shape_function_degree`), of the load intensity `q(x)` (`_q_degree`) and of `cos(theta(x))` (`_theta_degree`) by sampling and polynomial fitting, then returns the order that integrates their product exactly.

This is what lets `SimpleFEMSolver._assemble_distributed_load_vector` integrate an arbitrary `DistributedRadialLoad` — uniform or callable intensity, constant or callable direction — without the caller choosing a quadrature rule.