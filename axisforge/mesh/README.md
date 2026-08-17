# axisforge/mesh — Module Reference

1D FEM node generation, beam elements, and the mesh convergence study for the shaft solver in [`solvers/README.md`](../solvers/README.md#solvers--shaft-fem).

← back to [project root](../../README.md)

---

## mesh — 1D Shaft Mesh

**`mesh_1D.py` — `Mesh1D`**
Generates the 1D FEM node grid for one `ShaftSystem` from mandatory positions (section boundaries, bearings, gears, loads) plus optional `extra_mandatory` nodes, enforcing a minimum node spacing.

**`mesh_grade.py` — `Grader`**
Produces standardised mesh grades for a subdomain [x_lo, x_hi] by successive elementwise bisection (`grade_0` = base nodes, `grade_N` = N bisections). Consumed by the convergence study and by gear-face refinement.
- `get_grade("grade_N")` — sorted node positions at the requested refinement level.

**`Elements/elem.py` — `Elem`**
Single 1D beam element between two mesh nodes (length, E, I, A, Poisson ν, node indices).
- `from_mesh(mesh)` — builds the full element list from a `Mesh1D`, reading section properties and materials.
- `from_x_nodes(x_nodes, shaft_system)` — builds elements from an explicit node list (used by the submodel solver).
- `find_node_index(x_nodes, x)` — locate a node within tolerance.
- `validate()` — element sanity (positive length, plausible modulus units).

**`Elements/Timoshenko_Selective_Integration/timoshenko.py` — `TimoshenkoBeam`**
Timoshenko beam element with selective integration (shear factor 5/6).
- `stiffness_element(elem)` — 6×6 element stiffness (axial + shear + bending).
- `shape_functions`, `deformation_matrix`, `elasticity_matrix` — element interpolation and constitutive matrices.
- Natural-coordinate mapping helpers for distributed-load integration.

**`mesh_convergence_study.py` — `RichardsonGCI`, `MeshConvergenceStudy`**
Grid Convergence Index via Richardson extrapolation on the resultant transverse displacement, across ≥3 refinement levels. Produces per-load convergence records and the union of extra nodes to lock into production runs. Includes `print_report`.

Bearing extents are excluded automatically from refinement intervals (prescribed BC, no physical gain). GCI threshold: 1% default; safety factor: 1.25. Uses `SubmodelSolver` (see [`solvers/README.md`](../solvers/README.md#solvers--shaft-fem)) to evaluate metrics restricted to each refined subdomain.
