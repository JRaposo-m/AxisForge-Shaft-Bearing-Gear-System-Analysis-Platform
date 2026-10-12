# axisforge.mesh

One-dimensional node grid, refinement grades and beam elements for the shaft FEM. The package
produces the discretisation and the element matrices. It does not assemble the global system,
apply boundary conditions or loads, solve, or judge convergence; those belong to `solvers`.

## Contents

```
mesh/
    shaft/
        beam_model_settings.py     BeamModelSettings: beam theory, shear theory, integration rule
        mesh_generation/
            mesh_1D.py             Mesh1D: node grid of one ShaftSystem; MergeEvent
            mesh_grade.py          Grader: refinement grades by bisection; RefinementFloorReached
        element_type/
            elem.py                beam formulations, element base class, Elem (2-node)
            shear_factor.py        ShearFactor: Cowper and Hutchinson shear correction factors
        tests/                     tests of the package
```

Names exported from `axisforge.mesh`: `BeamModelSettings`, `VALID_BEAM_THEORIES`,
`VALID_SHEAR_THEORIES`, `VALID_INTEGRATION_METHODS`, `Mesh1D`, `Grader`, `Elem`, `ElemBase`,
`BeamFormulation`, `ShearDeformableBeamFormulation`, `EulerBernoulliBeam`, `TimoshenkoBeam`,
`FrameElement`, `QuadraticTimoshenkoElem`, `ShearFactor`, `available_formulations`.
`MergeEvent`, `RefinementFloorReached` and `register_formulation` are imported from their modules.

## Entry points

| Task | Call | Returns |
|---|---|---|
| Choose the beam model | `BeamModelSettings(beam_theory, shear_theory, integration_method)`; no defaults | frozen dataclass; `ValueError` for an invalid combination |
| Build the node grid | `Mesh1D(shaft_system, extra_mandatory=None, protected=None)`, then `.build()` or `.x_nodes` | node positions `list[float]` [mm] |
| Inspect the grid | `.n_nodes`, `.show_nodes()`, `.merge_report()`, `.validate()`, `.validate_or_raise()` | node count; `show_nodes` prints a table and returns `list[tuple[int, float]]`; `list[MergeEvent]`; `list[str]`; `None` |
| Refine a subdomain | `Grader(x_lo, x_hi, x_nodes).get_grade("grade_N")` | node positions `list[float]` [mm] |
| Inject a grade | `Mesh1D.add_grader(grader, grade)`, `Mesh1D.clear_graders()` | `None`; the cached grid is invalidated |
| Build the elements | `Elem.from_mesh(mesh, settings)` | `list[Elem]` in ascending order |
| Elements from node positions | `Elem.from_x_nodes(x_nodes, shaft_system, settings)` | `list[Elem]`, one per interval |
| Locate a node | `Elem.find_node_index(x_nodes, x, tol=NODE_LOOKUP_TOL_MM)` | index; `ValueError` if no node lies within `tol` |
| Element matrices | `Elem.stiffness_element()`, `.axial_stiffness_element()` | (4, 4) bending matrix; (2, 2) axial matrix |
| Shape and strain matrices | `Elem.shape_functions(zeta)`, `.bending_strain_matrix(zeta)`, `.shear_strain_matrix(zeta)`, `.shear_rigidity()` | arrays; shear rigidity K G A [N] |
| Shear correction factor | `ShearFactor().shear_correction_factor(v, ratio, E, A, theory="cowper", *, kGA_override=None)` | k [-] |
| List the formulations | `available_formulations()` | `{(beam_theory, element_order): class}` |

Example:

```python
from axisforge.mesh import BeamModelSettings, Elem, Grader, available_formulations

settings = BeamModelSettings(beam_theory="timoshenko", shear_theory="cowper",
                             integration_method="single_point")
elem = Elem(length=50.0, E=210_000.0, I=1.0e5, A=500.0, v=0.3,
            idx_node_1=0, idx_node_2=1, x_a=0.0, x_b=50.0, settings=settings)
print(elem.stiffness_element().shape)                                  # (4, 4)
print(Grader(0.0, 100.0, [0.0, 50.0, 100.0]).get_grade("grade_1"))     # [0.0, 25.0, ..., 100.0]
print(sorted(available_formulations()))     # [('euler_bernoulli', 'linear'), ('timoshenko', 'linear')]
```

Class and method detail, including the raised errors and the merge rules of `Mesh1D`, is in the
docstrings of `mesh_1D.py`, `mesh_grade.py`, `elem.py` and `shear_factor.py`.

## Conventions

Axes and signs of the whole system are in the root README, section Conventions.

- Lengths and node positions in mm along the shaft axis, with `x = 0` at the left face of the first
  section.
- Element data: Young's modulus E [MPa], second moment of area I [mm^4], area A [mm^2],
  Poisson ratio v [-], length [mm]. `Elem.validate()` flags E < 1000 as a probable GPa value.
- Bending degrees of freedom per element `[v_a, theta_a, v_b, theta_b]`, v [mm], theta [rad],
  with theta = dv/dx for Euler-Bernoulli; for Timoshenko v and theta are independent and the
  shear strain is gamma = dv/dx - theta. Axial degrees of freedom `[u_a, u_b]`. Bending and
  axial matrices are separate (4 x 4 and 2 x 2).
- Natural coordinate `zeta = 2 (x - x_c) / length` in [-1, 1]. The method is spelled
  `natural_coordenates` in the code.
- Each bending plane uses the same element matrix; the element carries no plane.
- Tolerances from `axisforge.config`: `MESH_MIN_NODE_DIST_MM` (0.1 mm, minimum distance between
  nodes), `NODE_LOOKUP_TOL_MM` (1e-6 mm, two positions closer than this are the same node),
  `SOLVER_TOLERANCE` (length tolerance in `Grader`). They are values of this code, not from a
  source.
- Mandatory nodes come from the physical model: section starts and shaft end, bearing centre and
  extent, gear centre and face limits, load positions, distributed-load limits and centroids,
  and the `protected` positions. Extra nodes (`extra_mandatory`, graders) are merged away when
  within `MESH_MIN_NODE_DIST_MM` of another node. Two mandatory nodes within that distance
  (and farther apart than `NODE_LOOKUP_TOL_MM`) raise `ValueError` in `build`. There is no maximum element length.
- Grades are named `grade_N`, N an integer >= 0. Each grade bisects every interval of the
  previous one once (ratio 2). `Grader` raises `RefinementFloorReached` (a `ValueError`) when a
  bisection would give an element of `MESH_MIN_NODE_DIST_MM` or less.
- `BeamModelSettings`: `euler_bernoulli` requires `shear_theory=None` and
  `integration_method=None`; `timoshenko` requires both, from `VALID_SHEAR_THEORIES`
  (`cowper`, `hutchinson`) and `VALID_INTEGRATION_METHODS` (`single_point`, `exact`).
- A formulation is selected by `(beam_theory, element_order)` through a registry filled by
  `@register_formulation`. Only the linear 2-node element is reachable: `element_order` is not a
  field of `BeamModelSettings` and `Elem` reads it with a default of `"linear"`.
  `QuadraticTimoshenkoElem` raises `NotImplementedError`.
- `Mesh1D` does not know the elements; `Elem` reads a `Mesh1D`. `ShearFactor` does not know
  `Elem`.
- Layer rule: `mesh` imports from `axisforge.core` and `axisforge.config` only.
- `Elem` carries `radius_ratio` (inner/outer radius, 0.0 for solid). The factories
  `from_mesh` and `from_x_nodes` leave it at 0.0, so a hollow section receives the solid-section
  shear factor; see the `ShearFactor` docstring.

## Standards

| Key | Used for |
|---|---|
| Onate2013_vol2 | beam theories, shape functions, shear strain, shear correction factor, integration rules |
| Cao2004 | shear correction factor for circular sections (comparison in `ShearFactor`) |

Section, equation and table numbers are in the docstrings.
