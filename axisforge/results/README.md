# axisforge.results

Result containers of AxisForge: the data shapes that the solvers fill and that readers (for
example `axisforge.outputs`) consume. A container stores values, units and shapes and checks
what it can check at construction. The package computes nothing, runs no solver and does not
read the model objects; the equations are in `axisforge.solvers`.

## Contents

```
results/
    __init__.py                  eager re-export of the 14 public names
    _base.py                     internal: DC dataclass options, abstract marker, shape and finite checks
    fem_results/
        shaft_results.py         ShaftResults, BearingNodeData (global shaft FEM)
        submodel_results.py      SubmodelResult (one sub-model solve, one mesh grade)
    convergence_results/
        convergence_results.py   ConvergenceRecord, MeshRefinementResult (mesh convergence study)
    bearings/
        load_distribution/
            load_distribution_results.py   row and bearing load distribution, ball and roller
        life/
            basic_life_results.py          BasicReferenceRatingLifeResult
        bearing_analysis_result.py         BearingAnalysisResult (distribution and life of one bearing)
```

## Entry points

The names are importable from `axisforge.results` and from `axisforge`. A user normally
reads a container returned by a solver. The classes are described in their docstrings.

| Task | Call | Returns |
|---|---|---|
| Global shaft FEM | `RigidSupportFEMSolver.solve` (`...shaft.fem_solvers.global_solver.rigid_support`), which calls `build_shaft_result` (`...global_solver.global_postprocessing`, also exported by `axisforge.solvers.machine_elements.shaft.fem_solvers`) | `ShaftResults`, with `bearing_nodes: list[BearingNodeData]` |
| Sub-model FEM | `build_submodel_result(solution)` (`...fem_solvers.submodel_solver.submodel_postprocessing`) | `SubmodelResult` |
| Mesh convergence | `MeshConvergenceStudy.new_record`, `.add_level` fill a `ConvergenceRecord`; `build_mesh_refinement_result(shaft_name, per_load)` (`axisforge.solvers.mesh.convergence_solver`) | `MeshRefinementResult` of `ConvergenceRecord` |
| Bearing analysis | `SolverBase.solve(shaft_system, bearings, shaft_results, psi_override=None)` (`axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.contact_solver`) | `dict[label, BearingAnalysisResult]` |
| Load distribution of a bearing | `BearingAnalysisResult.load_distribution` | `BallBearingResult` or `RollerBearingResult` |
| Rows of a bearing | `BearingResult.rows`, `.row` (single row only) | tuple of `BallLoadDistributionResult` or `RollerLoadDistributionResult` |
| Basic reference rating life | `BearingAnalysisResult.basic_life` | `BasicReferenceRatingLifeResult` or `None` |
| Records, text and CSV | `axisforge.outputs.solvers` | see `axisforge/outputs/README.md` |

The bearing solvers fill `basic_life` when they are built with `postprocess=True`. Only the
single-row ball and roller solvers do so.

A bearing container is built with `BearingResult.single(row, **shared)` or
`BearingResult.multirow(rows, **shared)`. The abstract classes `LoadDistributionResult`,
`LineContactLoadDistributionResult` and `BearingResult` raise `TypeError` when instantiated.

```python
import numpy as np
from axisforge.results import (BallBearingResult, BallLoadDistributionResult,
                               BearingAnalysisResult)

Q = np.array([800.0, 500.0, 0.0, 0.0, 0.0, 0.0, 0.0, 500.0])
row = BallLoadDistributionResult(
    phi_j=np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False),
    delta_j=np.where(Q > 0.0, 0.01, 0.0), alpha_j=np.zeros(8),
    Q_j=Q, Fr_row=1800.0, Fa_row=0.0, Mz=0.0)
bearing = BallBearingResult.single(
    row, label="A", delta_r=0.01, delta_a=0.0, psi=0.0, phi_Fr=np.pi / 2,
    Fr_xz=0.0, Fr_xy=1800.0, Fa=0.0, n_iter=12, residual=1e-8, ok=True)
analysis = BearingAnalysisResult(label="A", load_distribution=bearing)
print(bearing.n_rows, bearing.row.n_loaded, bearing.f_a)   # 1 3 None
```

## Conventions

* Frames and signs: see the section Conventions of the root README. The axial coordinate is `x`. The suffix `_xz` is the
  component along +Z and `_xy` the component along +Y; an unsuffixed name is the resultant
  (`hypot`). Axial force is positive along +X.
* Units: length and displacement in mm, force in N, bending moment in N mm, stress in MPa,
  section modulus in mm^3, rotation and angle in rad, stiffness in N/mm, `L10r` in 10^6 rev,
  `Pref` in N. The one exception is the torque `T` (and `T_total`) of `ShaftResults`, in N m.
  Angles are in rad in every container.
* `Fr_xz`, `Fr_xy` and `Fa` of `BearingNodeData` are rows of `K d` at the bearing node (support
  reaction plus nodal load): the force of the support on the shaft, positive along +Z, +Y and
  +X. The load on the bearing is the opposite. `R_xz`, `R_xy`, `R`, `R_axial` of `ShaftResults`
  are `K d - f_ext`. `M_xz`, `M_xy` of `BearingNodeData` are nodal moments, not reactions.
* `psi_xz`, `psi_xy` of `BearingNodeData` hold the same value as `theta_xz`, `theta_xy`.
* `V` is the force on the +x face along +v and `dM/dx = -V`.
* At a change of section, `d`, `W`, `Wt`, `sigma_b` and `tau` are those of the section to the
  right of the node. The stresses are nominal (no stress concentration).
* Bearing frames: `phi_Fr = arctan2(Fr_xy, Fr_xz)` is zero along +Z and positive toward +Y
  (`phi_Fr = pi/2 - theta`, with the load angle `theta` of the root README). `phi_j` of a row
  is local (0 on the radial load line); `phi_j_global(i) = (phi_j + phi_Fr) mod 2 pi`, in
  [0, 2 pi). The tilt of the solver is `psi = psi_xz cos(phi_Fr) + psi_xy sin(phi_Fr)`.
* `Mz` is the symbol of ISO_TS_16281_2008 (tilting moment in the plane of the load line), not a
  moment about the global Z axis.
* `delta_j` of a ball is never negative; that of a roller (Eq. (38)) is not clamped and is
  negative for a roller out of contact. `Q_j >= 0` is checked.
* `n_iter` holds `nfev`, the number of residual evaluations of the root find.
* `f_r` and `f_a` of `BearingResult` are `None` when the total load is about 0; read
  `rows[i].Fa_row` then.
* A `RollerBearingResult` is radial only: `Fa`, `delta_a` and every `Fa_row` are 0.0, and
  construction raises otherwise.
* Dataclass options: the bearing containers (`LoadDistributionResult` and subclasses,
  `BearingResult` and subclasses, `BasicReferenceRatingLifeResult`, `BearingAnalysisResult`)
  are frozen, keyword-only and `eq=False`; compare arrays with `np.allclose`. `with_stiffness`
  returns a copy. `ShaftResults`, `BearingNodeData`, `SubmodelResult`, `ConvergenceRecord`
  and `MeshRefinementResult` are plain mutable dataclasses; most fields have defaults.
* Validation in `__post_init__` is done by the bearing containers only: array shapes, finite
  scalars, `Q_j >= 0`, `Q_j == q_jk.sum(axis=1)` of a roller row, non-empty `rows`, equal labels,
  equal number of rows of load distribution and life; for the life, `L10r` finite and
  positive, one row type, and `L10r` not above the weakest row. The FEM and convergence
  containers validate nothing.
* A solve that did not converge is still stored: check `ok` and `residual`.
* Runtime imports are NumPy, the standard library and `axisforge.results._base`.
  `TYPE_CHECKING`-only imports from `axisforge.solvers` serve annotations. `core` and `mesh`
  do not import `results`.

## Standards

* ISO_TS_16281_2008: load distribution of ball and roller bearings and basic reference rating
  life L10r (clause 3; 4.2; 4.3.3-4.3.4; 5.2; 5.3.5-5.3.6). The modified reference rating life
  is not implemented.
* Onate2013_vol2: bending moment and shear force of the beam formulations
  (`ShaftResults`: Eq. (1.5), (1.16c), (1.24), (2.4b), (2.17), Fig. 1.2).
* Zaretsky2016: series-system bound on the combined life (Eq. (47), (49a), Appendix C (C11)).
* ISO_TR_1281: form of the catalogue life (Part 1, clause 8); the catalogue life is not
  implemented.
