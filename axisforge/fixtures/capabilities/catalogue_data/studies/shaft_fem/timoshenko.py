"""axisforge/fixtures/capabilities/catalogue_data/studies/shaft_fem/timoshenko.py

Timoshenko-specific leaves for CATALOGUE["studies"]["shafts"]["shaft_fem"]:
the rigid-bearing solve itself, plus its three convergence-study
variants (all pinned to Timoshenko today). See catalogue.py's own
module docstring for the leaf shape and `requires` token rules, and
this sub-package's __init__.py for why integration_method/shear_theory
are not split into their own capabilities here.
"""

from __future__ import annotations

TIMOSHENKO: dict = {
    "shaft_fem.timoshenko_rigid": {
        "description": (
            "1D Timoshenko beam FEM solve for every shaft in a "
            "SpurHelicalGearSystem (RigidBearingFEMSolver + "
            "ShaftResultsReader via "
            "fixtures/studies/shafts/fem_simple.py's "
            "solve_system()) -- every bearing a rigid support "
            "(v=0 always; u=0 additionally for 'locating' "
            "bearings). RigidBearingFEMSolver's own "
            "constraint_bearing parameter is not yet wired to "
            "any alternative, so this name describes the only "
            "physics actually applied today, not a chosen "
            "switch."
        ),
        "requires": {"construction": ("systems.parallel_axis_linear",)},
    },

    "shaft_fem.convergence.gears.timoshenko": {
        "description": (
            "Mesh convergence study (Richardson GCI on resultant "
            "transverse displacement) restricted to gear face-width "
            "intervals, Timoshenko theory, for every shaft in a "
            "SpurHelicalGearSystem (convergence_study.run_convergence(), "
            "fixtures/studies/shafts/convergence_studies/convergence_study.py) "
            "-- one fresh RigidBearingFEMSolver(theory='timoshenko') baseline "
            "solve per shaft, refined via SubmodelSolver/RichardsonGCI "
            "(convergence_solver.py) at each gear's face-width span, "
            "published into a ConvergenceResultsLibrary keyed by shaft name. "
            "Gears with no face width defined (b < "
            "MIN_FACE_WIDTH_FOR_CONVERGENCE_MM) are skipped and reported "
            "via warnings.warn(), never silently dropped."
        ),
        "requires": {"construction": ("systems.parallel_axis_linear",)},
    },
    "shaft_fem.convergence.external_distributed.timoshenko": {
        "description": (
            "Mesh convergence study restricted to DistributedRadialLoad "
            "spans (any load registered on shaft_system."
            "distributed_radial_loads, regardless of source= -- 'user' or "
            "'gear_mesh'), Timoshenko theory. Same "
            "convergence_study.run_convergence() entry point as "
            "'shaft_fem.convergence.gears.timoshenko', just a different "
            "`regions` set pinned underneath -- see that capability's own "
            "description for the shared mechanics."
        ),
        "requires": {"construction": ("systems.parallel_axis_linear",)},
    },
    "shaft_fem.convergence.total.timoshenko": {
        "description": (
            "Mesh convergence study over BOTH gear face-width intervals "
            "AND DistributedRadialLoad spans together (regions="
            "{'gears', 'external_distributed'}), Timoshenko theory. Does "
            "NOT include bearing intervals -- see this catalogue entry "
            "group's own top-of-file note for why 'total' still excludes "
            "them today."
        ),
        "requires": {"construction": ("systems.parallel_axis_linear",)},
    },
}