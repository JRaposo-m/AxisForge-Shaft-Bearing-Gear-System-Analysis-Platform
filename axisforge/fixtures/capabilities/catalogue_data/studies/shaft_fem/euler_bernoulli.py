"""axisforge/fixtures/capabilities/catalogue_data/studies/shaft_fem/euler_bernoulli.py

Euler-Bernoulli-specific leaf for CATALOGUE["studies"]["shafts"]["shaft_fem"].
See catalogue.py's own module docstring for the leaf shape and
`requires` token rules.
"""

from __future__ import annotations

EULER_BERNOULLI: dict = {
    "shaft_fem.euler_bernoulli_rigid": {
        "description": (
            "1D Euler-Bernoulli beam FEM solve for every shaft in a "
            "SpurHelicalGearSystem (EulerBernoulliRigidBearingFEMSolver "
            "+ ShaftResultsReader, same solve_system() entry point and "
            "same rigid-bearing boundary conditions as "
            "shaft_fem.timoshenko_rigid -- v=0 always, u=0 additionally "
            "for 'locating' bearings). Differs only in element physics: "
            "cubic Hermite shape functions, no independent shear-strain "
            "field (equivalent to G*As -> infinity, i.e. shear rigidity "
            "assumed infinite). Valid for slender shafts (L/D large, "
            "no short stepped sections near supports); for anything "
            "stubbier, shaft_fem.timoshenko_rigid is the one that does "
            "not neglect shear flexibility. Mutually informative with "
            "timoshenko_rigid -- same construction prerequisite, "
            "different solver class, pick one per study run."
        ),
        "requires": {"construction": ("systems.parallel_axis_linear",)},
    },
}