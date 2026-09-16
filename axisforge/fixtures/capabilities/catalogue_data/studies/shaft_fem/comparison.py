"""axisforge/fixtures/capabilities/catalogue_data/studies/shaft_fem/comparison.py

Theory-agnostic leaf for CATALOGUE["studies"]["shafts"]["shaft_fem"] --
kept separate from timoshenko.py/euler_bernoulli.py because it belongs
to neither theory (see its own description). See catalogue.py's own
module docstring for the leaf shape and `requires` token rules.
"""

from __future__ import annotations

COMPARISON: dict = {
    "shaft_fem.comparison": {
        "description": (
            "Two-solve theory comparison for every shaft in a "
            "SpurHelicalGearSystem (comparison_study.run_comparison() + "
            "print_comparison()/write_comparison_report(), fixtures/studies/"
            "shafts/fem_studies/comparison_study.py) -- calls "
            "fem_simple.solve_system() TWICE on the SAME already-resolved "
            "system, once per theory (default 'timoshenko' vs 'euler', "
            "overridable via run_comparison()'s own theory_a/theory_b "
            "kwargs), then reports sigma_b_max/v_max/bearing-reaction "
            "delta and delta%% per shaft via comparison_report.py's "
            "shaft_comparison_block(). Genuinely theory-agnostic -- does "
            "NOT itself require 'shaft_fem.timoshenko_rigid' or "
            "'shaft_fem.euler_bernoulli_rigid' to also be requested; it "
            "reaches RigidBearingFEMSolver's theory dispatch directly "
            "through fem_simple.solve_system(), the same construction "
            "prerequisite as those two, nothing more. Comparing two "
            "ShaftResults that were NOT solved on the same `system` object "
            "would silently produce a meaningless table -- see "
            "comparison_report.py's own top docstring; this capability "
            "does not (and cannot, from here) guard against that, it is "
            "the caller's responsibility, same as for the other two "
            "shaft_fem capabilities' `system` parameter."
        ),
        "requires": {"construction": ("systems.parallel_axis_linear",)},
    },
}