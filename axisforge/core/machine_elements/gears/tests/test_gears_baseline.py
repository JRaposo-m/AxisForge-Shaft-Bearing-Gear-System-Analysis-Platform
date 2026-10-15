"""Golden tests of the gear classes against the baseline of commit 39c7133 (cases 1-6b), and the
physics checks of the planetary train that need no standard (the mesh statics: ``test_mesh_forces.py``).

The expected values are the CSVs in ``tests/data/gears_baseline`` (vault note
``gears_mechsys_review_2026-10-12/baseline/_baseline.md``); the case objects are built in
``baseline_cases``. The code reproduces TODAY's results: where the review found today's result
wrong, the test pins it and the physics check that it breaks is ``xfail(strict=True)`` with its
question id (Qnn) and review row.
"""

import pytest

from axisforge.core.machine_elements.gears.tests import baseline_cases as bc

CASES = ("case1_external_spur", "case2_external_helical", "case3_helical_wheel_driver", "case4_internal_spur",
         "case5_internal_helical", "case4n_internal_spur", "case5n_internal_helical", "case6_planetary_spur",
         "case6b_planetary_helical")


@pytest.mark.parametrize("case", CASES)
def test_baseline_row_count(case):
    assert len(bc.run_baseline()[0][case]) == len(bc.baseline_rows(case))


@pytest.mark.parametrize(("case", "obj"), bc.baseline_objects(CASES))
def test_baseline_rows(case, obj):
    """Every CSV row of one recorded object (attributes, return values, exception text, validate()
    messages) is reproduced under the equivalence rule (``baseline_cases.REL_TOL``)."""
    assert bc.mismatches(case, obj) == []


def test_number_of_recorded_physics_checks():
    """The set of the baseline ``checks.csv`` (cases 1-7): 291 checks recorded. Gone: those of cases 3b, 3c,
    7c, 7d with the cases (Q03, answer 7), and the pair checks 1-5 on ``forces()``, deleted (Q03; the mesh
    statics is tested on ``MeshForces``): 132 checks, 120 pass, 4 fail (7.c9, Q09), 8 info."""
    results = [c.result for c in bc.run_baseline()[1]]
    assert (len(results), results.count("PASS"), results.count("FAIL"), results.count("INFO")) == (132, 120, 4, 8)


@pytest.mark.parametrize("c", bc.check_params("6", lambda c: None))
def test_physics_check(c):
    """Torque and power balance of the planetary train (cases 6, 6b). No library source: statics and
    kinematics."""
    assert c.result == "PASS", f"{c.desc}: got {c.got!r}, expected {c.expected!r} ({c.note})"


def test_tip_diameter_without_clearance():
    """Q04: ISO_21771_2007, 4.5.3, Eq. (33): d_a = d + 2 m_n (h_aP* + x) for k = 0, no clearance term [mm]."""
    p, _ = bc.pinion_wheel("C14")
    assert p.da == pytest.approx(p.d + 2.0 * p.mn * (p.haP + p.x), rel=bc.REL_TOL)
