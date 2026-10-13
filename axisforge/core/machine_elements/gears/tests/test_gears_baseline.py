"""Golden tests of the gear classes against the baseline of commit 39c7133 (cases 1-6b), and the
physics checks of the pairs and of the planetary train that need no standard.

The expected values are the CSVs in ``tests/data/gears_baseline`` (vault note
``gears_mechsys_review_2026-10-12/baseline/_baseline.md``); the case objects are built in
``baseline_cases``. The code reproduces TODAY's results: where the review found today's result
wrong, the test pins it and the physics check that it breaks is ``xfail(strict=True)`` with its
question id (Qnn) and review row.
"""
import math

import pytest

from axisforge.core.machine_elements.gears.tests import baseline_cases as bc

CASES = ("case1_external_spur", "case2_external_helical", "case3_helical_wheel_driver", "case4_internal_spur",
         "case5_internal_helical", "case6_planetary_spur", "case6b_planetary_helical")


@pytest.mark.parametrize("case", CASES)
def test_baseline_row_count(case):
    assert len(bc.run_baseline()[0][case]) == len(bc.baseline_rows(case))


@pytest.mark.parametrize(("case", "obj"), bc.baseline_objects(CASES))
def test_baseline_rows(case, obj):
    """Every CSV row of one recorded object (attributes, return values, exception text, validate()
    messages) is reproduced under the equivalence rule (``baseline_cases.REL_TOL``)."""
    assert bc.mismatches(case, obj) == []


def test_number_of_recorded_physics_checks():
    """The set of the baseline ``checks.csv`` (cases 1-7): 291 checks, 143 pass, 136 fail, 12 info."""
    results = [c.result for c in bc.run_baseline()[1]]
    assert (len(results), results.count("PASS"), results.count("FAIL"), results.count("INFO")) == (291, 143, 136, 12)


def _question(c):
    """Question id of a pair check that FAILS today (None: the check passes today)."""
    kind = c.id.split(".")[1]
    if c.id[0] in "123" and kind == "b":
        return "Q05: row 17 / X3, F_t on the driver (both F_t of an external pair point the same way)"
    if c.id[0] in "123" and kind == "d":
        return "Q05: row 17 / X3, F_r direction (both F_r of an external pair point toward the mesh)"
    if c.id == "3.a" and ("/3b_" in c.case or "/3c_" in c.case):
        # gear 1 is the wheel: forces() still puts T on the smaller working diameter and T_out = T u
        return "Q03: row 1 / X1, gear order and driver (output power 2.25 times the input)"
    if kind == "e" and c.case.endswith("/gear2"):
        # ring driving: with phi read as driver -> driven the radial forces separate (4.d, 5.d pass)
        # but the moment of F_t on the ring is +rot T_in; with phi read as gear1 -> gear2 it is the
        # other way round
        return "Q20: meaning of phi_deg for an internal pair and directions when the ring drives"
    return None


@pytest.mark.parametrize("c", bc.check_params("123456", _question))
def test_physics_check(c):
    """Power balance, action = reaction and separating radial force of the pairs (cases 1-5); torque
    and power balance of the train (cases 6, 6b); moment of F_t on the driver of an internal pair.
    No library source: statics and kinematics."""
    assert c.result == "PASS", f"{c.desc}: got {c.got!r}, expected {c.expected!r} ({c.note})"


# ----------------------------------------------------------------------------------------------
# alternatives under question: each test states the value under the source and fails today
# ----------------------------------------------------------------------------------------------
def _case2():
    """Pair of case 2 (H501) and its ``forces(60.75 N m, phi 0, rot +1)``."""
    bc.run_baseline()
    return bc.R[("case2_external_helical", "mesh")], bc.R["forces"][(("case2_external_helical", "base"), 0.0, 1)]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Q03: row 1 / X1, gear order and driver")
def test_alternative_wheel_driver_takes_the_torque_on_the_wheel():
    """Case 3a, the wheel declared as driver with T = 60.75 N m: F_t = 1000 T / r_w(wheel) [N] with
    r_w(wheel) = a_w z_wheel / (z1 + z2) [mm], and the torque on the pinion is T z_pinion / z_wheel
    [N m] (ISO_21771_2007, Eq. (52); power balance)."""
    bc.run_baseline()
    mesh = bc.R[("case3_helical_wheel_driver", bc.V3[0])]
    F = bc.R["forces"][(("case3_helical_wheel_driver", bc.V3[0]), 0.0, 1)]
    z_pinion, z_wheel = mesh.gear1.z, mesh.gear2.z
    r_w_wheel = mesh.al * z_wheel / (z_pinion + z_wheel)
    assert F["Ft"] == pytest.approx(1000.0 * bc.T_NM / r_w_wheel, rel=bc.REL_TOL)       # N m -> N mm
    assert F["T_out"] == pytest.approx(bc.T_NM * z_pinion / z_wheel, rel=bc.REL_TOL)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Q04: row 2 / X2, clearance cP in the tip diameter")
def test_alternative_tip_diameter_without_clearance():
    """ISO_21771_2007, 4.5.3, Eq. (33): d_a = d + 2 m_n (h_aP* + x) for k = 0, no clearance term [mm]."""
    p, _ = bc.pinion_wheel("C14")
    assert p.da == pytest.approx(p.d + 2.0 * p.mn * (p.haP + p.x), rel=bc.REL_TOL)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Q07a: row 19 / X4, Fa = Ft tan(beta) instead of F_bt tan(beta_b)")
def test_alternative_axial_force_from_base_helix_angle():
    """F_a = F_bt tan(beta_b) [N], with F_bt = 1000 T / r_b1 the transverse force along the line of
    action (plan section 2.3; no ISO formula, geometry of the helicoid)."""
    mesh, F = _case2()
    F_bt = 1000.0 * bc.T_NM / (mesh.gear1.db / 2.0)      # T [N m] -> [N mm], base radius [mm]
    assert F["Fa"] == pytest.approx(F_bt * math.tan(mesh.gear1.betab), rel=bc.REL_TOL)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Q07b: row 19 / X4, Fn is F_bt instead of F_bn = F_bt / cos(beta_b)")
def test_alternative_normal_force_on_the_flank():
    """F_bn = F_bt / cos(beta_b) [N]: the one force normal to the flank (plan section 2.3; no ISO
    formula). Today's ``Fn = Ft / cos(alphatw)`` equals F_bt."""
    mesh, F = _case2()
    F_bt = 1000.0 * bc.T_NM / (mesh.gear1.db / 2.0)      # T [N m] -> [N mm], base radius [mm]
    assert F["Fn"] == pytest.approx(F_bt / math.cos(mesh.gear1.betab), rel=bc.REL_TOL)
