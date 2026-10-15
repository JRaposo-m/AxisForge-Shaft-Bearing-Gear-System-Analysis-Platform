"""Golden tests of ``ShaftSystem`` and ``Transmission.resolve`` against the baseline of commit
39c7133 (cases 7a-7f, recorded on the system classes that ``Transmission`` and ``PairStage``
replace), and the physics checks of the stored shaft loads that need no standard.

The expected values are the CSVs in ``tests/data/gears_baseline`` (vault note
``gears_mechsys_review_2026-10-12/baseline/_baseline.md``); the case objects are built in
``axisforge.core.machine_elements.gears.tests.baseline_cases``. The code reproduces TODAY's results:
where the review found today's result wrong, the golden rows pin it and the physics check that it
breaks is ``xfail(strict=True)`` with its question id (Qnn) and review row.
"""
import math
import re

import pytest

from axisforge.core.machine_elements.gears.tests import baseline_cases as bc

CASES = tuple(f"case7{tag}" for tag, _, _, _ in bc.RUNS7)


@pytest.mark.parametrize("case", CASES)
def test_baseline_row_count(case):
    assert len(bc.run_baseline()[0][case]) == len(bc.baseline_rows(case))


@pytest.mark.parametrize(("case", "obj"), bc.baseline_objects(CASES))
def test_baseline_rows(case, obj):
    """Every CSV row of one recorded object (system, link, pair, shafts with their stored loads) is
    reproduced under the equivalence rule (``baseline_cases.REL_TOL``)."""
    assert bc.mismatches(case, obj) == []


def test_rows_of_the_replaced_classes_are_counted():
    """Q46: of the 1488 rows of cases 7a, 7b, 7e, 7f (7c, 7d deleted: Q03, answer 7), 8 are text that
    names a replaced class (compared against the new text) and 12 are state of those classes written
    back as recorded; with the 8 `driver` / `driven` rows of the four recorded pairs (Q50) that leaves
    1460 rows computed by the code."""
    got = bc.run_baseline()[0]
    rows = [(c, i, obj, name, text) for c in CASES for i, (obj, name, text) in enumerate(bc.baseline_rows(c))]
    renamed = 0
    for pattern, (now, count) in bc.RENAMED_PATTERNS.items():
        hits = [(c, i, re.fullmatch(pattern, text)) for c, i, _, _, text in rows if re.fullmatch(pattern, text)]
        assert len(hits) == count
        assert all(got[c][i][2] == now.format(*m.groups()) for c, i, m in hits)
        renamed += count
    echoed = 0
    for (obj, ending), (value, count, _) in bc.ECHOED_ROWS_CASE7.items():
        hits = [text for _, _, o, name, text in rows if o == obj and name.endswith(ending)]
        assert len(hits) == count and set(hits) == {bc.fmt(value)}
        echoed += count
    declared = sum(1 for _, _, obj, name, _ in rows if name in bc.ECHOED_ROWS[0] and obj != "link")
    assert (len(rows), len(rows) - renamed - echoed - declared, renamed, echoed, declared) == (1488, 1460, 8, 12, 8)


Q03 = "Q03: row 1 / X1, gear order and driver"
Q05 = "Q05: row 17 / X3"


def _question(c):
    """Question id of a case-7 check that FAILS today (None: the check passes today)."""
    wheel_source = c.case[5] in "ef"      # case7e, 7f: the wheel shaft receives the power
    # with the wheel as source the geometric radius of the wheel differs from the one the code uses
    wrong_radius = wheel_source and "geometric radii" in c.desc
    if c.id in ("7.b", "7.b2"):
        return f"{Q05}, F_t on the driver (both stored F_t point the same way)"
    if c.id == "7.d":
        return f"{Q05}, F_r direction (both stored F_r point toward the mesh)"
    if c.id == "7.c1":
        # the moment of the stored F_t on the driver shaft is +rot T_in instead of -rot T_in
        return f"{Q05}, F_t on the driver" + (f" + {Q03} (magnitude)" if wrong_radius else "")
    if c.id in ("7.c3", "7.c4") and "rot-1" in c.case:
        # the TorqueLoad signs follow the power flow; the F_t directions follow the sense of rotation
        return f"{Q05}, torque sign vs sense of rotation"
    if c.id in ("7.c12", "7.c13") and ("rot-1" in c.case or wrong_radius):
        sign = f"{Q05}, torque sign vs sense of rotation" if "rot-1" in c.case else ""
        return " + ".join(q for q in (sign, f"{Q03} (magnitude)" if wrong_radius else "") if q)
    if c.id in ("7.c2", "7.c7", "7.c8") and wheel_source:
        # T on the smaller working diameter and T_out = T u, whichever gear is the driver
        return f"{Q03} (driven torque 91.125 N m instead of 40.5 N m)"
    if c.id in ("7.c9", "7.c14"):
        return "Q09: X6, ShaftSystem.speed_rpm never set by resolve"
    if c.id == "7.x3":
        return "Q08: hand of helix, the axial thrust does not reverse with the sense of rotation"
    return None


@pytest.mark.parametrize("c", bc.check_params("7", _question))
def test_physics_check(c):
    """Action = reaction, torque = moment of the mesh force, power, speed, separating radial force
    and reversal of the sense of rotation, on the loads stored by ``resolve``. No library source:
    statics and kinematics (frame and signs in ``baseline_cases.physics_system``)."""
    assert c.result == "PASS", f"{c.desc}: got {c.got!r}, expected {c.expected!r} ({c.note})"


# ----------------------------------------------------------------------------------------------
# the two checks recorded as INFO (a value, no verdict), pinned at today's value
# ----------------------------------------------------------------------------------------------
@pytest.fixture(params=CASES)
def case7(request):
    """Resolved ``baseline_cases.Case7`` of one of the runs 7a-7f."""
    bc.run_baseline()
    return bc.R[(request.param, "system")]


def test_driven_shaft_torque_is_left_open(case7):
    """Check 7.c6: the only TorqueLoad on the driven shaft is the one of its gear, -T_out [N m]; the
    shaft is closed only by a load that the user adds."""
    sums = bc.shaft_sums(case7.driven_shaft)
    assert sums["Tnet"] == sums["T_gear"] == -case7.T_driven_Nm


def test_no_tilting_moment_of_the_axial_force(case7):
    """Check 7.c11 (row 18 / X4, question Q06): no moment F_a x r_w is stored on either shaft."""
    assert case7.driver_shaft.external_moments == [] and case7.driven_shaft.external_moments == []


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Q06: row 18 / X4, tilting moment F_a x r_w absent")
def test_alternative_tilting_moment_of_the_axial_force(case7):
    """The axial force of a helical mesh acts at the working pitch radius, so each shaft receives a
    bending moment abs(F_a) r_w [N mm] (statics; no library source for its direction; how it maps
    onto ``ExternalMoment`` is not ruled, so only its presence is asked)."""
    assert case7.driver_shaft.external_moments and case7.driven_shaft.external_moments


@pytest.mark.parametrize("driver", ("gear1", "gear2"))
def test_internal_link_takes_the_declared_driver(driver):
    """Q03: the stage passes its ``driver`` to ``forces()`` as the pair did before step 3 (default
    "gear1"), whatever gear ``gear_a`` wraps. Here ``gear_a`` wraps the RING of the pair of case 5
    (z 20/-60, mn 2 mm, beta 15 deg): with the default the torque is still taken on the pinion, and
    the stage says so in its warnings (Q45)."""
    from axisforge.core.machine_elements.gears import GearPair
    from axisforge.core.mechanical_system import GearElement, PairStage, Transmission

    pinion = bc.gear(2.0, 20, b=20.0, beta_n_deg=15.0, label="pinion", position=bc.GEAR_POS)
    ring = bc.gear(2.0, -60, b=20.0, beta_n_deg=15.0, label="ring", position=bc.GEAR_POS)
    pair = GearPair.assemble(bc.FAMILY, gear1=pinion, gear2=ring)
    s_ring, s_pinion = bc._shaft_system("shaft_ring"), bc._shaft_system("shaft_pinion")
    ge_ring, ge_pinion = GearElement(ring, role="driver", label="g_ring"), GearElement(pinion, role="driven", label="g_pinion")
    s_ring.add_gear(ge_ring)
    s_pinion.add_gear(ge_pinion)
    kw = {} if driver == "gear1" else {"driver": driver}
    stage = PairStage(pair=pair, shaft_a=s_ring, gear_a=ge_ring, shaft_b=s_pinion, gear_b=ge_pinion, phi_deg=bc.PHI7, **kw)
    system = Transmission([s_ring, s_pinion], [stage], source=s_ring)
    loaded_ring, loaded_pinion = system.resolve(bc.T_NM * bc.OMEGA, bc.N_RPM, 1).shafts
    assert any("not the gear on which forces() takes the torque" in w for w in system.warnings) == (driver == "gear1")

    F = bc.FAMILY.forces(pair, bc.T_NM, bc.PHI7, 1, driver=driver)       # [N], [N m]
    other = bc.FAMILY.forces(pair, bc.T_NM, bc.PHI7, 1, driver="gear2" if driver == "gear1" else "gear1")
    assert F["Ft"] != pytest.approx(other["Ft"], rel=0.1)                # the driver matters for an internal pair
    on_ring, on_pinion = bc.shaft_sums(loaded_ring), bc.shaft_sums(loaded_pinion)
    assert math.hypot(*on_ring["Ft_vec"]) == pytest.approx(F["Ft"], rel=1e-12)
    assert on_ring["Fa"] == pytest.approx(F["Fa"], rel=1e-12)
    assert on_pinion["T_gear"] == pytest.approx(-F["T_out"], rel=1e-12)   # power-flow sign of the driven gear
