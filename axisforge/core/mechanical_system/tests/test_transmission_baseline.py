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
    """Q46: of the 1416 rows of cases 7a, 7b, 7e, 7f (7c, 7d deleted: Q03, answer 7; rows of forces()
    deleted and rows of the hand and of the bending moments added: Q03, Q06, Q08), 8 are text that names a
    replaced class (compared against the new text) and 12 are state of those classes written back as
    recorded; with the 8 `driver` / `driven` rows of the four recorded pairs (Q50) that leaves 1388 rows
    computed by the code."""
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
    assert (len(rows), len(rows) - renamed - echoed - declared, renamed, echoed, declared) == (1416, 1388, 8, 12, 8)


def _question(c):
    """Question id of a case-7 check that FAILS today (None: the check passes today). Since Q03, Q05, Q06, Q07a,
    Q08 (José, 2026-10-15) only the speed written on the loaded copies is open."""
    if c.id in ("7.c9", "7.c14"):
        return "Q09: X6, ShaftSystem.speed_rpm never set by resolve"
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
    """Check 7.c6: the only TorqueLoad on the driven shaft is the one of its gear, the torque the mesh delivers
    (``torque_Nm`` of that shaft) [N m]; the shaft is closed only by a sink or a load that the user adds."""
    sums = bc.shaft_sums(case7.driven_shaft)
    assert sums["Tnet"] == sums["T_gear"] == pytest.approx(case7.T_driven_Nm, rel=1e-12)   # rho F_t / 1000 = -T_b


def test_tilting_moment_of_the_axial_force(case7):
    """Check 7.c11 (Q06, José 2026-10-15): the axial force of the helical mesh acts at the working pitch radius,
    so each shaft receives one bending moment abs(F_a) r_w [N mm] (ExternalMoment at the gear)."""
    m = case7.resolved.meshes[0]
    pair = case7.link.pair
    for shaft, gear in ((case7.driver_shaft, case7.link.gear_a.gear), (case7.driven_shaft, case7.link.gear_b.gear)):
        (M,) = shaft.external_moments
        r_w = pair.rl1 if gear is pair.gear1 else pair.rl2
        assert M.magnitude == pytest.approx(m.F_a_N * r_w, rel=1e-12) and M.position == bc.GEAR_POS


@pytest.mark.parametrize("rot", (1, -1))
def test_internal_stage_driven_by_the_ring(rot):
    """Q03, Q20 (José, 2026-10-15): the gear on the source shaft drives, here the RING of a helical internal pair
    (z 24/-60, mn 2 mm, beta 15 deg). The torque 60.75 N m goes on the ring; the pinion shaft receives
    -T_b = -60.75 x 24 / (-60) = +24.3 N m with the same sense (internal mesh); the radial forces separate and
    every closure check of the mesh passes, for both senses."""
    from axisforge.core.machine_elements.gears import GearPair
    from axisforge.core.mechanical_system import GearElement, PairStage, Transmission

    pinion = bc.gear(2.0, 24, b=20.0, beta_n_deg=15.0, hand=1, label="pinion", position=bc.GEAR_POS)
    ring = bc.gear(2.0, -60, b=20.0, beta_n_deg=15.0, hand=1, label="ring", position=bc.GEAR_POS)
    pair = GearPair.assemble(bc.FAMILY, gear1=pinion, gear2=ring)
    s_ring, s_pinion = bc._shaft_system("shaft_ring"), bc._shaft_system("shaft_pinion")
    ge_ring, ge_pinion = GearElement(ring, role="driver", label="g_ring"), GearElement(pinion, role="driven", label="g_pinion")
    s_ring.add_gear(ge_ring)
    s_pinion.add_gear(ge_pinion)
    stage = PairStage(pair=pair, shaft_a=s_ring, gear_a=ge_ring, shaft_b=s_pinion, gear_b=ge_pinion, phi_deg=bc.PHI7)
    system = Transmission([s_ring, s_pinion], [stage], source=s_ring, sinks=((s_pinion, 200.0),))
    resolved = system.resolve(bc.T_NM * bc.OMEGA, bc.N_RPM, rot)
    assert system.warnings == ()
    assert resolved.torque_Nm == pytest.approx((rot * bc.T_NM, rot * bc.T_NM * 24 / 60), rel=1e-12)
    assert resolved.rotation == (rot, rot)
    assert all(ok for _, _, ok in resolved.checks), [name for name, _, ok in resolved.checks if not ok]
    on_ring = bc.shaft_sums(resolved.shafts[0])
    m = resolved.meshes[0]
    assert math.hypot(*on_ring["Ft_vec"]) == pytest.approx(m.F_tw_N, rel=1e-12)
    assert m.F_tw_N == pytest.approx(2000.0 * bc.T_NM * 24 / 60 / pair.dl1, rel=1e-12)    # torque on gear 1 = T z1 / z2
    assert on_ring["T_gear"] == pytest.approx(-rot * bc.T_NM, rel=1e-12)              # the moment balances the source
