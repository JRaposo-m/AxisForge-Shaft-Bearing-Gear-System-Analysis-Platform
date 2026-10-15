"""Golden tests of ``ShaftSystem`` and ``Transmission.resolve`` against the baseline of commit
39c7133 (cases 7a-7f, recorded on the system classes that ``Transmission`` and ``PairStage``
replace), and the physics checks of the stored shaft loads that need no standard.

The expected values are the CSVs in
``axisforge/core/machine_elements/gears/tests/data/gears_baseline`` (vault note
``gears_mechsys_review_2026-10-12/baseline/_baseline.md``); the case objects are built in
``axisforge.core.machine_elements.gears.tests.baseline_cases``. The code reproduces TODAY's results:
where the review found today's result wrong, the golden rows pin it and the physics check that it
breaks is ``xfail(strict=True)`` with its question id (Qnn) and review row.
"""
import math

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


def test_rows_of_case7_are_counted():
    """Rows of cases 7a, 7b, 7e, 7f (7c, 7d deleted: Q03, answer 7; rows of forces() deleted and rows of the hand
    and of the bending moments added: Q03, Q06, Q08; rows of the sinks added and of role / rotation_dir
    deleted: Q16, Q17; the 4 `meshing_load_factor` rows deleted: Q46): all computed by the code but the 8
    `driver` / `driven` rows of the four recorded pairs (Q03)."""
    rows = [(obj, name) for c in CASES for obj, name, _ in bc.baseline_rows(c)]
    declared = sum(1 for obj, name in rows if name in bc.ECHOED_ROWS[0] and obj != "link")
    assert (len(rows), len(rows) - declared, declared) == (1424, 1416, 8)


@pytest.mark.parametrize("c", bc.check_params("7", lambda c: None))     # every check passes since block 4 (Q09)
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


def test_driven_shaft_is_closed_by_its_sink(case7):
    """Check 7.c6 (Q16, José 2026-10-15): the TorqueLoad of the gear on the driven shaft is the torque the mesh
    delivers (``torque_Nm`` of that shaft) [N m]; the sink at the right end takes it out, so the shaft nets zero."""
    sums = bc.shaft_sums(case7.driven_shaft)
    assert sums["T_gear"] == pytest.approx(case7.T_driven_Nm, rel=1e-12)          # rho F_t / 1000 = -T_b
    assert sums["Tnet"] == pytest.approx(0.0, abs=1e-12)


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
    ge_ring, ge_pinion = GearElement(ring, label="g_ring"), GearElement(pinion, label="g_pinion")
    s_ring.add_gear(ge_ring)
    s_pinion.add_gear(ge_pinion)
    stage = PairStage(pair=pair, shaft_a=s_ring, gear_a=ge_ring, shaft_b=s_pinion, gear_b=ge_pinion, phi_deg=bc.PHI7)
    system = Transmission([s_ring, s_pinion], [stage], source=s_ring, sinks=((s_pinion, 200.0),))
    resolved = system.resolve(bc.T_NM * bc.OMEGA, bc.N_RPM, rot)
    assert system.validate() == []
    assert resolved.torque_Nm == pytest.approx((rot * bc.T_NM, rot * bc.T_NM * 24 / 60), rel=1e-12)
    assert resolved.rotation == (rot, rot)
    assert all(ok for _, _, ok in resolved.checks), [name for name, _, ok in resolved.checks if not ok]
    on_ring = bc.shaft_sums(resolved.shafts[0])
    m = resolved.meshes[0]
    assert math.hypot(*on_ring["Ft_vec"]) == pytest.approx(m.F_tw_N, rel=1e-12)
    assert m.F_tw_N == pytest.approx(2000.0 * bc.T_NM * 24 / 60 / pair.dl1, rel=1e-12)    # torque on gear 1 = T z1 / z2
    assert on_ring["T_gear"] == pytest.approx(-rot * bc.T_NM, rel=1e-12)              # the moment balances the source
