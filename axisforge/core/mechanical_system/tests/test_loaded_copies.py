"""``ShaftSystem.with_loads`` and the loaded copies returned by ``Transmission.resolve``, on the
two shafts of baseline case 7a: pinion shaft source,
T = 60.75 N m at 1450 rpm, sense of rotation +1, wheel shaft driven (T_out = 91.125 N m).

The case objects are built in ``axisforge.core.machine_elements.gears.tests.baseline_cases``.
"""
import math

import pytest

from axisforge.core.loads import RadialLoad, TorqueLoad
from axisforge.core.machine_elements.gears.tests import baseline_cases as bc


def _load_values(shaft):
    """Comparable description of the loads of a shaft: type, label, position [mm], magnitude, angle [deg]."""
    return [(type(ld).__name__, ld.label, ld.source, ld.position, ld.magnitude, getattr(ld, "theta_deg", None))
            for ld in shaft.loads]


@pytest.fixture
def case7a():
    """Case 7a built and NOT resolved (``baseline_cases.Case7`` holding the user's shafts)."""
    return bc.build_case7("pinion", 1)


# ----------------------------------------------------------------------------------------------
# ShaftSystem.with_loads
# ----------------------------------------------------------------------------------------------
def test_with_loads_returns_a_copy_sharing_shaft_bearings_and_gears(case7a):
    user = case7a.sP
    user.add_load(RadialLoad(50.0, 10.0, 0.0, label="user"))       # 10 N at x = 50 mm
    copy = user.with_loads([TorqueLoad(100.0, 5.0, label="extra")], speed_rpm=1450.0, shaft_position=(1.0, 2.0))
    assert copy is not user
    assert copy.shaft is user.shaft
    assert all(a is b for a, b in zip(copy.bearings, user.bearings)) and len(copy.bearings) == 2
    assert all(a is b for a, b in zip(copy.gears, user.gears)) and len(copy.gears) == 1
    assert [ld.label for ld in copy.loads] == ["user", "extra"]     # the loads of the shaft, then the new ones
    assert (copy.speed_rpm, copy.shaft_position) == (1450.0, (1.0, 2.0))
    assert (copy.name, copy.design_life_hours, copy.shaft_origin_x, copy.label) == (
        user.name, user.design_life_hours, user.shaft_origin_x, user.label)
    # the user's shaft is untouched
    assert [ld.label for ld in user.loads] == ["user"]
    assert (user.speed_rpm, user.shaft_position) == (0.0, (0.0, 0.0))


def test_with_loads_keeps_speed_and_position_by_default(case7a):
    copy = case7a.sP.with_loads([])
    assert (copy.speed_rpm, copy.shaft_position) == (case7a.sP.speed_rpm, case7a.sP.shaft_position)


def test_with_loads_stores_the_loads_without_a_bounds_check(case7a):
    """Pins TODAY's behaviour (Q42): a load outside the shaft extent [0, 200] mm is stored, as the
    former ``set_gear_loads`` stored it; ``validate()`` of the copy reports it."""
    copy = case7a.sP.with_loads([TorqueLoad(200.5, 1.0)])
    assert [ld.position for ld in copy.loads] == [200.5]
    assert any("load position 200.5000 mm outside" in e for e in copy.validate())
    assert case7a.sP.loads == []


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Q42: resolve validates the loaded copies (bounds check of add_load)")
def test_alternative_with_loads_runs_the_bounds_check(case7a):
    """The alternative (plan section 4): the loads go through ``add_load``, which refuses a position
    outside the shaft extent."""
    try:
        case7a.sP.with_loads([TorqueLoad(200.5, 1.0)])
    except ValueError:
        return
    raise AssertionError("a load at x = 200.5 mm was stored on a shaft of 200 mm")


def test_distributed_gear_flush_with_the_shaft_end_resolves():
    """A gear whose face (b = 23 mm) ends at the right end of the shaft (L = 200 mm), one rounding
    step beyond it: the distributed mesh load gets x_hi = pos + b/2 > L and is stored all the same
    (today's behaviour; under the alternative of Q42 without a tolerance ``resolve`` would raise)."""
    flush = math.nextafter(200.0 - 23.0 / 2.0, math.inf)       # centre of the face width [mm]
    resolved = bc.resolve_case7(bc.build_case7("pinion", 1, distribute_loads=True, gear_pos=flush))
    x_hi = [ld.x_hi for ld in resolved.sP.distributed_radial_loads]
    assert len(x_hi) == 2 and all(200.0 < x < 200.0 + 1e-12 for x in x_hi)


# ----------------------------------------------------------------------------------------------
# resolve returns loaded copies
# ----------------------------------------------------------------------------------------------
def test_resolve_leaves_the_user_shafts_untouched(case7a):
    user_P, user_W = case7a.system.shafts
    resolved = bc.resolve_case7(case7a)
    assert resolved.sP is not user_P and resolved.sW is not user_W
    assert user_P.loads == [] and user_W.loads == []
    assert user_P.shaft_position == user_W.shaft_position == (0.0, 0.0)
    assert len(resolved.sP.loads) == 6 and len(resolved.sW.loads) == 5     # with the bending moment of F_a (Q06)
    # wheel axis at the working centre distance along phi = 30 deg from the pinion axis [mm]
    assert resolved.sW.shaft_position == pytest.approx(
        (resolved.mesh.al * 3 ** 0.5 / 2, resolved.mesh.al / 2), rel=1e-12)
    assert resolved.sP.shaft is user_P.shaft and resolved.sP.gears[0] is user_P.gears[0]


def test_validate_after_resolve_sees_the_mesh_loads():
    """Pins TODAY's behaviour (Q43): ``Transmission.validate()`` after ``resolve`` checks
    the loaded shafts, so a helical mesh on shafts with no locating bearing is reported (the axial
    thrust has no reaction path); before ``resolve`` there is no mesh load and nothing to report."""
    case = bc.build_case7("pinion", 1, ball_arrangement="non-locating")
    assert case.system.validate() == []
    bc.resolve_case7(case)
    messages = [e for e in case.system.validate() if "no axial reaction path" in e]
    assert [m.split(":")[0] for m in messages] == ["shaft_pinion", "shaft_wheel"]


def test_resolving_twice_gives_equal_loads(case7a):
    first = bc.resolve_case7(case7a)
    second = bc.resolve_case7(case7a)
    assert first.sP is not second.sP
    assert _load_values(first.sP) == _load_values(second.sP)
    assert _load_values(first.sW) == _load_values(second.sW)
    assert first.sW.shaft_position == second.sW.shaft_position


def test_resolve_keeps_the_user_loads_first(case7a):
    case7a.sW.add_load(RadialLoad(100.0, 10.0, 0.0, label="user"))
    resolved = bc.resolve_case7(case7a)
    assert [ld.label for ld in case7a.system.shafts[1].loads] == ["user"]
    assert [ld.label for ld in resolved.sW.loads] == ["user", "g_wheel:Fr", "g_wheel:Ft", "g_wheel:Fa", "g_wheel:T",
                                                        "g_wheel:M"]


# ----------------------------------------------------------------------------------------------
# sinks: closure of the output shaft (question Q16)
# ----------------------------------------------------------------------------------------------
def _torque_sum(shaft):
    return sum(ld.magnitude for ld in shaft.torque_loads)


def test_default_leaves_the_driven_shaft_open(case7a):
    """No sink: the wheel shaft carries only the TorqueLoad of its gear, -T_out = -T z2/z1 [N m]."""
    resolved = bc.resolve_case7(case7a)
    assert _torque_sum(resolved.sW) == pytest.approx(-bc.T_NM * case7a.w.z / case7a.p.z, rel=1e-12)
    assert _torque_sum(resolved.sP) == 0.0


def test_sink_closes_the_driven_shaft(case7a):
    """A sink at x = 200 mm on the wheel shaft adds the TorqueLoad that makes its torques sum to zero."""
    _, user_W = case7a.system.shafts
    resolved = bc.resolve_case7(case7a, sinks=((user_W, 200.0),))
    loaded_P, loaded_W = resolved.sP, resolved.sW
    sink = loaded_W.loads[-1]
    assert (type(sink), sink.label, sink.position, sink.source) == (TorqueLoad, "shaft_wheel:T_out", 200.0, "gear_mesh")
    assert sink.magnitude == pytest.approx(bc.T_NM * case7a.w.z / case7a.p.z, rel=1e-12)
    assert _torque_sum(loaded_W) == 0.0
    assert len(loaded_P.loads) == 6 and user_W.loads == []


def test_sink_on_a_shaft_outside_the_system_is_refused(case7a):
    """The sinks are written with the transmission, so the refusal comes when it is built."""
    other = bc.build_case7("pinion", 1).sW
    with pytest.raises(ValueError, match="not in this system"):
        bc.resolve_case7(case7a, sinks=((other, 200.0),))
