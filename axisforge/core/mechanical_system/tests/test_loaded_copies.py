"""``ShaftSystem.with_loads`` and the loaded copies returned by ``Transmission.resolve``, on the
two shafts of baseline case 7a: pinion shaft source,
T = 60.75 N m at 1450 rpm, sense of rotation +1, wheel shaft driven (it receives -91.125 N m) and closed
by a sink at x = 200 mm (Q16).

The case objects are built in ``axisforge.core.machine_elements.gears.tests.baseline_cases``.
"""
import math

import pytest

from axisforge.core.loads import DistributedRadialLoad, RadialLoad, TorqueLoad
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


def test_with_loads_runs_the_bounds_check_of_add_load(case7a):
    """Q42 (José, 2026-10-15): the loads of a copy go through ``add_load``: a load outside the shaft extent
    [0, 200] mm is refused, beyond the tolerance 1e-9 L."""
    with pytest.raises(ValueError, match="outside the shaft extent"):
        case7a.sP.with_loads([TorqueLoad(200.5, 1.0)])
    with pytest.raises(ValueError, match="outside the shaft extent"):
        case7a.sP.with_loads([TorqueLoad(200.0 * (1.0 + 1e-6), 1.0)])
    assert case7a.sP.loads == []


def test_distributed_gear_load_flush_with_the_shaft_end_is_accepted(case7a):
    """Q42: a gear whose face (b = 23 mm) ends at the right end of the shaft (L = 200 mm) has a distributed mesh
    load with x_hi = pos + b/2 one rounding step beyond L: accepted within the tolerance 1e-9 L. Checked on
    ``with_loads`` directly: on the case-7 shaft such a gear sits over the roller bearing and the shoulder, so
    ``resolve`` refuses the copy (Q43)."""
    flush = math.nextafter(200.0 - 23.0 / 2.0, math.inf)       # centre of the face width [mm]
    load = DistributedRadialLoad(flush - 11.5, flush + 11.5, magnitude=100.0, theta_deg=0.0, label="g:Fr",
                                 source="gear_mesh")
    assert 200.0 < load.x_hi < 200.0 + 1e-12
    assert case7a.sP.with_loads([load]).distributed_radial_loads == [load]


# ----------------------------------------------------------------------------------------------
# resolve returns loaded copies
# ----------------------------------------------------------------------------------------------
def test_resolve_leaves_the_user_shafts_untouched(case7a):
    user_P, user_W = case7a.system.shafts
    resolved = bc.resolve_case7(case7a)
    assert resolved.sP is not user_P and resolved.sW is not user_W
    assert user_P.loads == [] and user_W.loads == []
    assert user_P.shaft_position == user_W.shaft_position == (0.0, 0.0)
    assert len(resolved.sP.loads) == 6 and len(resolved.sW.loads) == 6     # bending moment of F_a (Q06), sink (Q16)
    assert (resolved.sP.speed_rpm, resolved.sW.speed_rpm) == pytest.approx((1450.0, 1450.0 * 20 / 30), rel=1e-12)   # Q09
    # wheel axis at the working centre distance along phi = 30 deg from the pinion axis [mm]
    assert resolved.sW.shaft_position == pytest.approx(
        (resolved.mesh.al * 3 ** 0.5 / 2, resolved.mesh.al / 2), rel=1e-12)
    assert resolved.sP.shaft is user_P.shaft and resolved.sP.gears[0] is user_P.gears[0]


def test_resolve_validates_the_loaded_shafts():
    """Q43 (José, 2026-10-15): ``resolve`` validates the loaded copies itself and raises one error with every
    message: a helical mesh on shafts with no locating bearing has no axial reaction path on either shaft.
    ``Transmission.validate()`` checks the written path and the user's shafts only (no mesh load: nothing to
    report), before and after."""
    case = bc.build_case7("pinion", 1, ball_arrangement="non-locating")
    assert case.system.validate() == []
    with pytest.raises(ValueError, match="invalid loaded shafts") as error:
        bc.resolve_case7(case)
    messages = [line for line in str(error.value).splitlines() if "no axial reaction path" in line]
    assert [m.split(":")[0].strip(" -") for m in messages] == ["shaft_pinion", "shaft_wheel"]
    assert case.system.validate() == []


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
                                                        "g_wheel:M", "shaft_wheel:T_out"]


# ----------------------------------------------------------------------------------------------
# sinks: closure of the output shaft (question Q16)
# ----------------------------------------------------------------------------------------------
def _torque_sum(shaft):
    return sum(ld.magnitude for ld in shaft.torque_loads)


def test_an_output_shaft_without_a_sink_is_refused(case7a):
    """Q16 (José, 2026-10-15): every output shaft (one that feeds no stage) needs a sink; without it the
    transmission is refused when it is built."""
    from axisforge.core.mechanical_system import Transmission

    with pytest.raises(ValueError, match="output shaft 'shaft_wheel' has no sink"):
        Transmission([case7a.sP, case7a.sW], [case7a.link], source=case7a.sP)


def test_sink_closes_the_driven_shaft(case7a):
    """The sink at x = 200 mm on the wheel shaft adds the TorqueLoad that makes its torques sum to zero: the
    reaction of the output, T z2/z1 = +91.125 N m against the -91.125 N m the mesh delivers."""
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
    with pytest.raises(ValueError, match="not in this system") as error:
        bc.resolve_case7(case7a, sinks=((other, 200.0),))
    assert "output shaft 'shaft_wheel' has no sink" in str(error.value)
