"""Tests of the written power path: ``mesh_forces_to_shaft_loads``, ``PairStage``, ``Transmission``
and ``ResolvedTransmission``.

The loads themselves are pinned row by row by ``test_transmission_baseline.py``. Here: the unit
guard of the functions that take a torque (case 2 numbers: 60.75 N m gives Ft 1659.831 N); the
checks of the written path (validation level L6); the stage contract with a stage of three shafts;
what the ``ResolvedTransmission`` adds (speeds, meshes under D-0010, closure checks); and the open
choices kept as today, each pinned, with the alternative as ``xfail(strict=True)`` and its question
id. The case objects are those of ``baseline_cases`` (case 7: H501 pair, z 20/30, mn 3.5 mm,
beta 15 deg, phi 30 deg, 60.75 N m at 1450 rpm).

Units: forces [N], torques [N m], positions [mm], angles [deg], speeds [rpm], power [W].
"""
import dataclasses

import pytest

from axisforge.core.machine_elements.gears import Gear

from axisforge.core.loads import AxialLoad, DistributedRadialLoad, RadialLoad, TorqueLoad
from axisforge.core.machine_elements.gears import GearPair
from axisforge.core.machine_elements.gears.tests import baseline_cases as bc
from axisforge.core.mechanical_system import (
    GearElement, PairStage, ResolvedTransmission, Transmission, mesh_forces_to_shaft_loads)

P_W = bc.T_NM * bc.OMEGA       # 9224.50 W: 60.75 N m at 1450 rpm


def xfail(question):
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason=question)


@pytest.fixture
def case7a():
    """Case 7a built and NOT resolved (pinion shaft source, sense of rotation +1)."""
    return bc.build_case7("pinion", 1)


def _stage(case, **changes):
    """The stage of a case with some fields replaced."""
    return dataclasses.replace(case.link, **changes)


def _check(resolved, text):
    """``(residual, ok)`` of the one closure check whose name contains ``text``."""
    (found,) = [c for c in resolved.checks if text in c[0]]
    return found[1], found[2]


# ----------------------------------------------------------------------------------------------
# mesh_forces_to_shaft_loads: the one crossing, today's loads
# ----------------------------------------------------------------------------------------------
def test_shaft_loads_of_case_2(case7a):
    """Unit guard and today's conventions: 60.75 N m [N m, not N mm] on the pinion, phi 30 deg, rot +1.
    Driver gear: Fr 674.505 N at 30 deg, Ft 1659.831 N at 120 deg, +444.750 N axial, +60.75 N m;
    driven gear: Fr at 210 deg, Ft at 120 deg, -444.750 N, -91.125 N m; all at x = 100 mm."""
    F = bc.FAMILY.forces(case7a.link.pair, 60.75, 30.0, 1)
    on_pinion = mesh_forces_to_shaft_loads(F, case7a.geP, "driver", 60.75)
    on_wheel = mesh_forces_to_shaft_loads(F, case7a.geW, "driven", 60.75)
    assert [type(ld) for ld in on_pinion] == [RadialLoad, RadialLoad, AxialLoad, TorqueLoad]
    assert [ld.label for ld in on_pinion] == ["g_pinion:Fr", "g_pinion:Ft", "g_pinion:Fa", "g_pinion:T"]
    assert {ld.position for ld in on_pinion + on_wheel} == {100.0}
    assert {ld.source for ld in on_pinion + on_wheel} == {"gear_mesh"}
    assert [ld.magnitude for ld in on_pinion] == pytest.approx([674.505, 1659.831, 444.750, 60.75], abs=5e-4)
    assert [ld.magnitude for ld in on_wheel] == pytest.approx([674.505, 1659.831, -444.750, -91.125], abs=5e-4)
    assert [ld.theta_deg for ld in on_pinion[:2]] == [30.0, 120.0]
    assert [ld.theta_deg for ld in on_wheel[:2]] == [210.0, 120.0]


def test_distributed_shaft_loads_carry_the_same_forces(case7a):
    """``distribute=True``: Fr and Ft are spread over the face width, 23 mm about x = 100 mm; the
    totals and the angles are those of the point loads."""
    F = bc.FAMILY.forces(case7a.link.pair, 60.75, 30.0, 1)
    point = mesh_forces_to_shaft_loads(F, case7a.geP, "driver", 60.75)
    spread = mesh_forces_to_shaft_loads(F, case7a.geP, "driver", 60.75, distribute=True)
    assert [type(ld) for ld in spread] == [DistributedRadialLoad, DistributedRadialLoad, AxialLoad, TorqueLoad]
    assert [(ld.x_lo, ld.x_hi) for ld in spread[:2]] == [(88.5, 111.5)] * 2
    assert [(ld.magnitude, getattr(ld, "theta_deg", None)) for ld in spread] == \
        [(ld.magnitude, getattr(ld, "theta_deg", None)) for ld in point]


def test_stage_takes_the_torque_in_newton_metre(case7a):
    """Unit guard of ``PairStage.resolve``: 60.75 N m gives Ft 1659.831 N and 91.125 N m on the wheel
    shaft, which turns the other way at 1450 x 20/30 = 966.667 rpm, at the centre distance along 30 deg."""
    out = case7a.link.resolve(60.75, 1450.0, 1, (0.0, 0.0))
    assert out["loads"][0][1].magnitude == pytest.approx(1659.831, abs=5e-4)
    assert out["torque_Nm"] == pytest.approx((60.75, 91.125), rel=1e-12)
    assert out["rotation"] == (1, -1) and out["speed_rpm"] == pytest.approx((1450.0, 966.6666666666666), rel=1e-12)
    al = case7a.link.pair.al
    assert out["position_mm"] == ((0.0, 0.0), pytest.approx((al * 3 ** 0.5 / 2, al / 2), rel=1e-12))
    assert out["meshes"][0].F_tw_N == pytest.approx(1659.831, abs=5e-4)


# ----------------------------------------------------------------------------------------------
# the written path: validation level L6
# ----------------------------------------------------------------------------------------------
def test_stage_stores_phi_modulo_360(case7a):
    assert _stage(case7a, phi_deg=390.0).phi_deg == 30.0 and _stage(case7a, phi_deg=-90.0).phi_deg == 270.0


def test_stage_is_frozen(case7a):
    with pytest.raises(dataclasses.FrozenInstanceError):
        case7a.link.share = 0.5


def test_shafts_of_a_stage_must_be_distinct(case7a):
    stage = _stage(case7a, shaft_b=case7a.sP)
    with pytest.raises(ValueError, match="cannot connect a shaft to itself"):
        Transmission([case7a.sP, case7a.sW], [stage], source=case7a.sP)


def test_shafts_of_a_stage_must_be_in_the_transmission(case7a):
    with pytest.raises(ValueError, match="'shaft_wheel' is not in the shafts"):
        Transmission([case7a.sP], [case7a.link], source=case7a.sP)


@pytest.mark.parametrize("share", [0.0, -0.5, 1.5])
def test_share_must_be_in_0_1(case7a, share):
    with pytest.raises(ValueError, match=r"share must be in \(0, 1\]"):
        Transmission([case7a.sP, case7a.sW], [_stage(case7a, share=share)], source=case7a.sP)


def test_bad_driver_is_refused(case7a):
    with pytest.raises(ValueError, match="driver must be 'gear1' or 'gear2'"):
        Transmission([case7a.sP, case7a.sW], [_stage(case7a, driver="pinion")], source=case7a.sP)


def test_gears_of_a_stage_must_sit_at_the_same_axial_position(case7a):
    """The two gears of a mesh are 0.5 mm apart along X once the wheel shaft is shifted."""
    case7a.sW.shaft_origin_x = 0.5
    with pytest.raises(ValueError, match="axial misalignment"):
        Transmission([case7a.sP, case7a.sW], [case7a.link], source=case7a.sP)


def test_source_must_be_one_of_the_shafts(case7a):
    other = bc.build_case7("pinion", 1).sP
    with pytest.raises(ValueError, match="source shaft 'shaft_pinion' is not in the shafts"):
        Transmission([case7a.sP, case7a.sW], [case7a.link], source=other)


def test_stage_must_start_from_a_shaft_that_has_power(case7a):
    """The path is used as written: with the wheel shaft as source the stage pinion -> wheel has an
    input shaft without power, and the wheel shaft would receive power twice."""
    with pytest.raises(ValueError, match="has no power yet") as error:
        Transmission([case7a.sP, case7a.sW], [case7a.link], source=case7a.sW)
    assert "already has power" in str(error.value)


def test_every_shaft_must_receive_power(case7a):
    with pytest.raises(ValueError, match="'shaft_wheel' receives no power"):
        Transmission([case7a.sP, case7a.sW], [], source=case7a.sP)


@pytest.mark.parametrize("n_rpm", [0.0, -1450.0])
def test_speed_must_be_positive(case7a, n_rpm):
    with pytest.raises(ValueError, match="rpm must be > 0"):
        case7a.system.resolve(P_W, n_rpm, 1)


@pytest.mark.parametrize("rotation", [0, 2, -1.5])
def test_rotation_must_be_plus_or_minus_one(case7a, rotation):
    with pytest.raises(ValueError, match=r"rotation must be \+1 or -1"):
        case7a.system.resolve(P_W, bc.N_RPM, rotation)


def test_path_is_checked_again_at_resolve(case7a):
    """Shafts are mutable: a shift made after the transmission was built is seen by ``resolve``."""
    case7a.sW.shaft_origin_x = 0.5
    with pytest.raises(ValueError, match="axial misalignment"):
        case7a.system.resolve(P_W, bc.N_RPM, 1)
    assert any("axial misalignment" in e for e in case7a.system.validate())


def test_new_checks_of_the_path_are_warnings():
    """Q45: in cases 7e, 7f the gear on the source shaft is the wheel while ``forces()`` takes the
    torque on gear 1, the pinion. The new check says so and does not refuse the recorded case."""
    assert bc.build_case7("pinion", 1).system.warnings == ()
    assert bc.build_case7("wheel", 1).system.warnings == ()        # gear 1 of that pair IS the wheel (Q03)
    (warning,) = bc.build_case7("wheel", 1, alt=True).system.warnings
    assert "is not the gear on which forces() takes the torque" in warning


def test_gear_elements_outside_the_pair_or_the_shaft_are_warnings(case7a):
    foreign = GearElement(bc.pinion_wheel("H501", position=bc.GEAR_POS)[1], role="driven", label="g_other")
    system = Transmission([case7a.sP, case7a.sW], [_stage(case7a, gear_b=foreign)], source=case7a.sP)
    assert [w.split(": ")[-1] for w in system.warnings] == [
        "gear_a and gear_b are not the two gears of the pair", "gear element 'g_other' is not on shaft 'shaft_wheel'"]


# ----------------------------------------------------------------------------------------------
# the stage contract: a stage may link more than two shafts
# ----------------------------------------------------------------------------------------------
class ThreeShaftStage:
    """Smallest stage that meets the contract of ``Transmission`` with three shafts: one input shaft
    and two output shafts. No gear is modelled; the numbers are chosen so that the power closes:
    each output shaft receives the input torque at half the speed, 2 x (T x n / 2) = T n."""

    share, label, warnings = 1.0, "split", ()

    def __init__(self, s_in, s_out1, s_out2):
        self.shafts = (s_in, s_out1, s_out2)

    def validate(self):
        return []

    def resolve(self, T_in_Nm, n_in_rpm, rot_in, position_in_mm):
        y, z = position_in_mm
        return {"loads": ([TorqueLoad(100.0, T_in_Nm, label="in:T")], [TorqueLoad(100.0, -T_in_Nm, label="out1:T")],
                          [TorqueLoad(100.0, -T_in_Nm, label="out2:T")]),
                "torque_Nm": (T_in_Nm, T_in_Nm, T_in_Nm), "rotation": (rot_in, -rot_in, -rot_in),
                "speed_rpm": (n_in_rpm, n_in_rpm / 2.0, n_in_rpm / 2.0),
                "position_mm": (position_in_mm, (y + 50.0, z), (y - 50.0, z)), "meshes": (), "checks": ()}


def test_transmission_accepts_a_stage_of_three_shafts():
    s1, s2, s3 = (bc._shaft_system(name) for name in ("s_in", "s_out1", "s_out2"))
    system = Transmission([s1, s2, s3], [ThreeShaftStage(s1, s2, s3)], source=s1, sinks=((s2, 200.0), (s3, 200.0)))
    resolved = system.resolve(P_W, bc.N_RPM, 1)
    assert [s.name for s in resolved.shafts] == ["s_in", "s_out1", "s_out2"]
    assert resolved.torque_Nm == pytest.approx((60.75, 60.75, 60.75), rel=1e-12)
    assert resolved.rotation == (1, -1, -1) and resolved.speed_rpm == (1450.0, 725.0, 725.0)
    assert [s.shaft_position for s in resolved.shafts] == [(0.0, 0.0), (50.0, 0.0), (-50.0, 0.0)]
    assert [[ld.label for ld in s.loads] for s in resolved.shafts] == [
        ["s_in:T_in", "in:T"], ["out1:T", "s_out1:T_out"], ["out2:T", "s_out2:T_out"]]
    assert resolved.meshes == () and all(ok for _, _, ok in resolved.checks)
    assert [line.split()[0] for line in resolved.summary().splitlines()[2:-1]] == ["s_in", "s_out1", "s_out2"]
    assert s1.loads == s2.loads == s3.loads == []


# ----------------------------------------------------------------------------------------------
# what the ResolvedTransmission holds
# ----------------------------------------------------------------------------------------------
def test_resolved_record_of_case_7a(case7a):
    resolved = case7a.system.resolve(P_W, bc.N_RPM, 1)
    assert isinstance(resolved, ResolvedTransmission) and resolved.transmission is case7a.system
    assert resolved.torque_Nm == pytest.approx((60.75, 91.125), rel=1e-12)
    assert resolved.rotation == (1, -1)
    assert [s.name for s in resolved.shafts] == ["shaft_pinion", "shaft_wheel"]
    assert resolved.warnings == ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        resolved.torque_Nm = ()


def test_speed_of_every_shaft_is_on_the_resolved_record():
    """ISO_21771_2007 Eq. (53): n_b = n_a z_a / z_b. Case 7a: 1450 and 966.667 rpm; case 7c (the wheel
    shaft is the source; shafts in the order pinion, wheel): 2175 and 1450 rpm."""
    a, c = bc.resolved_case7("pinion", 1), bc.resolved_case7("wheel", 1)
    assert a.resolved.speed_rpm == pytest.approx((1450.0, 1450.0 * 20 / 30), rel=1e-12)
    assert a.resolved.speed_rpm[1] == pytest.approx(966.667, abs=5e-4)
    assert c.resolved.speed_rpm == pytest.approx((2175.0, 1450.0), rel=1e-12)


def test_loaded_copies_keep_the_speed_the_user_set(case7a):
    """Pins TODAY's behaviour (Q09): ``ShaftSystem.speed_rpm`` of the copies is the user's value, 0.0
    by default, whatever the speed of the shaft."""
    case7a.sW.speed_rpm = 123.0
    resolved = case7a.system.resolve(P_W, bc.N_RPM, 1)
    assert [s.speed_rpm for s in resolved.shafts] == [0.0, 123.0]


@xfail("Q09: X6, ShaftSystem.speed_rpm of the loaded copies is not set by resolve")
def test_alternative_loaded_copies_carry_their_speed(case7a):
    resolved = case7a.system.resolve(P_W, bc.N_RPM, 1)
    assert [s.speed_rpm for s in resolved.shafts] == pytest.approx([1450.0, 966.6666666666666], rel=1e-12)


def test_meshes_are_the_mesh_under_d0010_not_the_stored_loads():
    """The ``MeshForces`` of each stage puts the torque on the gear of the source shaft. Case 7a:
    F_tw 1659.831 N, F_a 449.236 N (stored axial load: 444.750 N, Q07a). Case 7c and 7e (the wheel
    shaft is the source): F_tw 1106.554 N, where the stored Ft is 1659.831 N (Q03)."""
    a = bc.resolved_case7("pinion", 1)
    (mesh,) = a.resolved.meshes
    assert (mesh.F_tw_N, mesh.F_a_N) == pytest.approx((1659.831, 449.236), abs=5e-4)
    assert a.sP.axial_loads[0].magnitude == pytest.approx(444.750, abs=5e-4)
    for alt in (False, True):
        c = bc.resolved_case7("wheel", 1, alt=alt)
        assert c.resolved.meshes[0].F_tw_N == pytest.approx(1106.554, abs=5e-4)
        assert c.sW.radial_loads[1].magnitude == pytest.approx(1659.831, abs=5e-4)
    # the same physical mesh whichever gear is called gear 1. Force on the WHEEL (the driver): gear 1 in
    # 7c, gear 2 in 7e, whose mesh frame has e_c and e_t reversed (e_c runs from gear 1 to gear 2). The
    # axial components are not compared: the default hand is that of GEAR 1, another gear in 7e (Q08).
    plain, alt = (bc.resolved_case7("wheel", 1, alt=x).resolved.meshes[0] for x in (False, True))
    assert plain.on_1[:2] == pytest.approx((-alt.on_2[0], -alt.on_2[1]), rel=1e-12)
    assert plain.rho_mm[0] == pytest.approx(-alt.rho_mm[1], rel=1e-12)


@xfail("Q07a: row 19 / X4, the stored axial load is Ft tan(beta), not F_bt tan(beta_b)")
def test_alternative_stored_axial_load_is_the_axial_force_of_the_mesh():
    a = bc.resolved_case7("pinion", 1)
    assert abs(a.sP.axial_loads[0].magnitude) == pytest.approx(a.resolved.meshes[0].F_a_N, rel=1e-12)


# ----------------------------------------------------------------------------------------------
# closure post-conditions: stored, never raised
# ----------------------------------------------------------------------------------------------
def test_closure_checks_of_case_7a():
    """Which statics today's loads satisfy on case 7a (rot +1), and which not. The checks are
    necessary conditions only: "torque of gear = moment" passes here with the stored torque AND the
    stored F_t of the pinion both reversed against D-0010, and no check tests that F_r separates.

    - action = reaction: FAILS, 3319.663 N = 2 Ft (both stored Ft point the same way; Q05);
    - torque of each gear = moment of its mesh force: passes for rot +1 (not for rot -1; Q05);
    - power of the mesh: FAILS, 2 P = 18449.0 W (the torque signs follow the power flow; Q05);
    - torques of the pinion shaft sum to zero: passes; of the wheel shaft: FAILS, 91.125 N m (open
      end, no sink; Q16);
    - power of the train: passes.
    """
    resolved = bc.resolved_case7("pinion", 1).resolved
    assert [(name, ok) for name, _, ok in resolved.checks] == [
        ("stage: action = reaction, |F on gear_a + F on gear_b| [N]", False),
        ("stage: torque of gear_a = moment of its mesh force [N m]", True),
        ("stage: torque of gear_b = moment of its mesh force [N m]", True),
        ("stage: power of the mesh, T_a omega_a + T_b omega_b [W]", False),
        ("shaft 'shaft_pinion': sum of the torque loads [N m]", True),
        ("shaft 'shaft_wheel': sum of the torque loads [N m]", False),
        ("train: power leaving the end shafts - power entering [W]", True),
    ]
    assert _check(resolved, "action = reaction")[0] == pytest.approx(2 * 1659.8313933933, rel=1e-9)
    assert _check(resolved, "power of the mesh")[0] == pytest.approx(2 * P_W, rel=1e-9)
    assert _check(resolved, "'shaft_wheel'")[0] == pytest.approx(91.125, rel=1e-12)


def test_a_sink_closes_the_torque_of_the_output_shaft(case7a):
    resolved = bc.resolve_case7(case7a, sinks=((case7a.sW, 200.0),)).resolved
    assert _check(resolved, "'shaft_wheel'") == (0.0, True)


def test_closure_checks_that_fail_with_the_other_sense_or_source():
    """Case 7b (rot -1): the stored torques are not the moments of the stored Ft (Q05), residual 2 T.
    Case 7c (wheel source): the train gives 20755 W = 2.25 P for P = 9224.5 W in (Q03), the power of
    the mesh has the residual P + 2.25 P (Q03 + Q05), and the torque of a gear is not the moment of
    its force at the geometric radius."""
    b = bc.resolved_case7("pinion", -1).resolved
    assert _check(b, "torque of gear_a") == (pytest.approx(121.5, rel=1e-9), False)
    assert _check(b, "torque of gear_b") == (pytest.approx(182.25, rel=1e-9), False)
    c = bc.resolved_case7("wheel", 1).resolved
    assert _check(c, "train: power") == (pytest.approx(20755.12821548182 - P_W, rel=1e-9), False)
    assert _check(c, "power of the mesh") == (pytest.approx(3.25 * P_W, rel=1e-9), False)
    assert not _check(c, "torque of gear_a")[1] and not _check(c, "torque of gear_b")[1]


@xfail("Q05: row 17 / X3, directions of F_t on the driver and torque signs (action = reaction, power of the mesh)")
def test_alternative_every_closure_check_passes(case7a):
    """Under D-0010 the loads of a closed train satisfy every check (here case 7a with a sink)."""
    resolved = bc.resolve_case7(case7a, sinks=((case7a.sW, 200.0),)).resolved
    assert all(ok for _, _, ok in resolved.checks), [name for name, _, ok in resolved.checks if not ok]


# ----------------------------------------------------------------------------------------------
# shares (Q40) and negative power (Q41)
# ----------------------------------------------------------------------------------------------
def _two_wheels_on_one_pinion_shaft(**shares):
    """Pinion shaft of case 7 driving TWO wheel shafts through two pinions (two gear elements) at
    x = 100 mm, lines of centres at 30 and 210 deg."""
    c = bc.build_case7("pinion", 1)
    p2, w2 = bc.pinion_wheel("H501", position=bc.GEAR_POS)
    ge_p2, ge_w2 = GearElement(p2, role="driver", label="g_pinion2"), GearElement(w2, role="driven", label="g_wheel2")
    s_w2 = bc._shaft_system("shaft_wheel2")
    c.sP.add_gear(ge_p2)
    s_w2.add_gear(ge_w2)
    first = _stage(c, **({"share": shares["first"]} if shares else {}))
    second = PairStage(pair=GearPair.assemble(bc.FAMILY, gear1=p2, gear2=w2), shaft_a=c.sP, gear_a=ge_p2, shaft_b=s_w2,
                       gear_b=ge_w2, phi_deg=210.0, label="stage2", **({"share": shares["second"]} if shares else {}))
    return Transmission([c.sP, c.sW, s_w2], [first, second], source=c.sP)


def test_shares_split_the_torque_of_a_shaft():
    """Shares 0.7 and 0.3: the wheel shafts receive 0.7 and 0.3 of 91.125 N m, and the power closes."""
    system = _two_wheels_on_one_pinion_shaft(first=0.7, second=0.3)
    resolved = system.resolve(P_W, bc.N_RPM, 1)
    assert resolved.torque_Nm == pytest.approx((60.75, 0.7 * 91.125, 0.3 * 91.125), rel=1e-12)
    assert system.warnings == () and _check(resolved, "train: power")[1]
    assert _check(resolved, "'shaft_pinion'")[1]


def test_default_share_gives_the_full_torque_to_each_stage():
    """Pins TODAY's behaviour (Q40): two stages leaving one shaft with the default share each take
    the whole torque (as two links with separate driver gears did); the transmission warns, the
    pinion shaft does not close and the train delivers twice the input power."""
    system = _two_wheels_on_one_pinion_shaft()
    resolved = system.resolve(P_W, bc.N_RPM, 1)
    assert resolved.torque_Nm == pytest.approx((60.75, 91.125, 91.125), rel=1e-12)
    assert any("take 2.000000 times its torque" in w for w in system.warnings) and resolved.warnings == system.warnings
    assert _check(resolved, "train: power") == (pytest.approx(P_W, rel=1e-9), False)
    assert _check(resolved, "'shaft_pinion'") == (pytest.approx(60.75, rel=1e-9), False)


@xfail("Q40: shares of the stages leaving one shaft must sum to 1")
def test_alternative_shares_above_one_are_refused():
    try:
        _two_wheels_on_one_pinion_shaft()
    except ValueError:
        return
    raise AssertionError("two stages took the full torque of one shaft")


def test_negative_power_is_accepted(case7a):
    """Pins TODAY's behaviour (Q41): P < 0 gives negative RadialLoad magnitudes, reported by
    ``validate()`` of the loaded copy; the resolved record carries a warning."""
    resolved = case7a.system.resolve(-P_W, bc.N_RPM, 1)
    assert resolved.shafts[0].radial_loads[0].magnitude == pytest.approx(-674.505, abs=5e-4)
    assert any("magnitude must be >= 0" in e for e in resolved.shafts[0].validate())
    assert any("is negative" in w for w in resolved.warnings)


@xfail("Q41: P < 0 refused (validation level L6)")
def test_alternative_negative_power_is_refused(case7a):
    try:
        case7a.system.resolve(-P_W, bc.N_RPM, 1)
    except ValueError:
        return
    raise AssertionError("a negative power was resolved")


# ----------------------------------------------------------------------------------------------
# a stage that follows an internal stage (Q47)
# ----------------------------------------------------------------------------------------------
def _internal_then_external():
    """Internal spur pair of case 4 (z 20/-60, mn 2 mm; pinion on shaft 1, ring on shaft 2, x = 100 mm,
    phi 0) followed by the C14 spur pair (z 16/24; pinion on shaft 2, wheel on shaft 3, x = 50 mm,
    phi 90 deg); 9224.5 W at 1450 rpm, rot +1."""
    pin, ring = (Gear.assemble(bc.FAMILY, dataclasses.replace(g.data, position=100.0)) for g in bc.make_internal(0.0))
    p2, w2 = bc.pinion_wheel("C14", position=50.0)
    s1, s2, s3 = (bc._shaft_system(name) for name in ("shaft_1", "shaft_2", "shaft_3"))
    ge = {name: GearElement(g, role=role, label=name) for name, g, role in (
        ("pin", pin, "driver"), ("ring", ring, "driven"), ("p2", p2, "driver"), ("w2", w2, "driven"))}
    s1.add_gear(ge["pin"])
    s2.add_gear(ge["ring"])
    s2.add_gear(ge["p2"])
    s3.add_gear(ge["w2"])
    path = [PairStage(pair=GearPair.assemble(bc.FAMILY, gear1=pin, gear2=ring), shaft_a=s1, gear_a=ge["pin"], shaft_b=s2,
                      gear_b=ge["ring"], phi_deg=0.0, label="internal"),
            PairStage(pair=GearPair.assemble(bc.FAMILY, gear1=p2, gear2=w2), shaft_a=s2, gear_a=ge["p2"], shaft_b=s3,
                      gear_b=ge["w2"], phi_deg=90.0, label="external")]
    return Transmission([s1, s2, s3], path, source=s1).resolve(P_W, bc.N_RPM, 1)


def test_torque_is_negative_after_an_internal_stage():
    """Pins TODAY's behaviour (Q47): an internal pair returns T_out = T u with u = -3, so the torque
    entering shaft 2 is -182.25 N m; passed on as it is, it makes ``forces()`` of the next stage
    return negative forces, stored as negative load magnitudes (Fr -2056.361 N, Ft -4979.504 N on the
    pinion of stage 2), and the ``MeshForces`` of that stage takes the pinion as the DRIVEN gear
    (F_t on it +4979.504 N where the driving pinion would have -4979.504 N)."""
    resolved = _internal_then_external()
    assert resolved.torque_Nm == pytest.approx((60.75, -182.25, -273.375), rel=1e-12)
    assert resolved.rotation == (1, 1, -1) and resolved.speed_rpm == pytest.approx((1450.0, 1450.0 / 3, 1450.0 / 4.5), rel=1e-12)
    stage2 = {ld.label: ld.magnitude for ld in resolved.shafts[1].loads if ld.label.startswith("p2")}
    assert (stage2["p2:Fr"], stage2["p2:Ft"]) == pytest.approx((-2056.361, -4979.504), abs=5e-4)
    assert resolved.meshes[1].on_1[:2] == pytest.approx((-2056.361, 4979.504), abs=5e-4)
    assert any("magnitude must be >= 0" in e for e in resolved.shafts[1].validate())
    # the torque signs of the internal stage itself: stored torque of the pinion against the moment of its Ft (Q05)
    assert _check(resolved, "internal: torque of gear_a") == (pytest.approx(121.5, rel=1e-9), False)
    assert _check(resolved, "internal: power of the mesh") == (pytest.approx(2 * P_W, rel=1e-9), False)


@xfail("Q47: sign of the torque carried after an internal stage (negative load magnitudes in the next stage)")
def test_alternative_load_magnitudes_are_not_negative_after_an_internal_stage():
    resolved = _internal_then_external()
    assert all(ld.magnitude >= 0.0 for shaft in resolved.shafts for ld in shaft.radial_loads)


def test_warnings_follow_the_current_state(case7a):
    """``Transmission.warnings`` is read when asked: a gear element put on its shaft after the
    transmission was built no longer gives a warning."""
    foreign = GearElement(case7a.w, role="driven", label="g_late")
    system = Transmission([case7a.sP, case7a.sW], [_stage(case7a, gear_b=foreign)], source=case7a.sP)
    assert [w.split(": ")[-1] for w in system.warnings] == ["gear element 'g_late' is not on shaft 'shaft_wheel'"]
    case7a.sW.add_gear(foreign)
    assert system.warnings == ()
