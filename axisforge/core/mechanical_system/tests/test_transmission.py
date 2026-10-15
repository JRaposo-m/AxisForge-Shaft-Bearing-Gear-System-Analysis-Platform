"""Tests of the written power path: ``mesh_forces_to_shaft_loads``, ``PairStage``, ``Transmission``
and ``ResolvedTransmission``.

The loads themselves are pinned row by row by ``test_transmission_baseline.py``. Here: the loads of
one mesh under D-0010 and the unit guard of the functions that take a torque (case 2 numbers:
60.75 N m gives F_tw 1659.831 N); the checks of the written path (validation level L6); the stage
contract with a stage of three shafts; what the ``ResolvedTransmission`` adds (signed torques,
speeds, meshes, closure checks: all pass, Q03, Q05, Q47, José 2026-10-15); and the open choices of
other blocks kept as today, each pinned, with the alternative as ``xfail(strict=True)`` and its
question id. The case objects are those of ``baseline_cases`` (case 7: H501 pair, z 20/30, mn 3.5 mm,
beta 15 deg, phi 30 deg, 60.75 N m at 1450 rpm).

Units: forces [N], torques [N m], positions [mm], angles [deg], speeds [rpm], power [W].
"""
import dataclasses

import numpy as np
import pytest

from axisforge.core.machine_elements.gears import Gear

from axisforge.core.loads import AxialLoad, DistributedRadialLoad, ExternalMoment, RadialLoad, TorqueLoad
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
# mesh_forces_to_shaft_loads: the one crossing (D-0010 items 14-19)
# ----------------------------------------------------------------------------------------------
def test_shaft_loads_of_case_2(case7a):
    """Unit guard and D-0010: 60.75 N m [N m, not N mm] on the right-hand pinion, e_c at phi = 30 deg, rot +1.
    Pinion (driving): F_r 674.505 N at 210 deg (toward its own axis), F_tw 1659.831 N at 300 deg, +449.236 N
    axial (h rot F_a), torque -60.75 N m (the moment of the mesh force balances the 60.75 N m of the shaft),
    bending moment F_a r_w = 16442.08 N mm; wheel: F_r at 30 deg (toward its axis), F_tw at 120 deg (action =
    reaction), -449.236 N, -91.125 N m, 24663.12 N mm; all at x = 100 mm (``questions.md``, row Q05 added in
    step 5, alternative column)."""
    mesh = bc.FAMILY.mesh_forces(case7a.link.pair, 60.75, 1)
    on_pinion = mesh_forces_to_shaft_loads(mesh, 1, case7a.geP, 30.0)
    on_wheel = mesh_forces_to_shaft_loads(mesh, 2, case7a.geW, 30.0)
    assert [type(ld) for ld in on_pinion] == [RadialLoad, RadialLoad, AxialLoad, TorqueLoad, ExternalMoment]
    assert [ld.label for ld in on_pinion] == ["g_pinion:Fr", "g_pinion:Ft", "g_pinion:Fa", "g_pinion:T", "g_pinion:M"]
    assert {ld.position for ld in on_pinion + on_wheel} == {100.0}
    assert {ld.source for ld in on_pinion + on_wheel} == {"gear_mesh"}
    assert [ld.magnitude for ld in on_pinion] == pytest.approx([674.505, 1659.831, 449.236, -60.75, 16442.081], abs=5e-4)
    assert [ld.magnitude for ld in on_wheel] == pytest.approx([674.505, 1659.831, -449.236, -91.125, 24663.121], abs=5e-4)
    assert [ld.theta_deg for ld in on_pinion[:2]] == [210.0, 300.0]
    assert [ld.theta_deg for ld in on_wheel[:2]] == [30.0, 120.0]
    # bending moment vectors (M_y, M_z) [N mm] as the FEM consumes them (vector at theta + 90 deg)
    for ld, (my, mz) in ((on_pinion[4], (8221.040367139034, -14239.259606959507)),
                         (on_wheel[4], (12331.560550708551, -21358.88941043926))):
        vec = ld.magnitude * np.array([np.cos(np.radians(ld.theta_deg + 90.0)), np.sin(np.radians(ld.theta_deg + 90.0))])
        assert vec == pytest.approx([my, mz], rel=1e-9)


def test_distributed_shaft_loads_carry_the_same_forces(case7a):
    """``distribute=True``: F_r and F_t are spread over the face width, 23 mm about x = 100 mm; the
    totals and the angles are those of the point loads."""
    mesh = bc.FAMILY.mesh_forces(case7a.link.pair, 60.75, 1)
    point = mesh_forces_to_shaft_loads(mesh, 1, case7a.geP, 30.0)
    spread = mesh_forces_to_shaft_loads(mesh, 1, case7a.geP, 30.0, distribute=True)
    assert [type(ld) for ld in spread] == [DistributedRadialLoad, DistributedRadialLoad, AxialLoad, TorqueLoad, ExternalMoment]
    assert [(ld.x_lo, ld.x_hi) for ld in spread[:2]] == [(88.5, 111.5)] * 2
    assert [(ld.magnitude, getattr(ld, "theta_deg", None)) for ld in spread] == \
        [(ld.magnitude, getattr(ld, "theta_deg", None)) for ld in point]


def test_stage_takes_the_torque_in_newton_metre(case7a):
    """Unit guard of ``PairStage.resolve``: 60.75 N m gives F_tw 1659.831 N and delivers -91.125 N m to the wheel
    shaft (T_b = T_a z_b / z_a, the mesh delivers -T_b), which turns the other way at 1450 x 20/30 = 966.667 rpm,
    at the centre distance along 30 deg."""
    out = case7a.link.resolve(60.75, 1450.0, 1, (0.0, 0.0))
    assert out["loads"][0][1].magnitude == pytest.approx(1659.831, abs=5e-4)
    assert out["torque_Nm"] == pytest.approx((60.75, -91.125), rel=1e-12)
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


def test_the_driver_is_the_gear_on_the_input_shaft():
    """Q03: the stage has no ``driver``: gear_a (on shaft_a) drives, whichever gear of the pair it is. Cases 7a
    (pinion = gear 1) and 7e (wheel = gear 2 on the source shaft) give no warning."""
    assert bc.build_case7("pinion", 1).system.warnings == ()
    with pytest.raises(ValueError, match="gear 1 must be the pinion"):      # the pair of the deleted case 7c (Q03)
        bc.build_case7("wheel", 1)
    assert bc.build_case7("wheel", 1, alt=True).system.warnings == ()
    assert not hasattr(bc.build_case7("pinion", 1).link, "driver")


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
    each output shaft receives the input torque, reversed (-T) with the other sense, at half the speed:
    2 x ((-T) x (-n / 2)) = T n."""

    share, label, warnings = 1.0, "split", ()

    def __init__(self, s_in, s_out1, s_out2):
        self.shafts = (s_in, s_out1, s_out2)

    def validate(self):
        return []

    def resolve(self, T_in_Nm, n_in_rpm, rot_in, position_in_mm):
        y, z = position_in_mm
        return {"loads": ([TorqueLoad(100.0, -T_in_Nm, label="in:T")], [TorqueLoad(100.0, -T_in_Nm, label="out1:T")],
                          [TorqueLoad(100.0, -T_in_Nm, label="out2:T")]),
                "torque_Nm": (T_in_Nm, -T_in_Nm, -T_in_Nm), "rotation": (rot_in, -rot_in, -rot_in),
                "speed_rpm": (n_in_rpm, n_in_rpm / 2.0, n_in_rpm / 2.0),
                "position_mm": (position_in_mm, (y + 50.0, z), (y - 50.0, z)), "meshes": (), "checks": ()}


def test_transmission_accepts_a_stage_of_three_shafts():
    s1, s2, s3 = (bc._shaft_system(name) for name in ("s_in", "s_out1", "s_out2"))
    system = Transmission([s1, s2, s3], [ThreeShaftStage(s1, s2, s3)], source=s1, sinks=((s2, 200.0), (s3, 200.0)))
    resolved = system.resolve(P_W, bc.N_RPM, 1)
    assert [s.name for s in resolved.shafts] == ["s_in", "s_out1", "s_out2"]
    assert resolved.torque_Nm == pytest.approx((60.75, -60.75, -60.75), rel=1e-12)
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
    # answer 9 (D-0010 item 12): torque that enters each shaft, signed about +X; T omega > 0 on both shafts
    assert resolved.torque_Nm == pytest.approx((60.75, -91.125), rel=1e-12)
    assert resolved.rotation == (1, -1)
    assert all(T * rot > 0.0 for T, rot in zip(resolved.torque_Nm, resolved.rotation))
    assert [s.name for s in resolved.shafts] == ["shaft_pinion", "shaft_wheel"]
    assert resolved.warnings == ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        resolved.torque_Nm = ()


def test_speed_of_every_shaft_is_on_the_resolved_record():
    """ISO_21771_2007 Eq. (53): n_b = n_a z_a / z_b. Case 7a: 1450 and 966.667 rpm; case 7e (the wheel
    shaft is the source; shafts in the order pinion, wheel): 2175 and 1450 rpm."""
    a, c = bc.resolved_case7("pinion", 1), bc.resolved_case7("wheel", 1, alt=True)
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


@pytest.mark.parametrize("source, rot, alt", [("pinion", 1, False), ("pinion", -1, False), ("wheel", 1, True),
                                               ("wheel", -1, True)])
def test_axial_force_follows_hand_and_sense(source, rot, alt):
    """Q08 (D-0010 items 16-17): the axial force on the DRIVING gear is h rot F_a along +X (h its hand, rot the
    sense of its shaft), the opposite on the driven gear. H501: right-hand pinion (+1), left-hand wheel (-1). 7a:
    pinion drives, +F_a on the pinion; 7b: -F_a; 7e: the wheel drives (h -1, rot +1), -F_a on the wheel and +F_a on
    the pinion; 7f: the reverse. So the same physical gear gets the same axial force for the same source and
    sense whichever gear of the pair it is called."""
    c = bc.resolved_case7(source, rot, alt)
    F_a = c.resolved.meshes[0].F_a_N
    drv, drn = (c.sP, c.sW) if source == "pinion" else (c.sW, c.sP)
    h_drv = (c.p if source == "pinion" else c.w).hand
    assert [ld.magnitude for ld in drv.axial_loads] == pytest.approx([h_drv * rot * F_a], rel=1e-12)
    assert [ld.magnitude for ld in drn.axial_loads] == pytest.approx([-h_drv * rot * F_a], rel=1e-12)


def test_stored_loads_are_those_of_the_mesh():
    """The ``MeshForces`` of each stage puts the torque on the gear of the source shaft and the stored loads are
    built from it (Q03, Q05, Q07a). Case 7a: F_tw 1659.831 N, stored axial load F_bt tan(beta_b) = 449.236 N.
    Case 7e (the wheel shaft is the source): torque on the wheel, F_tw 2000 x 60.75 / 109.800 = 1106.554 N, and
    the pinion shaft receives -60.75 x 20 / 30 = -40.5 N m (``questions.md``, row Q03)."""
    a = bc.resolved_case7("pinion", 1)
    (mesh,) = a.resolved.meshes
    assert (mesh.F_tw_N, mesh.F_a_N) == pytest.approx((1659.831, 449.236), abs=5e-4)
    assert abs(a.sP.axial_loads[0].magnitude) == pytest.approx(mesh.F_a_N, rel=1e-12)
    c = bc.resolved_case7("wheel", 1, alt=True)
    assert c.resolved.meshes[0].F_tw_N == pytest.approx(1106.554, abs=5e-4)
    assert c.sW.radial_loads[1].magnitude == pytest.approx(1106.554, abs=5e-4)
    assert c.resolved.torque_Nm == pytest.approx((-40.5, 60.75), rel=1e-12)       # (pinion shaft, wheel shaft)


# ----------------------------------------------------------------------------------------------
# closure post-conditions: stored, never raised
# ----------------------------------------------------------------------------------------------
def test_closure_checks_of_case_7a():
    """The closure checks of case 7a (rot +1): every one passes except the open end of the wheel shaft, which
    has no sink (Q16): the torque the mesh delivers to it, -91.125 N m."""
    resolved = bc.resolved_case7("pinion", 1).resolved
    assert [(name, ok) for name, _, ok in resolved.checks] == [
        ("stage: action = reaction, |F on gear_a + F on gear_b| [N]", True),
        ("stage: torque of gear_a = moment of its mesh force [N m]", True),
        ("stage: torque of gear_b = moment of its mesh force [N m]", True),
        ("stage: power of the mesh, T_a omega_a + T_b omega_b [W]", True),
        ("stage: radial force separates the gears, gear_a [N]", True),
        ("stage: radial force separates the gears, gear_b [N]", True),
        ("shaft 'shaft_pinion': sum of the torque loads [N m]", True),
        ("shaft 'shaft_wheel': sum of the torque loads [N m]", False),
        ("train: power leaving the end shafts - power entering [W]", True),
    ]
    assert _check(resolved, "'shaft_wheel'")[0] == pytest.approx(91.125, rel=1e-12)


def test_a_sink_closes_the_torque_of_the_output_shaft(case7a):
    resolved = bc.resolve_case7(case7a, sinks=((case7a.sW, 200.0),)).resolved
    assert _check(resolved, "'shaft_wheel'") == (0.0, True)


def _internal_helical_stage(ring_drives, rot):
    """Helical internal pair of case 5n (z 24/-60, mn 2 mm, beta 15 deg, both right-hand) between two shafts of
    case 7, phi 30 deg, closed by a sink; the pinion shaft or the ring shaft is the source. Returns the resolved
    record."""
    pinion = bc.gear(2.0, 24, b=20.0, beta_n_deg=15.0, hand=1, label="pinion", position=bc.GEAR_POS)
    ring = bc.gear(2.0, -60, b=20.0, beta_n_deg=15.0, hand=1, label="ring", position=bc.GEAR_POS)
    pair = GearPair.assemble(bc.FAMILY, gear1=pinion, gear2=ring)
    s_p, s_r = bc._shaft_system("shaft_pinion"), bc._shaft_system("shaft_ring")
    ge_p, ge_r = GearElement(pinion, role="driver", label="g_pinion"), GearElement(ring, role="driven", label="g_ring")
    s_p.add_gear(ge_p)
    s_r.add_gear(ge_r)
    (sa, ga), (sb, gb) = ((s_r, ge_r), (s_p, ge_p)) if ring_drives else ((s_p, ge_p), (s_r, ge_r))
    stage = PairStage(pair=pair, shaft_a=sa, gear_a=ga, shaft_b=sb, gear_b=gb, phi_deg=30.0, label="internal")
    return Transmission([sa, sb], [stage], source=sa, sinks=((sb, 200.0),)).resolve(P_W, bc.N_RPM, rot)


@pytest.mark.parametrize("run", ["7a", "7b", "7e", "7f", "internal_pinion+1", "internal_pinion-1", "internal_ring+1",
                                 "internal_ring-1"])
def test_every_closure_check_passes(run):
    """Q03, Q05 (José, 2026-10-15): under D-0010 the loads of a closed train satisfy every check, whichever the
    source shaft and the sense: cases 7a, 7b, 7e, 7f, and a helical internal stage (24/-60, beta 15 deg) driven
    by its pinion or by its ring, with a sink on the output shaft: action = reaction, torque = moment of the mesh
    force, power of the mesh, separating radial force, torque of each shaft, power out = power in."""
    if run.startswith("internal"):
        resolved = _internal_helical_stage("ring" in run, int(run[-2:]))
    else:
        source, rot, alt = {"7a": ("pinion", 1, False), "7b": ("pinion", -1, False), "7e": ("wheel", 1, True),
                            "7f": ("wheel", -1, True)}[run]
        c = bc.build_case7(source, rot, alt)
        out = c.sW if source == "pinion" else c.sP
        resolved = bc.resolve_case7(c, sinks=((out, 200.0),)).resolved
    assert all(ok for _, _, ok in resolved.checks), [name for name, _, ok in resolved.checks if not ok]


def test_axial_force_on_a_driving_ring():
    """Q08, Q20: the ring of the helical internal stage drives (rot +1, right-hand): the torque 60.75 N m goes on the
    ring, the force on gear 1 is from T_1 = 60.75 x 24 / 60 = 24.3 N m, F_tw = 2000 x 24.3 / d_w1 = 978.000 N, and the
    axial force on the driving ring is h rot F_a = +262.054 N, the opposite on the pinion."""
    resolved = _internal_helical_stage(True, 1)
    m = resolved.meshes[0]
    assert m.F_tw_N == pytest.approx(978.000, abs=5e-4)
    ring_shaft, pinion_shaft = resolved.shafts
    assert [ld.magnitude for ld in ring_shaft.axial_loads] == pytest.approx([+1 * 1 * m.F_a_N], rel=1e-12)
    assert m.F_a_N == pytest.approx(262.054, abs=5e-4)
    assert [ld.magnitude for ld in pinion_shaft.axial_loads] == pytest.approx([-m.F_a_N], rel=1e-12)


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
    """Shares 0.7 and 0.3: the wheel shafts receive 0.7 and 0.3 of -91.125 N m, and the power closes."""
    system = _two_wheels_on_one_pinion_shaft(first=0.7, second=0.3)
    resolved = system.resolve(P_W, bc.N_RPM, 1)
    assert resolved.torque_Nm == pytest.approx((60.75, -0.7 * 91.125, -0.3 * 91.125), rel=1e-12)
    assert system.warnings == () and _check(resolved, "train: power")[1]
    assert _check(resolved, "'shaft_pinion'")[1]


def test_default_share_gives_the_full_torque_to_each_stage():
    """Pins TODAY's behaviour (Q40): two stages leaving one shaft with the default share each take
    the whole torque (as two links with separate driver gears did); the transmission warns, the
    pinion shaft does not close and the train delivers twice the input power."""
    system = _two_wheels_on_one_pinion_shaft()
    resolved = system.resolve(P_W, bc.N_RPM, 1)
    assert resolved.torque_Nm == pytest.approx((60.75, -91.125, -91.125), rel=1e-12)
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
    """Pins TODAY's behaviour (Q41): P < 0 is resolved with the signed torque (power flowing back: the gear on
    the source shaft is driven, its tangential force reversed); magnitudes stay >= 0; the resolved record
    carries a warning."""
    resolved = case7a.system.resolve(-P_W, bc.N_RPM, 1)
    assert resolved.torque_Nm == pytest.approx((-60.75, 91.125), rel=1e-12)
    assert resolved.shafts[0].radial_loads[1].magnitude == pytest.approx(1659.831, abs=5e-4)
    assert resolved.shafts[0].radial_loads[1].theta_deg == 120.0           # 300 deg when the pinion drives
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
    """Internal spur pair of case 4n (z 24/-60, mn 2 mm; pinion on shaft 1, ring on shaft 2, x = 100 mm,
    phi 0) followed by the C14 spur pair (z 16/24; pinion on shaft 2, wheel on shaft 3, x = 50 mm,
    phi 90 deg); 9224.5 W at 1450 rpm, rot +1."""
    pin, ring = (Gear.assemble(bc.FAMILY, dataclasses.replace(g.data, position=100.0)) for g in bc.make_internal(0.0, z1=24))
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


def test_stage_after_an_internal_stage_is_driven_by_its_input_shaft():
    """Q47 (José, 2026-10-15): torques signed by D-0010 items 12-13. The internal mesh (u = -2.5) delivers to shaft 2
    -T_b = -60.75 x (-60) / 24 = +151.875 N m, with the sense of shaft 1 (an internal mesh keeps it); stage 2 takes
    it on its pinion, which drives, and delivers -151.875 x 24 / 16 = -227.8125 N m to shaft 3, which turns the
    other way. Load magnitudes positive (F_r 1713.634 N, F_t 4149.587 N on the pinion of stage 2, = 2056.361 and
    4979.504 N x 151.875 / 182.25 of the 20/-60 pair), every check of the meshes passes, power out = power in."""
    resolved = _internal_then_external()
    T2 = -(60.75 * -60 / 24)                                   # the mesh delivers -T_b, T_b = T_a z_b / z_a
    assert resolved.torque_Nm == pytest.approx((60.75, T2, -(T2 * 24 / 16)), rel=1e-12)
    assert resolved.torque_Nm == pytest.approx((60.75, 151.875, -227.8125), rel=1e-12)
    assert resolved.rotation == (1, 1, -1) and resolved.speed_rpm == pytest.approx((1450.0, 1450.0 / 2.5, 1450.0 / 3.75), rel=1e-12)
    stage2 = {ld.label: ld.magnitude for ld in resolved.shafts[1].loads if ld.label.startswith("p2")}
    assert (stage2["p2:Fr"], stage2["p2:Ft"]) == pytest.approx((1713.634, 4149.587), abs=5e-4)
    assert resolved.meshes[1].on_1[:2] == pytest.approx((-1713.634, -4149.587), abs=5e-4)     # the pinion drives
    assert all(ld.magnitude >= 0.0 for shaft in resolved.shafts for ld in shaft.radial_loads)
    assert all(ok for name, _, ok in resolved.checks if "shaft_3" not in name), resolved.checks   # shaft 3: no sink


def test_warnings_follow_the_current_state(case7a):
    """``Transmission.warnings`` is read when asked: a gear element put on its shaft after the
    transmission was built no longer gives a warning."""
    foreign = GearElement(case7a.w, role="driven", label="g_late")
    system = Transmission([case7a.sP, case7a.sW], [_stage(case7a, gear_b=foreign)], source=case7a.sP)
    assert [w.split(": ")[-1] for w in system.warnings] == ["gear element 'g_late' is not on shaft 'shaft_wheel'"]
    case7a.sW.add_gear(foreign)
    assert system.warnings == ()
