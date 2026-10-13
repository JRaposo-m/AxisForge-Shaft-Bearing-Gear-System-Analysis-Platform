"""Tests of ``MeshForces`` and ``CylindricalFamily.mesh_forces`` (the mesh under D-0010).

Three groups: the six magnitudes of baseline case 2 against the hand arithmetic of the plan
(unit conversions of D-0010 item 5, and the "N mm by mistake" guard); the mesh frame and the sign
table of D-0010 items 14-16 (vault note ``units_and_conventions.md``, C.4); statics that need no
standard (action = reaction, torque = moment of the mesh force, one normal force). These are NEW
numbers: the forces that the shaft loads still use are those of ``forces()``, pinned in
``test_cylindrical_family.py`` and by the baseline rows.

Case 2: H501, z 20/30, mn 3.5 mm, beta 15 deg, T = 60.75 N m on gear 1; d_w1 = 73.200206 mm,
d_1 = 72.469333 mm, d_b1 = 67.814717 mm. Units: forces [N], torques [N m], lengths [mm].
"""
import math

import pytest

from axisforge.core.machine_elements.gears import CylindricalFamily, GearPair, MeshForces
from axisforge.core.machine_elements.gears.tests import baseline_cases as bc

FAMILY = CylindricalFamily()
REL = bc.REL_TOL
T = bc.T_NM                                # [N m] torque on gear 1


def pair(g1, g2):
    return GearPair.assemble(FAMILY, gear1=g1, gear2=g2)


CASES = {
    "case1": pair(*bc.pinion_wheel("C14")),          # external spur, z 16/24
    "case2": pair(*bc.pinion_wheel("H501")),         # external helical, z 20/30, beta 15 deg
    "case3b": pair(*bc.pinion_wheel("H501")[::-1]),  # the same gears, gear 1 = wheel (Q03)
    "case4": pair(*bc.make_internal(0.0)),           # internal spur, z 20/-60
    "case5": pair(*bc.make_internal(15.0)),          # internal helical (Q21)
}
# hand arithmetic of the plan (header) and of units_and_conventions.md, C.5, to the printed digit
CASE2_N = dict(F_tw_N=1659.831, F_r_N=674.505, F_a_N=449.236, F_bt_N=1791.646, F_bn_N=1847.109, F_t_nominal_N=1676.571)


# ----------------------------------------------------------------------------------------------
# magnitudes and units
# ----------------------------------------------------------------------------------------------
def test_case2_magnitudes():
    m = FAMILY.mesh_forces(CASES["case2"], T, 1)
    assert isinstance(m, MeshForces)
    for name, value in CASE2_N.items():
        assert getattr(m, name) == pytest.approx(value, abs=5e-4), name
    # the conversions written once (D-0010 item 5): F [N] = 2000 T [N m] / d [mm]
    assert m.F_tw_N == pytest.approx(2000.0 * 60.75 / 73.200206, rel=1e-8)
    assert m.F_t_nominal_N == pytest.approx(2000.0 * 60.75 / 72.469333, rel=1e-8)
    assert m.F_bt_N == pytest.approx(2000.0 * 60.75 / 67.814717, rel=1e-8)
    assert m.x_offset_mm == (0.0, 0.0)


def test_torque_is_in_newton_metre_not_newton_millimetre():
    """Unit guard: 60.75 N m gives 1659.831 N at d_w and 1676.571 N at d; the same torque written
    in N mm (60750) gives forces exactly 1000 times larger (1.66 MN on a module 3.5 gear)."""
    m = FAMILY.mesh_forces(CASES["case2"], 60.75, 1)
    wrong = FAMILY.mesh_forces(CASES["case2"], 60750.0, 1)
    assert (m.F_tw_N, m.F_t_nominal_N) == (pytest.approx(1659.831, abs=5e-4), pytest.approx(1676.571, abs=5e-4))
    assert wrong.F_tw_N == pytest.approx(1000.0 * m.F_tw_N, rel=REL) and wrong.F_tw_N > 1.6e6


def test_magnitudes_agree_with_todays_forces_where_the_definitions_agree():
    """F_tw, F_r, F_bt are today's Ft, Fr, Fn of ``forces()``; F_a and F_bn are not (Q07a, Q07b)."""
    p = CASES["case2"]
    m, F = FAMILY.mesh_forces(p, T, 1), FAMILY.forces(p, T)
    assert (m.F_tw_N, m.F_r_N, m.F_bt_N) == pytest.approx((F["Ft"], F["Fr"], F["Fn"]), rel=REL)
    assert m.F_a_N / F["Fa"] == pytest.approx(p.rl1 / p.gear1.r, rel=1e-12)       # +1.0 %: r_w / r
    assert m.F_bn_N / F["Fn"] == pytest.approx(1.0 / math.cos(p.gear1.betab), rel=1e-12)


@pytest.mark.parametrize("case", CASES)
def test_one_normal_force(case):
    """The three sets describe one force: F_tw^2 + F_r^2 + F_a^2 = F_bn^2 (geometry of the helicoid)."""
    m = FAMILY.mesh_forces(CASES[case], T, 1)
    assert math.hypot(m.F_tw_N, m.F_r_N, m.F_a_N) == pytest.approx(m.F_bn_N, rel=1e-12)


def test_spur_pair_has_no_axial_force():
    m = FAMILY.mesh_forces(CASES["case1"], T, 1)
    assert m.F_a_N == 0.0 and m.on_1[2] == 0.0 and m.F_bn_N == m.F_bt_N


def test_gear_1_is_the_wheel():
    """Q03: with the torque on a gear 1 that is the WHEEL, the force is on the working diameter of
    the wheel: F_tw = 2000 x 60.75 / 109.800309 = 1106.554 N (``forces()`` gives 1659.831 N)."""
    p = CASES["case3b"]
    m = FAMILY.mesh_forces(p, T, 1)
    assert m.F_tw_N == pytest.approx(1106.554, abs=5e-4)
    assert m.rho_mm == pytest.approx((54.900155, -36.600103), abs=5e-7)
    assert FAMILY.forces(p, T)["Ft"] == pytest.approx(1659.831, abs=5e-4)


# ----------------------------------------------------------------------------------------------
# mesh frame and sign table (D-0010 items 14-16)
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", ["case2", "case4"])
def test_pitch_point_from_each_axis(case):
    """rho along e_c (gear 1 -> gear 2): between the axes for an external pair (+r_w1, -r_w2);
    beyond the pinion, away from the ring centre, for an internal pair (-r_w1, -r_w2)."""
    p = CASES[case]
    rho_1, rho_2 = FAMILY.mesh_forces(p, T, 1).rho_mm
    sign = 1.0 if p.gear2.z > 0 else -1.0
    assert rho_1 == pytest.approx(sign * p.rl1, rel=1e-12) and rho_2 == pytest.approx(-p.rl2, rel=1e-12)
    assert rho_1 - rho_2 == pytest.approx(p.al, rel=1e-12)


@pytest.mark.parametrize("rot", [1, -1])
@pytest.mark.parametrize("hand", [1, -1])
def test_sign_table_external_pair_gear_1_driving(rot, hand):
    """On the driver: (-F_r, -rot F_tw, +h rot F_a); on the driven gear the opposite."""
    m = FAMILY.mesh_forces(CASES["case2"], rot * T, rot, hand1=hand)     # driving: T omega > 0
    assert m.on_1 == pytest.approx((-m.F_r_N, -rot * m.F_tw_N, hand * rot * m.F_a_N), rel=REL)
    assert m.on_2 == pytest.approx((+m.F_r_N, +rot * m.F_tw_N, -hand * rot * m.F_a_N), rel=REL)


@pytest.mark.parametrize("rot", [1, -1])
@pytest.mark.parametrize("hand", [1, -1])
def test_sign_table_internal_pair_pinion_driving(rot, hand):
    """On the pinion: (+F_r, +rot F_tw, +h rot F_a); the ring is pushed outward: (-F_r, ...)."""
    m = FAMILY.mesh_forces(CASES["case5"], rot * T, rot, hand1=hand)
    assert m.on_1 == pytest.approx((+m.F_r_N, +rot * m.F_tw_N, hand * rot * m.F_a_N), rel=REL)
    assert m.on_2 == pytest.approx((-m.F_r_N, -rot * m.F_tw_N, -hand * rot * m.F_a_N), rel=REL)


def test_default_hand_is_right_hand_gear_1():
    """Q08: with no hand given gear 1 is right-hand: driving in the +1 sense it is pushed along +X."""
    p = CASES["case2"]
    assert FAMILY.mesh_forces(p, T, 1) == FAMILY.mesh_forces(p, T, 1, hand1=1)
    assert FAMILY.mesh_forces(p, T, 1).on_1[2] > 0.0 > FAMILY.mesh_forces(p, T, 1, hand1=-1).on_1[2]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("T1, rot", [(T, 1), (-T, -1), (-T, 1), (T, -1)])
def test_action_equals_reaction(case, T1, rot):
    m = FAMILY.mesh_forces(CASES[case], T1, rot)
    assert m.on_2 == tuple(-f for f in m.on_1)


@pytest.mark.parametrize("case", ["case1", "case2", "case3b", "case4"])
@pytest.mark.parametrize("T1", [T, -T])
def test_torque_is_the_moment_of_the_mesh_force(case, T1):
    """Steady rotation of each gear: T_i + rho_i F_i,t / 1000 = 0; with action = reaction this gives
    T_2 = T_1 z_2 / z_1 with signed z (D-0010 item 13). Case 5 is left out: its working pitch circles
    are not tangent today (Q21)."""
    p = CASES[case]
    m = FAMILY.mesh_forces(p, T1, 1)
    assert -m.rho_mm[0] * m.on_1[1] / 1000.0 == pytest.approx(T1, rel=1e-12)
    assert -m.rho_mm[1] * m.on_2[1] / 1000.0 == pytest.approx(T1 * p.gear2.z / p.gear1.z, rel=1e-12)


def test_sense_of_rotation_alone_changes_nothing():
    """The forces follow from the signed torque: the same T_1 with the other sense is the same
    mesh force (gear 1 is then driven instead of driving)."""
    p = CASES["case2"]
    assert FAMILY.mesh_forces(p, T, 1) == FAMILY.mesh_forces(p, T, -1)


# ----------------------------------------------------------------------------------------------
# bad inputs, and the contact chain
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kw", [dict(rot1=0), dict(rot1=2), dict(rot1=1, hand1=0), dict(rot1=1, hand1="right")])
def test_bad_sense_or_hand_is_refused(kw):
    with pytest.raises(ValueError, match="must be"):
        FAMILY.mesh_forces(CASES["case2"], T, **kw)


def test_mesh_forces_are_frozen():
    m = FAMILY.mesh_forces(CASES["case2"], T, 1)
    with pytest.raises(AttributeError):
        m.F_bn_N = 0.0


def test_normal_force_feeds_the_contact_inputs():
    """``MeshForces.F_bn_N`` is the load of ``contact_inputs``: at the start of meshing of case 2 the
    load per unit length is 48.618 N/mm (the value pinned for F_bn = 1847.109 N in
    ``test_contact_inputs.py``), and the line loads times the summed line lengths give F_bn back."""
    p = CASES["case2"]
    m = FAMILY.mesh_forces(p, T, 1)
    out = FAMILY.contact_inputs(p, "gear1", bc.N_RPM, m.F_bn_N, n_points=5)
    assert out["w_N_per_mm"][0] == pytest.approx(48.61842917026523, rel=1e-12)
    assert out["w_N_per_mm"] * out["l_sum_mm"] == pytest.approx(m.F_bn_N, rel=1e-12)
