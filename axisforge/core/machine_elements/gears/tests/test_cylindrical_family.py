"""Tests of the gear family contract (``base``) and of the cylindrical family (``cylindrical``).

Four groups: the contract (registry, immutable objects, one family for external and internal
gears, bad inputs); closed-form identities of the involute geometry; the geometry rulings of
2026-10-15 (closed forms, relations between stored quantities, SDP Table 4-4); and one test per entry of
``cylindrical._LEGACY`` and per other unruled item, which pins TODAY's value on a baseline case. Where
the review gives the alternative as a relation between quantities of the code, that relation is a
second test marked ``xfail(strict=True)`` with its question id. The numbers are those of the
baseline CSVs (``tests/data/gears_baseline``) or of ``scripts/questions_values_step3.py`` (vault).
Units: lengths [mm], angles [rad], torques [N m], forces [N].
"""
import dataclasses
import math

import numpy as np
import pytest

from axisforge.core.machine_elements.gears import (
    CylindricalFamily, CylindricalGearData, Gear, GearFamily, GearPair, cylindrical)
from axisforge.core.machine_elements.gears.base import _GEAR_FAMILY_REGISTRY
from axisforge.core.machine_elements.gears.tests import baseline_cases as bc

FAMILY = CylindricalFamily()
REL = bc.REL_TOL


def pair(g1, g2, **kw):
    return GearPair.assemble(FAMILY, gear1=g1, gear2=g2, **kw)


def xfail(question):
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason=question)


# the baseline pairs: C14 external spur (z 16/24, mn 4.5 mm), H501 external helical (z 20/30, mn 3.5 mm,
# beta 15 deg), internal spur and internal helical (z 20/-60 and 24/-60, mn 2 mm, beta 0 / 15 deg). Old
# case 4 is an equivalence record that fails validate() (Q39, Q24), so it is assembled with check=False.
PAIRS = {"case1": pair(*bc.pinion_wheel("C14")), "case2": pair(*bc.pinion_wheel("H501")),
         "case4": pair(*bc.make_internal(0.0), check=False), "case5": pair(*bc.make_internal(15.0)),
         "case4n": pair(*bc.make_internal(0.0, z1=24)), "case5n": pair(*bc.make_internal(15.0, z1=24))}
# internal pair z 20/-60, mn 2 mm, x 0.2/0.2 (x1 + x2 != 0; no baseline case)
INTERNAL_SHIFTED = (bc.gear(2.0, 20, x=0.2, b=20.0), bc.gear(2.0, -60, x=0.2, b=20.0))


@pytest.fixture(scope="module")
def case2():
    return PAIRS["case2"]


@pytest.fixture(scope="module")
def case4():
    return PAIRS["case4"]


@pytest.fixture(scope="module")
def case5():
    return PAIRS["case5"]


# ----------------------------------------------------------------------------------------------
# contract
# ----------------------------------------------------------------------------------------------
def test_family_is_registered_with_its_traits():
    assert _GEAR_FAMILY_REGISTRY["CylindricalFamily"] is CylindricalFamily
    assert FAMILY.name == "cylindrical"
    assert FAMILY.axes == "parallel"


def test_a_family_defines_no_gear_alone_by_default():
    class PairOnly(GearFamily):
        name = "pair_only"
        axes = "intersecting"
        assemble_pair = mesh_forces = validate_pair = pair_summary = pair_repr = None

    with pytest.raises(NotImplementedError):
        Gear.assemble(PairOnly(), CylindricalGearData(2.0, 20))


def test_external_and_internal_gears_come_from_the_one_family(case4):
    pinion, ring = case4.gear1, case4.gear2
    assert type(pinion) is type(ring) is Gear
    assert pinion.family is ring.family and case4.family is FAMILY
    assert type(pinion.family) is CylindricalFamily
    assert (pinion.z, ring.z) == (20, -60)
    # diameters are magnitudes; the sign of z puts the tip of a ring inside its reference circle
    assert ring.d == pytest.approx(120.0, rel=REL) and ring.da < ring.d < ring.df
    assert pinion.df < pinion.d < pinion.da


def test_assembled_objects_refuse_writes(case2):
    with pytest.raises(AttributeError, match="immutable"):
        case2.al = 100.0
    with pytest.raises(AttributeError, match="immutable"):
        case2.driver = "gear2"            # the driver is an argument of forces(), not stored geometry
    with pytest.raises(AttributeError, match="immutable"):
        case2.gear1.x = 0.5
    with pytest.raises(dataclasses.FrozenInstanceError):
        case2.gear1.data.x = 0.5
    assert not hasattr(case2, "driver") and not hasattr(case2, "T")


def test_gear_data_is_keyword_only_after_z():
    with pytest.raises(TypeError):
        CylindricalGearData(2.0, 20, 0.3)     # the third positional argument was Ca in one old class and x in the other
    assert CylindricalGearData(2.0, 20, x=0.3).x == 0.3


def test_summary_and_warnings(case2):
    assert case2.warnings == () and case2.gear1.warnings == ()
    assert "family : cylindrical" in case2.gear1.summary()
    assert case2.summary() == FAMILY.pair_summary(case2)


def test_bad_inputs():
    with pytest.raises(ValueError, match="z must be != 0"):
        bc.gear(2.0, 0)
    bad = CylindricalGearData(-1.0, 20, b=-2.0, alpha_n_deg=60.0, beta_n_deg=50.0, Ra=-1.0, Rq=-1.0, Rz=-1.0, label="bad")
    assert bad.validate() == [
        "bad: mn must be > 0, got -1.0", "bad: b must be >= 0, got -2.0", "bad: alpha_n_deg out of range (0, 45), got 60.0",
        "bad: beta_n_deg out of range [0, 45), got 50.0", "bad: Ra must be >= 0, got -1.0", "bad: Rq must be >= 0, got -1.0",
        "bad: Rz must be >= 0, got -1.0"]
    g = bc.gear(2.0, 20, b=10.0)
    with pytest.raises(ValueError, match="Module mismatch"):
        pair(g, bc.gear(2.5, 30))
    with pytest.raises(ValueError, match="Pressure angle mismatch"):
        pair(g, bc.gear(2.0, 30, alpha_n_deg=25.0))
    with pytest.raises(ValueError, match="Helix angle mismatch"):
        pair(g, bc.gear(2.0, 30, beta_n_deg=10.0))
    with pytest.raises(ValueError, match="Internal gear must have more teeth"):
        pair(g, bc.gear(2.0, -20, check=False))       # the ring alone is invalid too (tip below its base circle)
    with pytest.raises(ValueError, match=r"gear 1 must be the pinion: \|z1\| <= \|z2\|"):
        pair(bc.gear(2.0, 30, b=10.0), g)
    with pytest.raises(ValueError, match="driver must be 'gear1' or 'gear2'"):
        FAMILY.forces(pair(g, bc.gear(2.0, 30, b=10.0)), 10.0, driver="pinion")


# ----------------------------------------------------------------------------------------------
# closed forms of the involute geometry (no standard needed beyond the definitions)
# ----------------------------------------------------------------------------------------------
def test_spur_gear_closed_forms():
    """Spur gear, mn 2 mm, z 20, default rack: d = mn z; db = d cos(alpha_n); pb = pi mn cos(alpha_n)."""
    g = bc.gear(2.0, 20)
    alpha = math.radians(20.0)
    assert g.d == pytest.approx(40.0, rel=REL)
    assert g.db == pytest.approx(40.0 * math.cos(alpha), rel=REL)
    assert g.pb == pytest.approx(math.pi * 2.0 * math.cos(alpha), rel=REL) and g.pbt == pytest.approx(g.pb, rel=REL)
    assert g.df == pytest.approx(40.0 - 2 * 2.0 * 1.25, rel=REL)        # root: d - 2 mn (haP + cP)
    assert math.isinf(g.pz) and math.isinf(g.px)                       # a spur gear has no lead


def test_pair_without_profile_shift_works_at_the_reference_centre_distance():
    """x1 + x2 = 0: alpha_wt = alpha_t and a_w = a (ISO_21771_2007, Eq. (55) with a zero sum)."""
    p = pair(bc.gear(2.0, 20, b=10.0, beta_n_deg=15.0), bc.gear(2.0, 47, b=10.0, beta_n_deg=15.0))
    assert p.alphatw == pytest.approx(p.gear1.alphat, rel=1e-10)        # through the root solver
    assert p.al == pytest.approx(p.a, rel=1e-10)
    assert p.dl1 == pytest.approx(p.gear1.d, rel=1e-10) and p.dl2 == pytest.approx(p.gear2.d, rel=1e-10)


@pytest.mark.parametrize("beta", (0.0, 15.0))
def test_line_of_action_identities(beta, case4):
    """The pitch point divides the base-tangent segment (T1C + T2C = T1T2, signed for a ring), the
    path points are one transverse base pitch apart, and the working pitch circles roll: dw2/dw1 = z2/z1."""
    ext = pair(*bc.pinion_wheel("H501" if beta else "C14"))
    for p in (ext, case4):
        assert p.T1C + p.T2C == pytest.approx(p.T1T2, rel=1e-12)
        assert p.AD == pytest.approx(p.gear1.pbt, rel=1e-12) and p.AE - p.AB == pytest.approx(p.gear1.pbt, rel=1e-12)
        assert p.dl2 / p.dl1 == pytest.approx(abs(p.gear2.z) / p.gear1.z, rel=1e-12)
        assert p.inv_alphatw == pytest.approx(math.tan(p.alphatw) - p.alphatw, rel=1e-12)
    assert ext.epslon_alpha == pytest.approx(ext.AE / ext.gear1.pbt, rel=1e-12)


def test_values_at_the_reference_diameter(case5):
    """At dy = d the pressure angle, helix angle and pitch are the reference ones; the radius of
    curvature of a ring flank is negative (concave)."""
    for g in (case5.gear1, case5.gear2):
        v = FAMILY.values_at(g, g.d)
        assert v["alpha_yt"] == pytest.approx(g.alphat, rel=1e-12)
        assert v["beta_y"] == pytest.approx(g.beta, rel=1e-12) and v["p_yt"] == pytest.approx(g.pt, rel=1e-12)
        assert math.copysign(1.0, v["L_y"]) == math.copysign(1.0, g.z)


# ----------------------------------------------------------------------------------------------
# geometry rulings of 2026-10-15 (Q04, Q27, Q21, Q22, Q26, Q57, Q23, Q03): one expression per
# quantity with signed z; each test is a closed form, a relation between stored quantities or a
# published value
# ----------------------------------------------------------------------------------------------
def _bottom_clearances(p):
    """Radial gaps [mm] between each tip circle and the root circle of the mate, at the working
    centre distance: (tip of gear 1 / root of gear 2, tip of gear 2 / root of gear 1). External
    pair: a_w - r_a1 - r_f2 and a_w - r_a2 - r_f1; internal pair (the pinion inside the ring):
    r_f2 - a_w - r_a1 and r_a2 - a_w - r_f1."""
    ra1, ra2, rf1, rf2 = p.da1_w / 2.0, p.da2_w / 2.0, p.gear1.rf, p.gear2.rf
    if p.gear2.z > 0:
        return p.al - ra1 - rf2, p.al - ra2 - rf1
    return rf2 - p.al - ra1, ra2 - p.al - rf1


@pytest.mark.parametrize("z2", (60, -60))
def test_bottom_clearance_at_the_reference_centre_distance(z2):
    """Q04, Q27: the tip is d + 2 (z/abs(z)) mn (haP + x) (ISO_21771_2007, 4.5.3, Eq. (33)) and the
    root d + 2 (z/abs(z)) mn (x - haP - cP) (Eq. (34)), so with x = 0 (a_w = a) each tip clears the
    root of its mate by the bottom clearance cP mn of the basic rack [mm]. Pinion z 24 (the internal
    pair 24/-60 of case 4n is valid)."""
    p = pair(bc.gear(2.0, 24, b=10.0), bc.gear(2.0, z2, b=10.0))
    assert p.al == pytest.approx(p.a, rel=1e-12)
    cP_mn = p.gear1.cP * p.gear1.mn                           # 0.25 x 2 mm = 0.5 mm
    assert _bottom_clearances(p) == (pytest.approx(cP_mn, abs=1e-9), pytest.approx(cP_mn, abs=1e-9))
    for g in (p.gear1, p.gear2):
        assert g.da == pytest.approx(g.d + 2.0 * math.copysign(1.0, g.z) * g.mn * (g.haP + g.x), rel=REL)


def test_case_1_transverse_contact_ratio_is_gearpies():
    """Q04: with the ISO tip, baseline case 1 (C14, z 16/24, x 0.1817/0.1715) has eps_alpha 1.4624, the
    value of GEARpie (CLASSES of the GEARpie runner, baseline/arrays)."""
    assert PAIRS["case1"].epslon_alpha == pytest.approx(1.4624308892703048, rel=1e-9)


def test_ring_has_the_base_cylinder_of_its_pinion(case5):
    """Q21: sin(beta_b) = sin(beta) cos(alpha_n), alpha_t from tan(alpha_n) / cos(beta) and d_b = d cos(alpha_t)
    for both kinds (ISO_21771_2007, Eq. (5)-(7), (13), (19)): two gears in mesh have the same base helix
    angle, transverse pressure angle and transverse base pitch."""
    pinion, ring = case5.gear1, case5.gear2
    assert ring.betab == pytest.approx(math.asin(math.sin(math.radians(15.0)) * math.cos(math.radians(20.0))), rel=REL)
    for name in ("betab", "alphat", "pbt"):
        assert getattr(ring, name) == pytest.approx(getattr(pinion, name), rel=REL)
    assert ring.db == pytest.approx(ring.d * math.cos(ring.alphat), rel=REL)


@pytest.mark.parametrize("beta", (0.0, 15.0))
def test_working_pitch_circles_roll_on_each_other(beta):
    """Q21, Q03: d_w = d_b / cos(alpha_wt) for each gear (ISO_21771_2007, Eq. (56)-(57)); the working
    pitch circles are tangent at C: d_w2 + s2 d_w1 = 2 a_w (s2 = -1 for a ring)."""
    for p in (pair(*bc.pinion_wheel("H501" if beta else "C14")), PAIRS["case5n" if beta else "case4n"]):
        s2 = math.copysign(1.0, p.gear2.z)
        assert p.dl1 == pytest.approx(p.gear1.db / math.cos(p.alphatw), rel=REL)
        assert p.dl2 == pytest.approx(p.gear2.db / math.cos(p.alphatw), rel=REL)
        assert p.dl2 + s2 * p.dl1 == pytest.approx(2.0 * p.al, rel=1e-12)


def test_gear_ratio_is_signed(case2, case4):
    """Q03: u = z2 / z1 (ISO_21771_2007, 5.2.1, Eq. (52)), negative for an internal pair."""
    assert case2.u == 1.5 and case4.u == -3.0


def test_gear_1_must_be_the_pinion():
    """Q03 (José answer 7): gear 1 is the pinion. The pairs of the deleted baseline cases 3b, 3c and
    7c, 7d (H501 with the wheel, z 30, as gear 1) are refused; the same mesh is built with the pinion
    as gear 1 (cases 3a, 7e, 7f)."""
    pinion, wheel = bc.pinion_wheel("H501")
    with pytest.raises(ValueError, match=r"gear 1 must be the pinion: \|z1\| <= \|z2\|, got z1=30, z2=20"):
        pair(wheel, pinion)                                                  # cases 3b, 3c
    for rot in (1, -1):
        with pytest.raises(ValueError, match="gear 1 must be the pinion"):
            bc.build_case7("wheel", rot)                                     # cases 7c, 7d
    with pytest.raises(ValueError, match=r"gear 1 must be the external gear \(pinion\), got z1=-60"):
        pair(bc.gear(2.0, -60), bc.gear(2.0, 20))                            # a ring as gear 1 (D-0010 item 10)
    assert pair(pinion, bc.gear(3.5, 20, beta_n_deg=15.0)).u == 1.0        # equal tooth numbers are accepted


@pytest.mark.parametrize("case", PAIRS)
def test_length_of_the_path_of_contact_is_A_to_E(case):
    """Q22: g_alpha = 1/2 (sqrt(d_a1**2 - d_b1**2) + s2 (sqrt(d_a2**2 - d_b2**2) - 2 a_w sin(alpha_wt)))
    (ISO_21771_2007, Eq. (77), stored a_w positive), positive and equal to the stored distance A-E for
    both kinds; for an external pair it is the sum of the two addendum parts."""
    p = PAIRS[case]
    assert p.galpha > 0.0
    assert p.galpha == pytest.approx(p.AE, rel=1e-12)
    if p.gear2.z > 0:
        assert p.galpha == pytest.approx(p.galpha1 + p.galpha2, rel=1e-12)
    assert p.epslon_alpha == pytest.approx(p.galpha / p.gear1.pbt, rel=1e-12)    # Eq. (90)


@pytest.mark.parametrize("case", PAIRS)
def test_reduced_radius_at_the_pitch_point_uses_the_signed_ring_radius(case):
    """Q22: ReqI = 1 / (1/rho_C1 + 1/rho_C2) with rho_C = r_w sin(alpha_wt), rho_C2 < 0 for a ring
    (ISO_21771_2007, Eq. (81)-(82) and NOTE; D-0010 item 8): the stored T1C, T2C, and the reduced radius
    of ``curvature_radii`` at C [mm]. Case 4: 10.2606 mm."""
    p = PAIRS[case]
    assert p.ReqI == pytest.approx(1.0 / (1.0 / p.T1C + 1.0 / p.T2C), rel=1e-12)
    assert p.ReqI == pytest.approx(FAMILY.curvature_radii(p, 0.0)["rho_red_t_mm"], rel=1e-12)
    if case == "case4":
        assert p.ReqI == pytest.approx(10.260604299772375, rel=1e-9)


@pytest.mark.parametrize("which", ("case2", "case5", "internal_x0.2"))
def test_pair_at_the_centre_distance_derived_from_its_shifts_is_consistent(which):
    """Q26, Q57: imposing the centre distance the family derived from x1 + x2 gives the same working
    pressure angle, no profile shift message (ISO_21771_2007, Eq. (62) with the signed z1 + z2), zero
    backlash and no warning, for both kinds."""
    p = PAIRS[which] if which in PAIRS else pair(*INTERNAL_SHIFTED)
    again = pair(p.gear1, p.gear2, al=p.al)
    assert again.alphatw == pytest.approx(p.alphatw, rel=1e-10)
    assert again.jbn == pytest.approx(0.0, abs=1e-9) and again.warnings == ()
    assert p.jbn is None
    assert again.profile_shift_check[0] == pytest.approx(p.gear1.x + p.gear2.x, abs=1e-9)


@pytest.mark.parametrize("which, sign", (("case2", 1.0), ("case5", -1.0), ("internal_x0.2", -1.0)))
def test_backlash_of_a_larger_imposed_centre_distance(which, sign):
    """Q57, Q26: with a_w larger than the zero-backlash centre distance a_w0 the normal backlash j_bn of
    ISO_21771_2007, 5.3, Eq. (63) is positive for an external pair and negative for an internal pair
    (moving the pinion away from the ring centre pushes its teeth into the ring). To first order
    abs(j_bn) = 2 (a_w - a_w0) sin(alpha_wt) cos(beta_b) (ISO_21771_2007, 5.5.2-5.5.3, Eq. (102), (104):
    backlash from a change of centre distance). The pair is built and carries ONE warning with the sum implied by
    a_w (Eq. (62)) and j_bn; ``validate()`` has no message for it (lead decision, 2026-10-15)."""
    p = PAIRS[which] if which in PAIRS else pair(*INTERNAL_SHIFTED)
    for factor, rel in ((1.002, 2e-2), (1.0 + 1e-6, 1e-5)):
        q = pair(p.gear1, p.gear2, al=p.al * factor)
        assert math.copysign(1.0, q.jbn) == sign
        assert abs(q.jbn) == pytest.approx(2.0 * (q.al - p.al) * math.sin(q.alphatw) * math.cos(q.gear1.betab), rel=rel)
        assert len(q.warnings) == 1 and q.warnings[0].startswith("profile shifts do not match the centre distance")
        assert "implies x1 + x2" in q.warnings[0] and "jbn" in q.warnings[0]
        assert q.validate() == p.validate()             # the mismatch adds nothing to validate()
        assert "warning: profile shifts do not match" in q.summary()


def test_tip_alteration_gives_the_bottom_clearance_at_the_working_centre_distance():
    """Q23: k = s2 (a_w - a) / mn - (x1 + x2), the same k on both tips, d_a = d + 2 (z/abs(z)) mn (haP + x + k)
    (ISO_21771_2007, 4.5.2, Eq. (33): k < 0 shortens the addendum): with addendum_reduction both
    bottom clearances at a_w are cP mn, external (case 1, x 0.1817/0.1715) and internal (z 20/-60,
    x 0.2/0.2). Without it, k = 0 and each clearance is cP mn + k mn (here k < 0 for the external pair,
    the tips are shortened; k > 0 for the internal one). No library source needed: geometry of the
    two circles."""
    for gears in ((PAIRS["case1"].gear1, PAIRS["case1"].gear2), INTERNAL_SHIFTED):
        p = pair(*gears, addendum_reduction=True)
        cP_mn = p.gear1.cP * p.gear1.mn
        assert p.k != 0.0
        assert _bottom_clearances(p) == (pytest.approx(cP_mn, abs=1e-9), pytest.approx(cP_mn, abs=1e-9))
        assert p.da1_w == pytest.approx(p.gear1.da + 2.0 * p.gear1.mn * p.k, rel=REL)
        assert p.da2_w == pytest.approx(p.gear2.da + 2.0 * math.copysign(1.0, p.gear2.z) * p.gear2.mn * p.k, rel=REL)
        plain = pair(*gears)
        assert plain.k == 0.0
        assert _bottom_clearances(plain) == (pytest.approx(cP_mn + p.k * p.gear1.mn, abs=1e-9),) * 2


def test_tip_alteration_of_sdp_table_4_4():
    """Q23, Q04: SDP-SI "Elements of Metric Gear Technology", Table 4-4 (external spur, m 3 mm, 20 deg,
    z 12/24, x 0.6/0.36, tip from the working centre distance): alpha_w 26.0886 deg, a_w 56.4999 mm,
    d_a 44.840/79.400 mm, d_f 32.100/66.660 mm (printed rounding); k = -0.12671 in the ISO sign (vault
    ``80_Inbox/2026-10-15/internal_gears_validation_khk.md``)."""
    p = pair(bc.gear(3.0, 12, x=0.6), bc.gear(3.0, 24, x=0.36), addendum_reduction=True)
    assert math.degrees(p.alphatw) == pytest.approx(26.0886, abs=5e-5)
    assert p.al == pytest.approx(56.4999, abs=5e-5)
    assert (p.da1_w, p.da2_w) == (pytest.approx(44.840, abs=5e-4), pytest.approx(79.400, abs=5e-4))
    assert (p.gear1.df, p.gear2.df) == (pytest.approx(32.100, abs=5e-4), pytest.approx(66.660, abs=5e-4))
    assert p.k == pytest.approx(-0.12671, abs=5e-6)


# ----------------------------------------------------------------------------------------------
# unit guard: one test per public function that takes a torque
# ----------------------------------------------------------------------------------------------
def test_forces_takes_the_torque_in_newton_metre(case2):
    """Baseline case 2, 60.75 N m on gear 1: Ft = 1659.83 N = 2000 T / d_w1. The same number given in
    N mm (60750) gives 1.66e6 N: nothing converts silently."""
    F = FAMILY.forces(case2, bc.T_NM, 0.0, 1)
    assert F["Ft"] == pytest.approx(1659.8313933933089, rel=REL)
    assert F["Ft"] == pytest.approx(2000.0 * bc.T_NM / case2.dl1, rel=REL)          # T [N m], d_w [mm]
    assert F["T_out"] == pytest.approx(91.125, rel=REL)                             # [N m]
    assert FAMILY.forces(case2, 1000.0 * bc.T_NM, 0.0, 1)["Ft"] == pytest.approx(1659831.3933933089, rel=REL)


# ----------------------------------------------------------------------------------------------
# _LEGACY: one test per entry (today's value), and the alternative where it is a relation
# ----------------------------------------------------------------------------------------------
def test_every_legacy_entry_has_both_kinds_and_a_test():
    assert all(set(entry) == {"external", "internal"} for entry in cylindrical._LEGACY.values())
    tested = {name[len("test_legacy_"):] for name in globals() if name.startswith("test_legacy_")}
    assert set(cylindrical._LEGACY) <= tested


def test_legacy_virtual_teeth(case5):
    """Q25 (I5): a ring has no virtual number of teeth, so the Henriot split raises AttributeError."""
    assert case5.gear1.zn == pytest.approx(20 / math.cos(math.radians(15.0)) ** 3, rel=REL)
    assert not hasattr(case5.gear2, "zn") and not hasattr(case5.gear2, "de")
    with pytest.raises(AttributeError):
        FAMILY.correction_for_gs_equilibrium(case5.gear1, case5.gear2)
    with pytest.raises(AttributeError):
        pair(case5.gear1, case5.gear2, equalise_gs=True)


def test_legacy_beta_range():
    """Q55 (José, 2026-10-15): keep. A negative helix angle is reported for an external gear and accepted for a ring."""
    assert bc.gear(2.0, 60, beta_n_deg=-5.0, label="g", check=False).validate() == ["g: beta_n_deg out of range [0, 45), got -5.0"]
    assert bc.gear(2.0, -60, beta_n_deg=-5.0, label="g").validate() == []


def test_legacy_gear_limits():
    """Q55 (José, 2026-10-15): keep. Undercut is checked on an external gear only; the tip of a ring must lie
    between its base and reference circles."""
    assert bc.gear(2.0, 8, label="g", check=False).validate() == ["g: gear is undercut — z=8, mn=2.0, alpha=20.0°, beta=0.0°, x=0.0"]
    assert FAMILY.undercutting(bc.gear(2.0, 8, check=False)) and not FAMILY.undercutting(bc.gear(2.0, 20))
    assert bc.gear(2.0, -60, x=-2.0, label="g", check=False).validate() == [
        "g: tip diameter da=124.0000 ≥ pitch diameter d=120.0000. Internal gear addendum must reduce da below d."]


def test_legacy_force_input(case2, case5):
    """Q03 (X1): the driver changes the forces of an internal pair and not those of an external pair
    (the alternative for the external pair is ``test_alternative_wheel_driver_takes_the_torque_on_the_wheel``)."""
    assert FAMILY.forces(case2, bc.T_NM, driver="gear2") == FAMILY.forces(case2, bc.T_NM, driver="gear1")
    assert FAMILY.forces(case5, bc.T_NM, driver="gear1")["Ft"] == pytest.approx(2933.999697353043, rel=1e-11)   # Q21: r_w1 of one base-cylinder rule
    ring_drives = FAMILY.forces(case5, bc.T_NM, driver="gear2")
    assert ring_drives["Ft"] == pytest.approx(977.9998991176811, rel=REL)
    assert ring_drives["T_out"] == pytest.approx(-20.25, rel=REL)


def test_contact_beyond_a_base_tangent_point_is_an_error(case4):
    """Q39: the path A..E must lie between the base-tangent points, T1A >= 0 and s2 T2E >= 0, for both kinds.
    Old case 4 starts 0.075 mm beyond T1 (for an internal pair this is SDP-SI Eq. (5-2), one message); the
    external pair z 7/60 (undercut pinion) 2.87 mm beyond T1; the external pair z 12/12, x 0.6/-0.6 ends beyond
    T2 (T1A > 0, s2 T2E < 0). Every other pair of PAIRS validates to []."""
    assert case4.validate() == ["Interference: the path of contact passes a base-tangent point (T1A = -0.0749 mm, "
                                "s2 T2E = 25.1172 mm; both must be >= 0). For an internal pair T1A < 0 is the "
                                "involute interference of SDP-SI Eq. (5-2)."]
    low = pair(bc.gear(2.0, 7, b=10.0, label="p", check=False), bc.gear(2.0, 60, b=10.0, label="w"), check=False)
    assert low.T1A < 0.0 and any(m.startswith("Interference: the path of contact passes") for m in low.validate())
    end = pair(bc.gear(2.0, 12, x=0.6, check=False), bc.gear(2.0, 12, x=-0.6, check=False), check=False)
    assert end.T1A > 0.0 > end.T2E
    assert any(m.startswith("Interference: the path of contact passes") for m in end.validate())
    for p in PAIRS.values():
        if p is not case4:
            assert p.T1A >= 0.0 and math.copysign(1.0, p.gear2.z) * p.T2E >= 0.0 and p.validate() == []


KHK_LIMITS = {60: (21, 51, 43), 80: (20, 72, 64), 100: (19, 92, 84), 120: (19, 112, 104), 160: (19, 152, 144),
              200: (18, 192, 184)}


@pytest.mark.parametrize("z2", KHK_LIMITS)
def test_internal_interference_reproduces_the_khk_limits(z2):
    """Q24: KHK catalogue, internal gears, p. 208, "Allowable Mating Pinions" (m 1, 20 deg, x = 0): smallest pinion
    without involute interference, largest without trochoid and without trimming interference, for z2 = 60..200,
    from the tests of SDP-SI section 5, Eq. (5-2)-(5-10) on the pairs z1/-z2 (vault
    ``80_Inbox/2026-10-15/internal_gears_validation_khk.md``). Involute interference is the one Q39 message
    (T1A < 0 = SDP Eq. (5-2))."""
    flags = {}
    for z1 in range(8, z2 - 1):
        msgs = pair(bc.gear(1.0, z1, check=False), bc.gear(1.0, -z2), check=False).validate()
        flags[z1] = [not any("SDP-SI Eq. (5-2)" in m for m in msgs)] + [
            not any(m.startswith(f"{kind} interference of the internal pair") for m in msgs) for kind in ("trochoid", "trimming")]
    involute_lower = min(z1 for z1, ok in flags.items() if ok[0])
    trochoid_upper = max(z1 for z1, ok in flags.items() if ok[1])
    trimming_upper = max(z1 for z1, ok in flags.items() if ok[2])
    assert (involute_lower, trochoid_upper, trimming_upper) == KHK_LIMITS[z2]
    # one limit each, not a band
    assert all(ok[0] for z1, ok in flags.items() if z1 >= involute_lower)
    assert all(ok[1] for z1, ok in flags.items() if z1 <= trochoid_upper)
    assert all(ok[2] for z1, ok in flags.items() if z1 <= trimming_upper)


@pytest.mark.parametrize("z2", (60, 100, 200))
@pytest.mark.parametrize("x", (0.0, 0.3))
@pytest.mark.parametrize("beta", (0.0, 15.0))
def test_sdp_involute_test_is_the_q39_criterion(z2, x, beta):
    """For an internal pair the margin of SDP-SI Eq. (5-2), z1/z2 - 1 + tan(alpha_a2)/tan(alpha_w), equals
    T1A / (r_b2 tan(alpha_wt)): the involute-interference test of SDP is the Q39 criterion T1A >= 0 (one message,
    lead decision 2026-10-15). Pinions z1 = 0.3, 0.5, 0.6 z2, mn 2 mm, x1 = x, x2 = 0."""
    for z1 in (int(0.3 * z2), int(0.5 * z2), int(0.6 * z2)):
        p = pair(bc.gear(2.0, z1, x=x, beta_n_deg=beta, check=False), bc.gear(2.0, -z2, beta_n_deg=beta), check=False)
        alpha_a2 = math.acos(p.gear2.rb / (p.da2_w / 2.0))
        margin = z1 / z2 - 1.0 + math.tan(alpha_a2) / math.tan(p.alphatw)
        assert margin == pytest.approx(p.T1A / (p.gear2.rb * math.tan(p.alphatw)), rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("case", ("case4n", "case5n"))
def test_new_internal_cases_are_valid(case):
    """Q15 (José answer 4): the internal cases 24/-60 spur and helical (beta 15 deg) replace old case 4 for contact
    results: the path lies between the base-tangent points, galpha = A-E, ReqI = curvature at C, no interference
    of SDP-SI section 5, and the pair assembles with the checks."""
    p = PAIRS[case]
    assert p.T1A >= 0.0 and -p.T2E >= 0.0
    assert p.galpha == pytest.approx(p.AE, rel=1e-12)
    assert p.ReqI == pytest.approx(FAMILY.curvature_radii(p, 0.0)["rho_red_t_mm"], rel=1e-12)
    assert p.warnings == ()


def test_legacy_text(case2, case4):
    """Q50: the texts keep the class names of the two pair classes the family replaces."""
    assert repr(case2).startswith("HelicalGearMeshing(z1=20, z2=30, ")
    assert repr(case4).startswith("InternalGearMeshing(z1=20, z2=-60, ")
    assert case2.summary().splitlines()[0].startswith("── HelicalGearMeshing ─")
    assert case4.summary().splitlines()[1] == "  gear1 (ext): z=20, mn=2.0, x=0.0, β=0.0°"


def test_renamed_text_rows_are_counted():
    """Q50: the baseline rows that name a removed class and are compared against the new text."""
    cases = [p.stem for p in sorted(bc.DATA_DIR.glob("*.csv"))]
    got = bc.run_baseline()[0]
    for old, (new, count) in bc.RENAMED_TEXT.items():
        hits = [(c, i) for c in cases for i, (_, _, text) in enumerate(bc.baseline_rows(c)) if text == old]
        assert len(hits) == count
        assert all(got[c][i][2] == new for c, i in hits)
    # the rows `driver` / `driven` of the pairs are the declared argument, not a result of the code
    names, count = bc.ECHOED_ROWS
    echoed = sum(1 for c in cases for obj, name, _ in bc.baseline_rows(c) if name in names and obj != "link")
    renamed = sum(n for _, n in bc.RENAMED_TEXT.values())
    total = sum(len(bc.baseline_rows(c)) for c in cases)
    assert echoed == count == 70                 # 8 rows deleted with cases 3b, 3c, 7c, 7d (Q03), 20 added in 4n, 5n (Q15)
    # Q46 (step 5): the rows of case 7 that are text or state of the two replaced system classes, counted
    # one by one in mechanical_system/tests/test_transmission_baseline.py
    renamed += sum(n for _, n in bc.RENAMED_PATTERNS.values())
    echoed += sum(n for _, n, _ in bc.ECHOED_ROWS_CASE7.values())
    assert (total, total - echoed - renamed, renamed, echoed) == (7866, 7770, 14, 82)     # rows: total, computed, renamed text, echoed


# ----------------------------------------------------------------------------------------------
# other unruled items kept as today
# ----------------------------------------------------------------------------------------------
def test_ring_tip_diameter_without_the_clearance(case4):
    """Q27: ring of baseline case 4, d = 120 mm, tip 116 mm = d - 2 mn (haP + x) (ISO_21771_2007, 4.5.3, Eq. (33))."""
    ring = case4.gear2
    assert ring.da == pytest.approx(116.0, rel=REL)
    assert ring.da == pytest.approx(ring.d - 2.0 * ring.mn * (ring.haP + ring.x), rel=REL)


def test_assemble_raises_one_error_with_every_message(case4):
    """Q51 (José, answer 3): Gear.assemble and GearPair.assemble run validate() and raise ONE ValueError with every
    message; check=False builds the object (equivalence records). Old case 4: the one Q39 / SDP Eq. (5-2) message."""
    with pytest.raises(ValueError) as err:
        pair(*bc.make_internal(0.0))
    assert str(err.value).splitlines() == case4.validate() and len(case4.validate()) == 1
    assert case4.T1A == pytest.approx(-0.07492456399709191, rel=1e-9)        # built with check=False
    with pytest.raises(ValueError, match="gear is undercut"):
        bc.gear(2.0, 8)


def test_low_contact_ratio_is_a_warning_not_an_error():
    """Answer 6 (D-0026 item 9): 1 <= eps_alpha < 1.2 goes to pair.warnings and the pair assembles; eps_alpha < 1
    stays an error of validate(). Spur pair z 30/60, mn 2 mm, haP 0.6 (eps_alpha 1.082)."""
    p = pair(bc.gear(2.0, 30, b=10.0, haP=0.6), bc.gear(2.0, 60, b=10.0, haP=0.6))
    assert 1.0 <= p.epslon_alpha < 1.2 and p.validate() == []
    assert len(p.warnings) == 1 and p.warnings[0].startswith("transverse contact ratio")
    short = pair(bc.gear(2.0, 30, b=10.0, haP=0.4), bc.gear(2.0, 60, b=10.0, haP=0.4), check=False)
    assert short.epslon_alpha < 1.0 and any("< 1.0" in m for m in short.validate()) and short.warnings == ()


def test_gear_messages_appear_once_in_the_pair():
    """Q56 (José, 2026-10-15): each message of a gear once in the pair."""
    low = pair(bc.gear(2.0, 8, b=10.0, label="p", check=False), bc.gear(2.0, 60, b=10.0, label="w"), check=False)
    assert sum("undercut" in m for m in low.validate()) == 1


def test_henriot_split_returns_non_physical_shifts_for_some_pairs():
    """Q13: outside the baseline the split does not raise; for z 12/54 (mn 2 mm, x = 0) it returns
    x1 = -x2 = 4.55, a profile shift of 4.5 modules (not usable)."""
    x1, x2 = FAMILY.correction_for_gs_equilibrium(bc.gear(2.0, 12, b=20.0, check=False), bc.gear(2.0, 54, b=20.0))
    assert (x1, x2) == (pytest.approx(4.548969365135395, rel=1e-9), pytest.approx(-4.548969365135395, rel=1e-9))
    with pytest.raises(RuntimeError, match="no solution for x2"):
        FAMILY.correction_for_gs_equilibrium(*bc.pinion_wheel("H501"))
