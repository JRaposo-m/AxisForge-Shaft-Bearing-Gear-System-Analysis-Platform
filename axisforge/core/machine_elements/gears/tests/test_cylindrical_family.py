"""Tests of the gear family contract (``base``) and of the cylindrical family (``cylindrical``).

Four groups: the contract (registry, immutable objects, one family for external and internal
gears, bad inputs); closed-form identities of the involute geometry; the geometry rulings of
2026-10-15 (closed forms, relations between stored quantities, SDP Table 4-4); and one test per former
``cylindrical._LEGACY`` entry (now one rule for both kinds) and per other unruled item, which pins TODAY's value on a baseline case. Where
the review gives the alternative as a relation between quantities of the code, that relation is a
second test marked ``xfail(strict=True)`` with its question id. The numbers are those of the
baseline CSVs (``gears/tests/data/gears_baseline``) or of ``scripts/questions_values_step3.py``
(vault).
Units: lengths [mm], angles [rad], torques [N m], forces [N].
"""
import dataclasses
import inspect
import math

import pytest

from axisforge.core.machine_elements.gears import (
    CylindricalFamily,
    CylindricalGearData,
    Gear,
    GearFamily,
    GearPair,
    ManufacturedCylindricalFamily,
    cylindrical,
)
from axisforge.core.machine_elements.gears.base import _GEAR_FAMILY_REGISTRY
from axisforge.core.machine_elements.gears.tests import baseline_cases as bc

FAMILY = CylindricalFamily()
REL = bc.REL_TOL


def pair(g1, g2, **kw):
    return GearPair.assemble(FAMILY, gear1=g1, gear2=g2, **kw)


def xfail(question):
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason=question)


def assert_undercut_only_as_cut(g):
    """Ruling B (José, 2026-10-21): the ideal gear ``g`` (external, x < x_Emin) carries no undercut warning or
    message; the same data as cut with zero allowances (x_Ei = x) is refused (MG-10)."""
    assert g.warnings == () and not any("undercut" in m for m in g.validate())
    with pytest.raises(ValueError, match="undercut at the lower allowance"):
        ManufacturedCylindricalFamily().from_ideal(g, E_sns=0.0, E_sni=0.0)


# the baseline pairs: C14 external spur (z 16/24, mn 4.5 mm), H501 external helical (z 20/30, mn 3.5 mm,
# beta 15 deg), internal spur and internal helical (z 20/-60 and 24/-60, mn 2 mm, beta 0 / 15 deg). Old
# case 4 is an equivalence record that fails validate() (Q39, Q24), so it is assembled with check=False.
PAIRS = {"case1": pair(*bc.pinion_wheel("C14")), "case2": pair(*bc.pinion_wheel("H501")),
         "case4": pair(*bc.make_internal(0.0), check=False), "case5": pair(*bc.make_internal(15.0)),
         "case4n": pair(*bc.make_internal(0.0, z1=24)), "case5n": pair(*bc.make_internal(15.0, z1=24))}
# internal pair z 20/-60, mn 2 mm, x 0.2/0.2 (x1 + x2 != 0; no baseline case)
INTERNAL_SHIFTED = (bc.gear(2.0, 20, x=0.2, b=20.0), bc.gear(2.0, -60, x=0.2, b=20.0))
# prefix of the explicit SDP-SI Eq. (5-2) message of ``_internal_interference`` (the equation label may change)
INVOLUTE_INTERFERENCE = "involute interference of the internal pair (SDP-SI Eq. ("


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
        case2.driver = "gear2"            # the driver is not stored geometry (Q03, D-0010 item 11)
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
        "bad: beta_n_deg out of range [0, 45), got 50.0",
        "bad: hand must be +1 (right) or -1 (left) for a helical gear (0 only for beta_n_deg = 0), got 0",
        "bad: Ra must be >= 0, got -1.0", "bad: Rq must be >= 0, got -1.0", "bad: Rz must be >= 0, got -1.0"]
    g = bc.gear(2.0, 20, b=10.0)
    with pytest.raises(ValueError, match="Module mismatch"):
        pair(g, bc.gear(2.5, 30))
    with pytest.raises(ValueError, match="Pressure angle mismatch"):
        pair(g, bc.gear(2.0, 30, alpha_n_deg=25.0))
    with pytest.raises(ValueError, match="Helix angle mismatch"):
        pair(g, bc.gear(2.0, 30, beta_n_deg=10.0, hand=-1))
    with pytest.raises(ValueError, match="Internal gear must have more teeth"):
        pair(g, bc.gear(2.0, -20, check=False))       # the ring alone is invalid too (tip below its base circle)
    with pytest.raises(ValueError, match=r"gear 1 must be the pinion: \|z1\| <= \|z2\|"):
        pair(bc.gear(2.0, 30, b=10.0), g)


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
    p = pair(bc.gear(2.0, 20, b=10.0, beta_n_deg=15.0, hand=1), bc.gear(2.0, 47, b=10.0, beta_n_deg=15.0, hand=-1))
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


def test_gear_values_at_the_reference_diameter(case5):
    """At dy = d the pressure angle, helix angle and pitch are the reference ones; the radius of
    curvature of a ring flank is negative (concave)."""
    for g in (case5.gear1, case5.gear2):
        v = FAMILY.gear_values_at(g, g.d)
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


def test_hand_of_helix_is_required_and_checked_at_assembly():
    """Q08 (D-0010 item 17): a helical gear needs hand +1 or -1 (assembly refuses it otherwise); an external helical
    pair has opposite hands, an internal one the same hand (ISO_21771_2007, 5.1.3, 5.1.4); a spur gear may have none."""
    with pytest.raises(ValueError, match="hand must be"):
        bc.gear(2.0, 20, beta_n_deg=15.0)
    assert bc.gear(2.0, 20).hand == 0
    right, left = bc.gear(2.0, 20, beta_n_deg=15.0, hand=1), bc.gear(2.0, 40, beta_n_deg=15.0, hand=-1)
    assert pair(right, left).gear2.hand == -1
    with pytest.raises(ValueError, match="Hand of helix mismatch"):
        pair(right, bc.gear(2.0, 40, beta_n_deg=15.0, hand=1))
    with pytest.raises(ValueError, match="Hand of helix mismatch"):
        pair(bc.gear(2.0, 24, b=20.0, beta_n_deg=15.0, hand=1), bc.gear(2.0, -60, b=20.0, beta_n_deg=15.0, hand=-1))
    assert pair(bc.gear(2.0, 24, b=20.0, beta_n_deg=15.0, hand=-1), bc.gear(2.0, -60, b=20.0, beta_n_deg=15.0, hand=-1)).validate() == []


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
    assert pair(pinion, bc.gear(3.5, 20, beta_n_deg=15.0, hand=-1)).u == 1.0        # equal tooth numbers are accepted


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
        # the input gears have k = 0; the gears returned by the pair carry its k (04a7edc), so their da is da_w
        g1, g2 = gears
        assert p.da1_w == pytest.approx(g1.da + 2.0 * g1.mn * p.k, rel=REL)
        assert p.da2_w == pytest.approx(g2.da + 2.0 * math.copysign(1.0, g2.z) * g2.mn * p.k, rel=REL)
        assert (p.gear1.da, p.gear2.da) == (pytest.approx(p.da1_w, rel=REL), pytest.approx(p.da2_w, rel=REL))
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
# profile shift split x_split (04a7edc): ISO_21771_2007, 5.6.3, Eq. (114)-(117), p. 44 (PDF p. 48). The split
# values are regression values computed by cowork on 2026-10-23: no published reference.
# ----------------------------------------------------------------------------------------------
def _g(z, x, **kw):
    """Gear of mn 2 mm, alpha_n 20 deg, b 20 mm."""
    return bc.gear(2.0, z, x=x, b=20.0, **kw)


def _root_specific_sliding(p):
    """zeta_f1 at A and zeta_f2 at E (gear 1 driving), recomputed from the path points and the signed radii of
    curvature: Eq. (116) 1 - rho2 / (u rho1) and Eq. (117) 1 - u rho1 / rho2 (u = z2 / z1 signed)."""
    g = FAMILY.path_points(p, "gear1")
    r = FAMILY.curvature_radii(p, [g["A"], g["E"]])
    rho1, rho2 = r["rho1_t_mm"], r["rho2_t_mm"]
    return 1.0 - rho2[0] / (p.u * rho1[0]), 1.0 - p.u * rho1[1] / rho2[1]


X_SPLIT_CASES = {
    "external_spur": ((20, 0.0, {}), (40, 0.0, {}), 0.2456),
    "external_helical": ((20, 0.3, dict(beta_n_deg=15.0, hand=1)), (40, 0.0, dict(beta_n_deg=15.0, hand=-1)), 0.3065),
    "internal_spur": ((20, 0.2, {}), (-60, 0.2, {}), 0.4027),
    "internal_helical": ((20, 0.2, dict(beta_n_deg=12.0, hand=1)), (-60, 0.2, dict(beta_n_deg=12.0, hand=1)), 0.3825),
}


@pytest.mark.parametrize("name", X_SPLIT_CASES)
def test_x_split_equal_root_specific_sliding(name):
    """ISO_21771_2007, 5.6.3, Eq. (116) = (117), p. 44 (PDF p. 48): with x_split="equal_specific_sliding" the
    specific sliding at the root of gear 1 (at A) equals that at the root of gear 2 (at E), recomputed
    independently (``_root_specific_sliding``); the sum x1 + x2 and so a_w are kept. x1: regression value
    (cowork 2026-10-23), no published reference."""
    (z1, x1, kw1), (z2, x2, kw2), x1_split = X_SPLIT_CASES[name]
    g1, g2 = _g(z1, x1, **kw1), _g(z2, x2, **kw2)
    p = pair(g1, g2, x_split="equal_specific_sliding")
    zeta_f1, zeta_f2 = _root_specific_sliding(p)
    assert zeta_f1 == pytest.approx(zeta_f2, abs=1e-9)
    assert p.x1 + p.x2 == pytest.approx(x1 + x2, abs=1e-12)
    assert p.al == pytest.approx(pair(g1, g2).al, abs=1e-9)
    assert p.x1 == pytest.approx(x1_split, abs=1e-4)


def test_x_split_symmetric_pair():
    """Equal teeth z 20/20: equal root specific sliding (ISO_21771_2007, 5.6.3, Eq. (116) = (117)) by symmetry
    needs equal shifts, x1 = x2 = (0 + 0.4) / 2 = 0.2. No library source for the value: symmetry."""
    p = pair(_g(20, 0.0), _g(20, 0.4), x_split="equal_specific_sliding")
    assert (p.x1, p.x2) == (pytest.approx(0.2, abs=1e-9), pytest.approx(0.2, abs=1e-9))


def test_x_split_with_al():
    """With al given the split uses the sum of ISO_21771_2007, 5.2.5, Eq. (62) (zero backlash at a_w), not the
    shifts of the gears, with one warning; no warning when the gears' sum already equals it. z 20/40, x 0/0,
    a_w 61.0 mm: x1 0.3884, x1 + x2 0.5298 (regression values, cowork 2026-10-23, no published reference)."""
    p = pair(_g(20, 0.0), _g(40, 0.0), al=61.0, x_split="equal_specific_sliding")
    assert p.x1 == pytest.approx(0.3884, abs=1e-4)
    assert p.x1 + p.x2 == pytest.approx(p.profile_shift_check[0], abs=1e-12)
    assert p.profile_shift_check[0] == pytest.approx(0.5298, abs=1e-4)
    assert p.jbn == pytest.approx(0.0, abs=1e-12)
    assert sum(w.startswith("x_split with al given") for w in p.warnings) == 1
    s = p.profile_shift_check[0]
    q = pair(_g(20, s / 2), _g(40, s / 2), al=61.0, x_split="equal_specific_sliding")
    assert not any(w.startswith("x_split with al given") for w in q.warnings)


def test_x_split_input_gears_unchanged():
    """The split builds new gears for the pair; the input gears keep their x and ``x_given`` records them. The
    default path (x_split None) returns the input gears. No library source: contract of ``assemble_pair``."""
    g1, g2 = _g(20, 0.3), _g(40, 0.0)
    p = pair(g1, g2, x_split="equal_specific_sliding")
    assert (g1.x, g2.x) == (0.3, 0.0)
    assert (p.gear1.x, p.gear2.x) == (p.x1, p.x2) and p.x1 != g1.x
    assert p.x_given == (0.3, 0.0) and p.x_split == "equal_specific_sliding"
    plain = pair(g1, g2)
    assert plain.x_split is None and plain.gear1 is g1 and plain.gear2 is g2


def test_x_split_unknown_criterion():
    """An unknown split criterion is refused. No library source: contract of ``assemble_pair``."""
    with pytest.raises(ValueError, match="x_split must be one of"):
        pair(_g(20, 0.0), _g(40, 0.0), x_split="equal_sliding_speed")


def test_x_split_with_addendum_reduction():
    """Split and tip alteration together, z 20/40, x 0.3/0: x1 0.3176, k -0.0102246 (ISO_21771_2007, 4.5.2;
    regression values, cowork 2026-10-23, no published reference), and the root specific sliding still equal
    (5.6.3, Eq. (116) = (117)) on the working tips."""
    p = pair(_g(20, 0.3), _g(40, 0.0), addendum_reduction=True, x_split="equal_specific_sliding")
    assert p.x1 == pytest.approx(0.3176, abs=1e-4)
    assert p.k == pytest.approx(-0.0102246, abs=1e-6)
    zeta_f1, zeta_f2 = _root_specific_sliding(p)
    assert zeta_f1 == pytest.approx(zeta_f2, abs=1e-9)


def test_working_gears_carry_k():
    """Tip alteration (04a7edc): with addendum_reduction the gears returned by the pair are new gears with the k of
    the pair, so their da is the working tip diameter, d + 2 (z/abs(z)) mn (haP + x + k) (ISO_21771_2007, 4.5.3,
    Eq. (33)); without it the pair returns the input gears and each keeps the k of its data (da1_w = da). Gear z 20,
    x 0.3, mn 2, data k -0.05: da = 40 + 4 (1 + 0.3 - 0.05) = 45.0 mm."""
    g1, g2 = _g(20, 0.3), _g(40, 0.0)
    p = pair(g1, g2, addendum_reduction=True)
    assert p.gear1.k == p.k and p.gear1 is not g1
    assert (p.gear1.da, p.gear2.da) == (pytest.approx(p.da1_w, rel=1e-12), pytest.approx(p.da2_w, rel=1e-12))
    plain = pair(g1, g2)
    assert plain.gear1 is g1 and plain.da1_w == pytest.approx(g1.da, rel=1e-12)
    gk = _g(20, 0.3, k=-0.05)
    q = pair(gk, g2)
    assert q.da1_w == pytest.approx(gk.da, rel=1e-12) and gk.da == pytest.approx(45.0, rel=1e-12)
    assert q.k == 0.0


# SDP-SI "Elements of Metric Gear Technology", section 5 (internal gears), Tables 5-1 and 5-4
# (06_Library/_incoming/SDPSI2026_MetricGearTechnology_InternalHelical.pdf, intake pending; vault
# ``80_Inbox/2026-10-15/internal_gears_validation_khk.md``). SDP writes z2 > 0 for the ring and the ring
# shift with the opposite sign: x2_ISO = -x2_KHK (ring addendum (1 - x2_KHK) m). Default rack: haP 1, cP 0.25.
def test_internal_pair_of_sdp_table_5_1():
    """SDP Table 5-1 (internal spur, m 3 mm, 20 deg, z 16/-24, x 0 / -0.5 ISO = +0.5 KHK), to the printed digits:
    inv alpha_wt 0.060401, alpha_wt 31.0937 deg, a_w 13.1683 mm, d_b 45.105/67.658, d_w 52.673/79.010,
    d_a 54.000/69.000, d_f 40.500/82.500 mm (Q21, Q26, Q04; d_w = d_b / cos alpha_wt).

    alpha_wt follows SDP's rounding chain: SDP inverts the ROUNDED inv alpha_wt 0.060401, which gives
    31.093674 deg (printed 31.0937); the exact involute from the data, 0.0604006632 (ISO_21771_2007, Eq. (62)),
    gives 31.093621 deg, which the code returns and the test asserts.

    Built with check=False because the code refuses the pair, rightly: the path starts 0.029 mm beyond T1
    (T1A < 0), which is SDP's own involute-interference test, Eq. (5-2) (margin -0.0014); since 255e7e7 the
    explicit Eq. (5-2) test of ``_internal_interference`` reports it a second time (two messages, José
    2026-10-24). The pinion z 16, x 0
    is undercut (x < x_Emin = 0.0641, ISO_21771_2007 Eq. (135)): the ideal gear says nothing, the same gear as
    cut is refused (ruling B, José 2026-10-21). SDP prints the table as a geometry example, not as a pair free of interference."""
    p = pair(bc.gear(3.0, 16), bc.gear(3.0, -24, x=-0.5), check=False)
    assert p.inv_alphatw == pytest.approx(0.060401, abs=5e-7)
    assert math.degrees(p.alphatw) == pytest.approx(31.09362, abs=5e-6)       # exact, from the data
    assert p.al == pytest.approx(13.1683, abs=5e-5)
    assert (p.gear1.db, p.gear2.db) == (pytest.approx(45.105, abs=5e-4), pytest.approx(67.658, abs=5e-4))
    assert (p.dl1, p.dl2) == (pytest.approx(52.673, abs=5e-4), pytest.approx(79.010, abs=5e-4))
    assert (p.da1_w, p.da2_w) == (pytest.approx(54.000, abs=5e-4), pytest.approx(69.000, abs=5e-4))
    assert (p.gear1.df, p.gear2.df) == (pytest.approx(40.500, abs=5e-4), pytest.approx(82.500, abs=5e-4))
    msgs = p.validate()
    assert_undercut_only_as_cut(p.gear1)
    assert len(msgs) == 2 and msgs[0].startswith("Interference: the path of contact passes a base-tangent point")
    assert msgs[1].startswith(INVOLUTE_INTERFERENCE)


# z1, x2 (KHK), alpha_wt [deg], a_w [mm], eps_alpha: m 1 mm, z2 = -50, x1 = 0
SDP_TABLE_5_4 = ((49, 1.00, 61.0605, 0.971, 1.105), (48, 0.60, 46.0324, 1.354, 1.512), (47, 0.40, 37.4155, 1.775, 1.726),
                 (46, 0.30, 32.4521, 2.227, 1.835), (45, 0.20, 28.2019, 2.666, 1.933), (44, 0.11, 24.5356, 3.099, 2.014),
                 (43, 0.06, 22.3755, 3.557, 2.053), (42, 0.01, 20.3854, 4.010, 2.088))


@pytest.mark.parametrize(("z1", "x2_khk", "alpha_wt_deg", "a_w", "eps_alpha"), SDP_TABLE_5_4)
def test_small_tooth_difference_of_sdp_table_5_4(z1, x2_khk, alpha_wt_deg, a_w, eps_alpha):
    """SDP Table 5-4 (internal spur, m 1 mm, 20 deg, z2 -50, z1 49..42, x1 0, x2_ISO = -x2_KHK), to the printed
    digits: alpha_wt (up to 61 deg: brentq bracket [1e-6, 1.5] rad), a_w, eps_alpha (Q22, Q26).

    SDP states that these pairs have no involute and no trochoid interference but do have trimming
    interference (they cannot be assembled radially): the messages of ``validate`` agree, and the pairs are
    built with check=False for that reason only. For z1 49..47 the arcsin argument of Eq. (5-10) exceeds 1
    (d_a1 > d_a2): no real value, counted as failed (``_internal_interference`` Notes).
    The ring of z1 = 49 (x2_KHK 1.00) has zero addendum, d_a = d, which the gear check refuses (Q55); it is
    built with check=False too."""
    ring = bc.gear(1.0, -50, x=-x2_khk, check=(x2_khk != 1.00))
    p = pair(bc.gear(1.0, z1), ring, check=False)
    assert math.degrees(p.alphatw) == pytest.approx(alpha_wt_deg, abs=5e-5)
    assert p.al == pytest.approx(a_w, abs=5e-4)
    assert p.epslon_alpha == pytest.approx(eps_alpha, abs=5e-4)
    msgs = p.validate()
    assert any(m.startswith("trimming interference of the internal pair") for m in msgs)
    assert not any(m.startswith("Interference:") or m.startswith("trochoid interference") for m in msgs)
    assert len(msgs) == 1 + (z1 == 49)     # trimming, and for z1 49 the ring tip of the gear check


# ----------------------------------------------------------------------------------------------
# former _LEGACY entries (removed in 255e7e7, José 2026-10-24): one rule for both kinds of teeth
# ----------------------------------------------------------------------------------------------
def test_virtual_teeth_of_both_kinds(case5):
    """Virtual number of teeth zn = z / cos(beta)**3, signed like z (negative for a ring), and
    de = mn abs(zn) [mm], for both kinds (was external only, Q25). External gear: ISO_6336-1_2019, 9.3.1,
    Formula (81) (unchecked); no library source for the sign convention of the ring."""
    for g in (case5.gear1, case5.gear2):
        assert g.zn == pytest.approx(g.z / math.cos(math.radians(15.0)) ** 3, rel=REL)
        assert g.de == pytest.approx(g.mn * abs(g.zn), rel=REL)
    assert case5.gear1.zn > 0.0 > case5.gear2.zn


def test_beta_range_of_both_kinds():
    """Q55 (José, 2026-10-15), extended to both kinds (José 2026-10-24): a negative helix angle is reported for an
    external gear and for a ring (the hand carries the direction). No library source."""
    for z in (60, -60):
        assert bc.gear(2.0, z, beta_n_deg=-5.0, hand=1, label="g", check=False).validate() == [
            "g: beta_n_deg out of range [0, 45), got -5.0"]


def test_gear_limits_of_both_kinds():
    """Q55 (José, 2026-10-15), one rule for both kinds (José 2026-10-24): the tip circle must lie on the tooth side
    of the reference circle (positive addendum (haP + x) mn) and above the base circle. The ideal layer has no
    undercut check (ruling B, José 2026-10-21), the manufactured gear refuses it. No library source."""
    g = bc.gear(2.0, 8, label="g")
    assert g.validate() == []
    assert_undercut_only_as_cut(g)
    assert bc.gear(2.0, -60, x=-2.0, label="g", check=False).validate() == [
        "g: tip circle da=124.0000 on the wrong side of the reference circle d=120.0000 "
        "(addendum (haP + x) mn = -2.0000 mm must be > 0)."]
    assert bc.gear(2.0, 60, x=-2.0, label="g", check=False).validate() == [
        "g: tip circle da=116.0000 on the wrong side of the reference circle d=120.0000 "
        "(addendum (haP + x) mn = -2.0000 mm must be > 0)."]


def test_contact_beyond_a_base_tangent_point_is_an_error(case4):
    """Q39: the path A..E must lie between the base-tangent points, T1A >= 0 and s2 T2E >= 0, for both kinds.
    Old case 4 starts 0.075 mm beyond T1 (for an internal pair this is SDP-SI Eq. (5-2): two messages since
    255e7e7, the T1A < 0 one of ``validate_pair`` and the explicit Eq. (5-2) test of ``_internal_interference``,
    José 2026-10-24); the external pair z 7/60 (undercut pinion) 2.87 mm beyond T1; the external pair z 12/12,
    x 0.6/-0.6 ends beyond T2 (T1A > 0, s2 T2E < 0). Every other pair of PAIRS validates to []."""
    msgs = case4.validate()
    assert msgs[0] == ("Interference: the path of contact passes a base-tangent point (T1A = -0.0749 mm, "
                       "s2 T2E = 25.1172 mm; both must be >= 0). For an internal pair T1A < 0 is the "
                       "involute interference of SDP-SI Eq. (5-2).")
    assert len(msgs) == 2 and msgs[1].startswith(INVOLUTE_INTERFERENCE)
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
        h = 1 if beta else 0
        p = pair(bc.gear(2.0, z1, x=x, beta_n_deg=beta, hand=h, check=False), bc.gear(2.0, -z2, beta_n_deg=beta, hand=h), check=False)
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


def test_one_pair_text_for_both_kinds(case2, case4):
    """Q50 (José, 2026-10-15): one text for external and internal pairs, named ``GearPair``; the kind is the
    sign of z2. ASCII only."""
    assert repr(case2).startswith("GearPair(z1=20, z2=30, mn=3.5, beta=15.0 deg, ")
    assert repr(case4).startswith("GearPair(z1=20, z2=-60, ")
    lines = case4.summary().splitlines()
    assert lines[1:3] == ["  gear1     : z=20, mn=2.0, x=0.0, beta=0.0 deg", "  gear2     : z=-60, mn=2.0, x=0.0, beta=0.0 deg"]
    assert len(lines) == len(case2.summary().splitlines())
    assert all(t.isascii() for t in (repr(case2), case2.summary(), repr(case4), case4.summary()))


def test_echoed_rows_are_counted():
    """Q03: the rows `driver` / `driven` of the pairs are the declared argument, not a result of the code."""
    cases = [p.stem for p in sorted(bc.DATA_DIR.glob("*.csv"))]
    names, count = bc.ECHOED_ROWS
    echoed = sum(1 for c in cases for obj, name, _ in bc.baseline_rows(c) if name in names and obj != "link")
    total = sum(len(bc.baseline_rows(c)) for c in cases)
    # 7212 rows: 7246 after block 4, less 18 deleted by Q13, 4 by Q46 and 12 undercutting() by ruling B (2026-10-21);
    # 7260: plus 48 values_at rows of the rings (s_yt, psi_y, eta_y, e_yt at 2 diameters, 6 cases; 6a61c79);
    # 7269: plus 9 validate() rows of the second SDP Eq. (5-2) message (case4 5, case6 4; 255e7e7)
    assert (total, total - echoed, echoed, count) == (7269, 7199, 70, 70)


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
    message; check=False builds the object (equivalence records). Old case 4: the Q39 / SDP Eq. (5-2) message and the
    explicit Eq. (5-2) message (two since 255e7e7, José 2026-10-24)."""
    with pytest.raises(ValueError) as err:
        pair(*bc.make_internal(0.0))
    assert str(err.value).splitlines() == case4.validate() and len(case4.validate()) == 2
    assert case4.T1A == pytest.approx(-0.07492456399709191, rel=1e-9)        # built with check=False
    with pytest.raises(ValueError, match="mn must be > 0"):
        bc.gear(-2.0, 8)
    assert_undercut_only_as_cut(bc.gear(2.0, 8))        # the z 8 gear assembles: no ideal undercut check


def test_assemble_gear_calls_no_clause_7_helper(monkeypatch):
    """Ruling B (José, 2026-10-21): the ideal layer does not use the generating tool. With every clause 7 helper
    and the tool record patched to raise, an undercut external gear, a ring and their pairs still assemble."""
    def forbidden(*args, **kwargs):
        raise AssertionError("clause 7 helper called by the ideal layer")

    for name in ("_min_generating_profile_shift", "_generating_profile_shift", "_generated_root_diameter",
                 "_root_form_diameter"):
        monkeypatch.setattr(ManufacturedCylindricalFamily, name, staticmethod(forbidden))
    monkeypatch.setattr(cylindrical.GeneratingRackProfile, "from_basic_rack", forbidden)
    pinion = bc.gear(2.0, 8)
    assert pinion.warnings == () and pinion.validate() == []
    pair(pinion, bc.gear(2.0, 60), check=False)
    pair(*bc.make_internal(0.0), check=False)


def test_ideal_family_has_none_of_the_manufactured_names():
    """Ruling (José, 2026-10-21): the manufactured family subclasses the ideal one and the dependency goes one way.
    ``CylindricalFamily`` has no name that ``ManufacturedCylindricalFamily`` adds (only ``assemble_gear``,
    ``validate_gear``, ``assemble_pair``, ``validate_pair`` are overridden), and the IDEAL section of the module
    (between its two section banners) names no manufactured class, record or field."""
    overridden = {"assemble_gear", "validate_gear", "assemble_pair", "validate_pair"}
    added = {n for n in vars(ManufacturedCylindricalFamily) if not n.startswith("__")} - overridden - {"_abc_impl"}
    assert {"from_ideal", "_min_generating_profile_shift", "span_measurement"} <= added
    assert not [n for n in added if hasattr(CylindricalFamily, n)]
    assert all(n in vars(CylindricalFamily) for n in overridden)
    for old in ("undercutting", "assemble_manufactured_gear", "assemble_manufactured_pair",
                "validate_manufactured_gear", "validate_manufactured_pair"):
        assert not hasattr(ManufacturedCylindricalFamily, old), old
    source = inspect.getsource(cylindrical)
    ideal = source[source.index("# IDEAL:"):source.index("# MANUFACTURED:")]
    for token in ("Manufactur", "GeneratingRackProfile", "GeneratingPinionCutter", "InspectionLimits", "cutter",
                  "x_Emin", "E_sn"):
        assert token not in ideal, token


def test_low_contact_ratio_is_a_warning_not_an_error():
    """Answer 6 (D-0026 item 9): 1 <= eps_alpha < 1.2 goes to pair.warnings and the pair assembles; eps_alpha < 1
    stays an error of validate(). Spur pair z 30/60, mn 2 mm, haP 0.6 (eps_alpha 1.082)."""
    p = pair(bc.gear(2.0, 30, b=10.0, haP=0.6), bc.gear(2.0, 60, b=10.0, haP=0.6))
    assert 1.0 <= p.epslon_alpha < 1.2 and p.validate() == []
    assert len(p.warnings) == 1 and p.warnings[0].startswith("transverse contact ratio")
    short = pair(bc.gear(2.0, 30, b=10.0, haP=0.4), bc.gear(2.0, 60, b=10.0, haP=0.4), check=False)
    assert short.epslon_alpha < 1.0 and any("< 1.0" in m for m in short.validate()) and short.warnings == ()


def test_gear_messages_appear_once_in_the_pair():
    """Q56 (José, 2026-10-15): each message of a gear once in the pair. The pinion z 8 is undercut, which the
    ideal layer does not check (ruling B, José 2026-10-21), so the gear message is a negative roughness."""
    low = pair(bc.gear(2.0, 8, b=10.0, Ra=-1.0, label="p", check=False), bc.gear(2.0, 60, b=10.0, label="w"), check=False)
    assert sum("Ra must be >= 0" in m for m in low.validate()) == 1
    assert_undercut_only_as_cut(low.gear1)


def test_henriot_split_is_deleted(case2):
    """Q13 (José, 2026-10-15): the Henriot split of the profile shifts is deleted; the pair takes the shifts of the
    gear data."""
    with pytest.raises(TypeError):
        pair(case2.gear1, case2.gear2, equalise_gs=True)
    assert not hasattr(FAMILY, "correction_for_gs_equilibrium") and not hasattr(cylindrical, "_henriot_shifts")
